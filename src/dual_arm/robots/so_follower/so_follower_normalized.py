#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0

import json
import logging
import os
import time
from functools import cached_property
from pathlib import Path
from typing import TypeAlias

from dual_arm.cameras.utils import make_cameras_from_configs
from dual_arm.motors import Motor, MotorCalibration, MotorNormMode
from dual_arm.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from dual_arm.processor import RobotAction, RobotObservation
from dual_arm.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)


# =========================================================
# 속도 최적화 옵션
# =========================================================
FAST_CALIBRATION_PATH = os.getenv(
    "FAST_CALIBRATION_PATH",
    "/home/roma/dual_arm/src/follower_calibration.json",
)

FAST_SKIP_WRITE_CALIBRATION = os.getenv("FAST_SKIP_WRITE_CALIBRATION", "1") == "1"
FAST_SKIP_CAMERAS = os.getenv("FAST_SKIP_CAMERAS", "0") == "1"
FAST_SKIP_HOME = os.getenv("FAST_SKIP_HOME", "1") == "1"
FAST_MIN_CONFIGURE = os.getenv("FAST_MIN_CONFIGURE", "0") == "1"
FAST_SKIP_MODE_WRITE = os.getenv("FAST_SKIP_MODE_WRITE", "1") == "1"

FAST_HOME_WAIT = float(os.getenv("FAST_HOME_WAIT", "0.02"))

FAST_WORK_PROFILE_VELOCITY = int(os.getenv("FAST_WORK_PROFILE_VELOCITY", "1000"))
FAST_PROFILE_ACCELERATION = int(os.getenv("FAST_PROFILE_ACCELERATION", "20"))
FAST_HOME_PROFILE_VELOCITY = int(os.getenv("FAST_HOME_PROFILE_VELOCITY", "300"))

FAST_CAMERA_MAX_AGE_MS = int(os.getenv("FAST_CAMERA_MAX_AGE_MS", "200"))
FAST_OBS_SKIP_CAMERAS = os.getenv("FAST_OBS_SKIP_CAMERAS", "0") == "1"


class SOFollower(Robot):
    config_class = SOFollowerRobotConfig
    name = "so_follower"

    EXTENDED_POSITION_MOTOR_IDS = {1, 2, 3, 4}
    POSITION_MOTOR_IDS = {5, 6, 7}

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        motors = {
            "shoulder_pan": Motor(1, "xl430-w250", norm_mode_body),
            "shoulder_lift": Motor(2, "xl430-w250", norm_mode_body),
            "elbow_flex": Motor(3, "xl430-w250", norm_mode_body),
            "wrist_flex": Motor(4, "xl430-w250", norm_mode_body),
            "wrist_roll": Motor(5, "xl430-w250", norm_mode_body),
            "wrist_pitch": Motor(6, "xl430-w250", norm_mode_body),
            "gripper": Motor(7, "xl430-w250", norm_mode_body),
        }

        self.bus = DynamixelMotorsBus(
            port=config.port,
            motors=motors,
            calibration=self.calibration,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        # 반복 계산 줄이기용 캐시
        self._gear_ratios = config.gear_ratios
        self._motor_directions = config.motor_directions
        self._motor_ids = {name: motor.id for name, motor in self.bus.motors.items()}
        self._motor_names = tuple(self.bus.motors.keys())

        self._all_home_positions = {name: 2048 for name in self._motor_names}

        self._ext_mode_values = {
            name: OperatingMode.EXTENDED_POSITION.value
            for name, motor in self.bus.motors.items()
            if motor.id in self.EXTENDED_POSITION_MOTOR_IDS
        }

        self._pos_mode_values = {
            name: OperatingMode.POSITION.value
            for name, motor in self.bus.motors.items()
            if motor.id in self.POSITION_MOTOR_IDS
        }

        self._all_acc_values = {name: FAST_PROFILE_ACCELERATION for name in self._motor_names}
        self._home_vel_values = {name: FAST_HOME_PROFILE_VELOCITY for name in self._motor_names}
        self._work_vel_values = {name: FAST_WORK_PROFILE_VELOCITY for name in self._motor_names}

    def _is_ax12a(self, motor_name: str) -> bool:
        return False

    def _raw_to_norm_m100_100(self, motor_name: str, raw_val: int | float) -> float:
        """
        follower raw encoder position을 -100~100 정규화 값으로 변환.

        range_min -> -100
        range_max ->  100
        중간값    ->    0
        """
        cal = self.calibration[motor_name]

        raw_val = self._handle_overflow(raw_val)

        range_min = cal.range_min
        range_max = cal.range_max

        if range_max == range_min:
            return 0.0

        norm = (raw_val - range_min) / (range_max - range_min) * 200.0 - 100.0

        return max(-100.0, min(100.0, norm))

    def _norm_m100_100_to_raw(self, motor_name: str, norm_val: int | float) -> float:
        """
        -100~100 follower target action을 follower raw encoder position으로 역변환.

        -100 -> range_min
           0 -> 중간값
         100 -> range_max
        """
        cal = self.calibration[motor_name]

        norm_val = max(-100.0, min(100.0, float(norm_val)))

        range_min = cal.range_min
        range_max = cal.range_max

        raw = (norm_val + 100.0) / 200.0 * (range_max - range_min) + range_min

        return raw

    def _apply_gear_direction_correction(
        self,
        motor_name: str,
        raw_val: int | float,
    ) -> int:
        """
        teleop raw position을 follower target raw position으로 변환한다.

        보정 순서:
        1. 2048 중심 기준 delta 계산
        2. gear ratio 적용
        3. direction 적용
        4. 모터별 안전 범위 제한

        주의:
        - 이 함수는 teleop raw -> follower target raw 변환에만 사용한다.
        - send_action()에서는 다시 적용하지 않는다.
        """
        motor_id = self._motor_ids[motor_name]

        ratio = self._gear_ratios.get(motor_name, 1.0)
        direction = self._motor_directions.get(motor_name, 1.0)

        raw_val = self._handle_overflow(raw_val)

        corrected_val = 2048 + ((raw_val - 2048) * ratio * direction)

        if motor_id in self.POSITION_MOTOR_IDS:
            corrected_val = max(0, min(4095, corrected_val))

        if motor_name == "gripper":
            corrected_val = max(2500, min(4000, corrected_val))

        return int(corrected_val)

    def teleop_raw_to_follower_raw_action(self, action: RobotAction) -> RobotAction:
        """
        teleop에서 받은 raw action을 follower가 실제로 가야 할 raw target action으로 변환한다.

        입력:
            teleop raw action

        출력:
            follower target raw action

        여기서 gear ratio / direction 보정을 적용한다.
        """
        follower_raw_action = {}

        for key, val in action.items():
            if not key.endswith(".pos"):
                continue

            motor_name = key[:-4]

            if motor_name not in self._motor_ids:
                logger.warning(f"Unknown motor name in teleop raw action: {motor_name}")
                continue

            follower_target_raw = self._apply_gear_direction_correction(
                motor_name=motor_name,
                raw_val=val,
            )

            follower_raw_action[key] = float(follower_target_raw)

        return follower_raw_action

    def normalize_follower_raw_action(self, action: RobotAction) -> RobotAction:
        """
        follower target raw action을 dataset 저장용 -100~100 action으로 변환한다.

        입력:
            follower target raw action

        출력:
            follower target normalized action
        """
        norm_action = {}

        for key, val in action.items():
            if not key.endswith(".pos"):
                continue

            motor_name = key[:-4]

            if motor_name not in self._motor_ids:
                logger.warning(f"Unknown motor name in follower raw action normalization: {motor_name}")
                continue

            norm_action[key] = float(
                self._raw_to_norm_m100_100(motor_name, val)
            )

        return norm_action

    def send_raw_action(self, action: RobotAction) -> RobotAction:
        """
        follower target raw action을 그대로 Goal_Position으로 전송한다.

        중요:
        - 여기서는 gear ratio / direction을 다시 적용하지 않는다.
        - teleop_raw_to_follower_raw_action()에서 이미 보정된 값을 받는다고 가정한다.
        """
        goal_pos = {}

        for key, val in action.items():
            if not key.endswith(".pos"):
                continue

            motor_name = key[:-4]

            if motor_name not in self._motor_ids:
                logger.warning(f"Unknown motor name in raw action: {motor_name}")
                continue

            motor_id = self._motor_ids[motor_name]
            raw_val = self._handle_overflow(val)

            if motor_id in self.POSITION_MOTOR_IDS:
                raw_val = max(0, min(4095, raw_val))

            if motor_name == "gripper":
                raw_val = max(2500, min(4000, raw_val))

            goal_pos[motor_name] = int(raw_val)

        if goal_pos:
            try:
                self.bus.sync_write("Goal_Position", goal_pos, normalize=False)
            except Exception as e:
                logger.warning(f"모터 전송 에러: {e}")

        return {
            f"{motor_name}.pos": float(pos)
            for motor_name, pos in goal_pos.items()
        }

    def _handle_overflow(self, val: int | float) -> int | float:
        """
        XL430 extended position에서 32-bit overflow가 발생한 값을 보정.

        예:
            4294967196 -> -100

        주의:
            이 함수는 읽어온 값을 사람이 이해 가능한 signed 값으로 바꾸는 용도입니다.
            모터에 보낼 때 4294967296을 다시 더하면 안 됩니다.
            dynamixel.py 내부에서 two's complement 변환을 처리합니다.
        """
        if val > 2147483647:
            return val - 4294967296
        return val

    def _load_calibration_from_json(self) -> None:
        """
        기존 코드의 여러 경로 탐색을 제거하고, FAST_CALIBRATION_PATH를 우선 사용.
        write_calibration도 1회만 수행.
        """
        cal_path = Path(FAST_CALIBRATION_PATH).expanduser()

        if not cal_path.is_absolute():
            cal_path = cal_path.resolve()

        if not cal_path.exists():
            fallback_paths = (
                Path("/home/roma/dual_arm/src/follower_calibration.json"),
                Path("src/follower_calibration.json"),
                Path("follower_calibration.json"),
            )
            cal_path = next((p for p in fallback_paths if p.exists()), None)

        if cal_path is not None and cal_path.exists():
            with open(cal_path, "r") as f:
                data = json.load(f)

            self.calibration = {
                name: MotorCalibration(**val)
                for name, val in data.items()
            }
            logger.info(f"Loaded calibration directly from {cal_path}")
        else:
            logger.error("CRITICAL: calibration file NOT FOUND. Using fixed fallback values.")
            self.calibration = {
                name: MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=0,
                    range_max=4095,
                )
                for name, m in self.bus.motors.items()
            }

        self.bus.calibration = self.calibration

        if FAST_SKIP_WRITE_CALIBRATION:
            logger.info("Follower write_calibration skipped for fast start.")
            return

        try:
            self.bus.write_calibration(self.calibration, cache=True)
        except Exception as e:
            logger.warning(f"Calibration injected, but write_calibration failed ignored: {e!r}")

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (
                self.config.cameras[cam].height,
                self.config.cameras[cam].width,
                3,
            )
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self._motors_ft,
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        if FAST_SKIP_CAMERAS:
            return self.bus.is_connected
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = False, connect_cameras: bool = True) -> None:
        t0 = time.perf_counter()

        self.bus.connect()
        t_bus = time.perf_counter()

        if not self.is_calibrated and calibrate:
            logger.info("Calibration is missing or mismatched. Running calibration...")
            self.calibrate()
        elif not self.is_calibrated:
            logger.info("Calibration is missing. Loading from JSON.")
            self._load_calibration_from_json()
        t_cal = time.perf_counter()

        if connect_cameras and not FAST_SKIP_CAMERAS:
            for cam in self.cameras.values():
                cam.connect()
        t_cam = time.perf_counter()

        self.configure()
        t_cfg = time.perf_counter()

        logger.info(
            f"{self} connected. "
            f"timing: bus={t_bus - t0:.3f}s, "
            f"calib={t_cal - t_bus:.3f}s, "
            f"cams={t_cam - t_cal:.3f}s, "
            f"config={t_cfg - t_cam:.3f}s, "
            f"total={t_cfg - t0:.3f}s"
        )

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        self.bus.disable_torque()

        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )

            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")

        xl_motors = [n for n in self.bus.motors if not self._is_ax12a(n)]

        if xl_motors:
            for motor in xl_motors:
                motor_id = self.bus.motors[motor].id

                if motor_id in self.EXTENDED_POSITION_MOTOR_IDS:
                    self.bus.write(
                        "Operating_Mode",
                        motor,
                        OperatingMode.EXTENDED_POSITION.value,
                        normalize=False,
                    )
                else:
                    self.bus.write(
                        "Operating_Mode",
                        motor,
                        OperatingMode.POSITION.value,
                        normalize=False,
                    )

            input(f"Move {self} to the middle of its range of motion and press ENTER....")
            homing_offsets = self.bus.set_half_turn_homings(xl_motors)
        else:
            input(f"Move {self} to its rest position usually center and press ENTER....")
            homing_offsets = {motor: 0 for motor in self.bus.motors}

        for motor in self.bus.motors:
            if self._is_ax12a(motor):
                homing_offsets[motor] = 0

        print(
            "Move all joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )

        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets.get(motor, 0),
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """
        시작 시간 최적화 버전.

        FAST_MIN_CONFIGURE=1:
            configure 전체를 최소화하고 토크만 켬.
        FAST_SKIP_MODE_WRITE=1:
            Operating_Mode/Profile 설정을 생략하고 home/work velocity만 필요 최소 수준으로 적용.
        FAST_SKIP_HOME=1:
            2048 원점 이동을 생략.
        """
        if FAST_MIN_CONFIGURE:
            self.bus.enable_torque()
            logger.info(f"{self} minimal configuration complete.")
            return

        if not FAST_SKIP_MODE_WRITE:
            with self.bus.torque_disabled():
                if self._ext_mode_values:
                    self.bus.sync_write("Operating_Mode", self._ext_mode_values, normalize=False)
                if self._pos_mode_values:
                    self.bus.sync_write("Operating_Mode", self._pos_mode_values, normalize=False)

                self.bus.sync_write("Profile_Acceleration", self._all_acc_values, normalize=False)
                self.bus.sync_write("Profile_Velocity", self._home_vel_values, normalize=False)

        self.bus.enable_torque()

        if not FAST_SKIP_HOME:
            self.bus.sync_write("Goal_Position", self._all_home_positions, normalize=False)
            if FAST_HOME_WAIT > 0:
                time.sleep(FAST_HOME_WAIT)

        self.bus.sync_write("Profile_Velocity", self._work_vel_values, normalize=False)
        logger.info(f"{self} fast configuration complete.")

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        데이터 수집용 observation.

        raw position을 읽은 뒤 -100~100으로 정규화해서 저장한다.

        중요:
        - dataset observation은 follower 현재 상태 기준이다.
        - observation에는 gear ratio / direction을 적용하지 않는다.
        """
        raw_motor_positions = self.bus.sync_read("Present_Position", normalize=False)

        obs_dict = {}

        for motor_name, motor_val in raw_motor_positions.items():
            obs_dict[f"{motor_name}.pos"] = float(
                self._raw_to_norm_m100_100(motor_name, motor_val)
            )

        if not FAST_OBS_SKIP_CAMERAS:
            for cam_key, cam in self.cameras.items():
                obs_dict[cam_key] = cam.read_latest(max_age_ms=FAST_CAMERA_MAX_AGE_MS)

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        -100~100 follower target action을 raw로 역변환해서 모터에 전송한다.

        중요:
        - action은 이미 follower 기준 target이다.
        - 여기서는 gear ratio / direction을 적용하지 않는다.
        - gear ratio / direction은 teleop raw를 follower target raw로 바꾸는 단계에서만 적용한다.

        추론 시 policy output이 -100~100이면 이 함수를 그대로 사용하면 된다.
        """
        follower_raw_action = {}

        for key, val in action.items():
            if not key.endswith(".pos"):
                continue

            motor_name = key[:-4]

            if motor_name not in self._motor_ids:
                logger.warning(f"Unknown motor name in normalized action: {motor_name}")
                continue

            raw_val = self._norm_m100_100_to_raw(motor_name, val)
            follower_raw_action[key] = float(raw_val)

        return self.send_raw_action(follower_raw_action)
    
    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        if not FAST_SKIP_CAMERAS:
            for cam in self.cameras.values():
                cam.disconnect()
        logger.info(f"{self} disconnected.")


SO100Follower: TypeAlias = SOFollower
SO101Follower: TypeAlias = SOFollower