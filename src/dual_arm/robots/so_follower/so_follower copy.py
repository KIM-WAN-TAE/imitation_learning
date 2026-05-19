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
# calibration json 절대경로를 지정하면 경로 탐색을 하지 않음.
FAST_CALIBRATION_PATH = os.getenv(
    "FAST_CALIBRATION_PATH",
    "/home/roma/dual_arm/src/follower_calibration.json",
)

# 1이면 calibration을 bus에 주입만 하고 write_calibration()은 생략. 시작 속도 최우선.
FAST_SKIP_WRITE_CALIBRATION = os.getenv("FAST_SKIP_WRITE_CALIBRATION", "1") == "1"

# 1이면 connect()에서 카메라 연결 생략. 텔레옵만 빠르게 확인할 때 사용.
FAST_SKIP_CAMERAS = os.getenv("FAST_SKIP_CAMERAS", "0") == "1"

# 1이면 configure()에서 원점 복귀 생략. 이미 위치가 맞아 있을 때만 사용.
FAST_SKIP_HOME = os.getenv("FAST_SKIP_HOME", "1") == "1"

# 1이면 configure() 자체를 최소화. 이미 모터 모드/속도 설정이 되어 있을 때 가장 빠름.
FAST_MIN_CONFIGURE = os.getenv("FAST_MIN_CONFIGURE", "0") == "1"

# 1이면 매번 Operating_Mode/Profile 값을 쓰지 않음. 이미 설정된 경우 시작 시간 단축.
FAST_SKIP_MODE_WRITE = os.getenv("FAST_SKIP_MODE_WRITE", "1") == "1"

# 홈 이동 대기 시간. 기존 0.5초에서 기본 0.05초로 단축.
FAST_HOME_WAIT = float(os.getenv("FAST_HOME_WAIT", "0.02"))

# 작업 속도 설정값.
FAST_WORK_PROFILE_VELOCITY = int(os.getenv("FAST_WORK_PROFILE_VELOCITY", "1000"))
FAST_PROFILE_ACCELERATION = int(os.getenv("FAST_PROFILE_ACCELERATION", "20"))
FAST_HOME_PROFILE_VELOCITY = int(os.getenv("FAST_HOME_PROFILE_VELOCITY", "300"))

# 카메라 프레임 대기 시간. 기존 5000ms는 너무 김.
FAST_CAMERA_MAX_AGE_MS = int(os.getenv("FAST_CAMERA_MAX_AGE_MS", "200"))

# 관찰에서 카메라 읽기를 생략. dataset 영상이 필요 없고 텔레옵 반응성만 볼 때 사용.
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

    def _load_calibration_from_json(self) -> None:
        """
        기존 코드의 여러 경로 탐색을 제거하고, FAST_CALIBRATION_PATH를 우선 사용.
        write_calibration도 1회만 수행.
        """
        cal_path = Path(FAST_CALIBRATION_PATH).expanduser()

        if not cal_path.is_absolute():
            cal_path = cal_path.resolve()

        if not cal_path.exists():
            # fallback은 최소화하되, 기존 호환성은 유지
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
        raw_motor_positions = self.bus.sync_read("Present_Position", normalize=False)

        obs_dict = {}
        def handle_overflow(val: int | float) -> int | float:
            """XL430 32-bit signed integer overflow 보정."""
            if val > 2147483647:
                return val - 4294967296
            return val

        for motor_name, motor_val in raw_motor_positions.items():
            motor_val = handle_overflow(motor_val)
            obs_dict[f"{motor_name}.pos"] = motor_val

        if not FAST_OBS_SKIP_CAMERAS:
            for cam_key, cam in self.cameras.items():
                obs_dict[cam_key] = cam.read_latest(max_age_ms=FAST_CAMERA_MAX_AGE_MS)

        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        corrected_goal_pos = {}
        gear_ratios = self._gear_ratios
        motor_directions = self._motor_directions
        motor_ids = self._motor_ids

        for key, val in action.items():
            if not key.endswith(".pos"):
                continue

            motor_name = key[:-4]
            motor_id = motor_ids[motor_name]
            ratio = gear_ratios.get(motor_name, 1.0)
            direction = motor_directions.get(motor_name, 1.0)

            corrected_val = 2048 + ((val - 2048) * ratio * direction)

            if motor_id in self.POSITION_MOTOR_IDS:
                corrected_val = 0 if corrected_val < 0 else 4095 if corrected_val > 4095 else corrected_val

            if motor_name == "gripper":
                corrected_val = 2500 if corrected_val < 2500 else 4000 if corrected_val > 4000 else corrected_val

            corrected_goal_pos[motor_name] = int(corrected_val)

        if corrected_goal_pos:
            try:
                self.bus.sync_write("Goal_Position", corrected_goal_pos, normalize=False)
            except Exception as e:
                logger.warning(f"모터 전송 에러: {e}")

        return {key: val for key, val in action.items() if key.endswith(".pos")}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        if not FAST_SKIP_CAMERAS:
            for cam in self.cameras.values():
                cam.disconnect()
        logger.info(f"{self} disconnected.")


SO100Follower: TypeAlias = SOFollower
SO101Follower: TypeAlias = SOFollower