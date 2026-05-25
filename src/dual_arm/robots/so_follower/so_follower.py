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
# 속도 최적화 옵션 (Deprecated: Config 클래스로 이동됨)
# =========================================================

class SOFollower(Robot):
    config_class = SOFollowerRobotConfig
    name = "so_follower"

    EXTENDED_POSITION_MOTOR_IDS = {1, 2, 3, 4}
    POSITION_MOTOR_IDS = {5, 6, 7, 8}

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = (
            MotorNormMode.DEGREES
            if config.use_degrees
            else MotorNormMode.RANGE_M100_100
        )

        motors = {
            "shoulder_pan": Motor(1, "xl430-w250", norm_mode_body),
            "shoulder_lift": Motor(2, "xl430-w250", norm_mode_body),
            "elbow_pitch": Motor(3, "xl430-w250", norm_mode_body),
            "elbow_roll": Motor(4, "xl430-w250", norm_mode_body),
            "wrist_yaw": Motor(5, "xl430-w250", norm_mode_body),
            "wrist_roll": Motor(6, "xl430-w250", norm_mode_body),
            "wrist_pitch": Motor(7, "xl430-w250", norm_mode_body),
            "gripper": Motor(8, "xl430-w250", norm_mode_body),
        }

        self._follower_home_positions = {
            "shoulder_pan": 2388,
            "shoulder_lift": 2048,
            "elbow_pitch": 2048,
            "elbow_roll": 2048,
            "wrist_yaw": 2048,
            "wrist_roll": 2048,
            "wrist_pitch": 2958,
            "gripper": 2048,
        }

        self._leader_home_positions = {
            "shoulder_pan": 2728,
            "shoulder_lift": 2048,
            "elbow_pitch": 2048,
            "elbow_roll": 2958,
            "wrist_yaw": 2048,
            "wrist_roll": 2048,
            "wrist_pitch": 2958,
            "gripper": 2048,
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
        self._motor_ids = {
            name: motor.id
            for name, motor in self.bus.motors.items()
        }
        self._motor_names = tuple(self.bus.motors.keys())

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

        # config 값이 숫자일 수도 있고 dict일 수도 있으므로
        # 반드시 모터별 int 값으로 변환한다.
        self._all_acc_values = self._expand_motor_values(
            config.profile_accelerations,
            self._motor_names,
            200,
        )

        self._home_vel_values = self._expand_motor_values(
            config.home_profile_velocities,
            self._motor_names,
            1000,
        )

        self._work_vel_values = self._expand_motor_values(
            config.work_profile_velocities,
            self._motor_names,
            1000,
        )

        self._gripper_load_threshold = 650
        self.current_target_color = "none"

    def _is_ax12a(self, motor_name: str) -> bool:
        return False

    def _expand_motor_values(self, value, motor_names, default):
        """
        config 값이 숫자 하나이면 모든 모터에 같은 값을 적용.
        config 값이 dict이면 각 모터 이름에 해당하는 값을 적용.

        예:
        value = 200
        -> {"shoulder_pan": 200, ...}

        value = {"shoulder_pan": 200, "elbow_pitch": 150}
        -> 없는 모터는 default 사용.
        """
        if isinstance(value, dict):
            return {
                name: int(value.get(name, default))
                for name in motor_names
            }

        return {
            name: int(value)
            for name in motor_names
        }

    def _is_gripper_overload_risk(self) -> bool:
        """
        gripper 모터의 부하/전류를 읽어서 overload 위험 여부를 판단한다.

        True  = overload 위험 있음
        False = 정상
        """
        try:
            load_dict = self.bus.sync_read("Present_Load", normalize=False)
            gripper_load = load_dict.get("gripper", 0)

            gripper_load = self._handle_overflow(gripper_load)
            gripper_load = abs(float(gripper_load))

            return gripper_load >= self._gripper_load_threshold

        except Exception as e:
            logger.warning(f"gripper load read failed: {e}")
            return False

    def _handle_overflow(self, val: int | float) -> int | float:
        """
        XL430 extended position에서 32-bit overflow가 발생한 값을 보정.
        """
        if val > 2147483647:
            return val - 4294967296
        return val

    def _load_calibration_from_json(self) -> None:
        """
        기존 코드의 여러 경로 탐색을 제거하고,
        config.calibration_path를 우선 사용.
        """
        cal_path = Path(self.config.calibration_path).expanduser()

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
            logger.error(
                "CRITICAL: calibration file NOT FOUND. "
                "Using fixed fallback values."
            )
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

        if self.config.skip_write_calibration:
            logger.info("Follower write_calibration skipped for fast start.")
            return

        try:
            self.bus.write_calibration(self.calibration, cache=True)
        except Exception as e:
            logger.warning(
                f"Calibration injected, but write_calibration failed ignored: {e!r}"
            )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float
            for motor in self.bus.motors
        }

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
        if self.config.skip_cameras:
            return self.bus.is_connected

        return self.bus.is_connected and all(
            cam.is_connected
            for cam in self.cameras.values()
        )

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

        if connect_cameras and not self.config.skip_cameras:
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
                logger.info(
                    f"Writing calibration file associated with the id {self.id} "
                    "to the motors"
                )
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")

        xl_motors = [
            n for n in self.bus.motors
            if not self._is_ax12a(n)
        ]

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
            homing_offsets = {
                motor: 0
                for motor in self.bus.motors
            }

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
        모터 동작 모드, 가속도, 속도 설정.

        주의:
        - Profile_Acceleration, Profile_Velocity에는 반드시 int 값이 들어가야 한다.
        - dict 전체가 들어가면 motors_bus.py 내부 int(value)에서 TypeError 발생.
        """
        with self.bus.torque_disabled():
            self.bus.configure_motors()

            for motor in self.bus.motors:
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

                if motor == "shoulder_pan":
                    # PID 제어 값 임의 설정 (XL430-W250 기준)
                    self.bus.write("Position_P_Gain", motor, 2000, normalize=False)
                    self.bus.write("Position_I_Gain", motor, 0, normalize=False)
                    self.bus.write("Position_D_Gain", motor, 3600, normalize=False)
                    logger.info(f"[PID SET] {motor}: P=2000, I=0, D=3600")

                self.bus.write(
                    "Profile_Acceleration",
                    motor,
                    self._all_acc_values[motor],
                    normalize=False,
                )

                self.bus.write(
                    "Profile_Velocity",
                    motor,
                    self._work_vel_values[motor],
                    normalize=False,
                )

        self.bus.enable_torque()

        for motor in self.bus.motors:
            try:
                mode = self.bus.read("Operating_Mode", motor, normalize=False)
                logger.info(f"[MODE CHECK] {motor}: Operating_Mode={mode}")
            except Exception as e:
                logger.warning(f"[MODE CHECK] {motor} read failed: {e}")

        logger.info(f"{self} configuration complete.")

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(
                f"Connect the controller board to the '{motor}' motor only "
                "and press enter."
            )
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        데이터 수집용 observation.
        """
        raw_motor_positions = self.bus.sync_read(
            "Present_Position",
            normalize=False,
        )

        obs_dict = {}

        for motor_name, motor_val in raw_motor_positions.items():
            motor_val = self._handle_overflow(motor_val)
            obs_dict[f"{motor_name}.pos"] = float(motor_val)

        if not self.config.obs_skip_cameras:
            for cam_key, cam in self.cameras.items():
                obs_dict[cam_key] = cam.read_latest(
                    max_age_ms=self.config.camera_max_age_ms
                )

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        raw action 기준으로 모터에 명령을 보낸다.

        처리 순서:
        1. action에서 .pos 명령만 사용
        2. action 값은 raw position이라고 가정
        3. 2048 중심 기준으로 gear ratio와 direction 적용
        4. position 모터는 0~4095로 제한
        5. gripper는 2800~4000으로 제한
        6. Goal_Position으로 전송

        주의:
        - action 값에 -100~100 정규화 값을 넣으면 안 된다.
        - observation도 raw, action도 raw 기준으로 맞춘다.
        """
        corrected_goal_pos = {}

        gear_ratios = self._gear_ratios
        motor_directions = self._motor_directions
        motor_ids = self._motor_ids

        for key, val in action.items():
            if not key.endswith(".pos"):
                continue

            motor_name = key[:-4]

            if motor_name not in motor_ids:
                logger.warning(f"Unknown motor name in action: {motor_name}")
                continue

            motor_id = motor_ids[motor_name]
            ratio = gear_ratios.get(motor_name, 1.0)
            direction = motor_directions.get(motor_name, 1.0)

            raw_val = self._handle_overflow(val)

            home_pos = self._leader_home_positions.get(motor_name, 2048)
            init_follow_pos = self._follower_home_positions.get(motor_name, 2048)

            corrected_val = init_follow_pos + (
                (raw_val - home_pos) * ratio * direction
            )

            if motor_id in self.POSITION_MOTOR_IDS:
                corrected_val = max(0, min(4095, corrected_val))

            if motor_name == "wrist_pitch":
                corrected_val = max(1080, min(3120, corrected_val))

            if motor_name == "gripper":
                # 타겟 색상에 따른 동적 그리퍼 범위 설정
                if self.current_target_color == "green":
                    if corrected_val >3300:
                        corrected_val = 3869
                    corrected_val = max(3050, min(4000, corrected_val))
                elif self.current_target_color == "red":
                    if corrected_val >3500:
                        corrected_val = 3869
                    corrected_val = max(3200, min(4000, corrected_val))
                elif self.current_target_color == "blue":
                    corrected_val = max(2770, min(4000, corrected_val))
                else:
                    # 기본 범위 (none 또는 기타)
                    corrected_val = max(2600, min(4000, corrected_val))

            corrected_goal_pos[motor_name] = int(corrected_val)

        if corrected_goal_pos:
            try:
                self.bus.sync_write(
                    "Goal_Position",
                    corrected_goal_pos,
                    normalize=False,
                )
            except Exception as e:
                logger.warning(f"모터 전송 에러: {e}")

        return {
            f"{motor_name}.pos": float(pos)
            for motor_name, pos in corrected_goal_pos.items()
        }

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        if not self.config.skip_cameras:
            for cam in self.cameras.values():
                cam.disconnect()

        logger.info(f"{self} disconnected.")


SO100Follower: TypeAlias = SOFollower#
SO101Follower: TypeAlias = SOFollower