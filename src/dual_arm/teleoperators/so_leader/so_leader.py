#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0

import json
import logging
import os
import time
from pathlib import Path
from typing import TypeAlias

from dual_arm.motors import Motor, MotorCalibration, MotorNormMode
from dual_arm.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from dual_arm.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_so_leader import SOLeaderTeleopConfig

logger = logging.getLogger(__name__)


# =========================================================
# Leader fast-start options (Deprecated: Config 클래스로 이동됨)
# =========================================================

class SOLeader(Teleoperator):
    """Generic SO leader base for SO-100/101/10X teleoperators."""

    config_class = SOLeaderTeleopConfig
    name = "so_leader"

    def __init__(self, config: SOLeaderTeleopConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        motors = {
            "shoulder_pan": Motor(21, "xl430-w250", norm_mode_body),
            "shoulder_lift": Motor(22, "xl430-w250", norm_mode_body),
            "elbow_pitch": Motor(23, "xl430-w250", norm_mode_body),
            "elbow_roll": Motor(24, "xl430-w250", norm_mode_body),
            "wrist_yaw": Motor(25, "xl430-w250", norm_mode_body),
            "wrist_roll": Motor(26, "xl430-w250", norm_mode_body),
            "wrist_pitch": Motor(27, "xl430-w250", norm_mode_body),
            "gripper": Motor(28, "xl430-w250", norm_mode_body),
        }

        self.bus = DynamixelMotorsBus(
            port=config.port,
            motors=motors,
            calibration=self.calibration,
        )

        self._motor_names = tuple(self.bus.motors.keys())
        self._all_position_mode = {name: OperatingMode.POSITION.value for name in self._motor_names}
        self._all_acc_values = {name: 10 for name in self._motor_names}
        self._all_vel_values = {name: 50 for name in self._motor_names}
        self._all_home_positions ={
        "shoulder_pan": 2728, 
        "shoulder_lift": 2048, 
        "elbow_pitch": 2048,
        "elbow_roll": 2958, 
        "wrist_yaw": 2048, 
        "wrist_roll": 2048, 
        "wrist_pitch": 2958, 
        "gripper": 2048,
    }
    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def _is_ax12a(self, motor_name: str) -> bool:
        return False

    def _load_calibration_from_json(self) -> None:
        """
        calibration 파일을 직접 경로로 읽어서 leader connect 시간을 줄임.
        """
        cal_path = Path(self.config.calibration_path).expanduser()
        if not cal_path.is_absolute():
            cal_path = cal_path.resolve()

        if not cal_path.exists():
            fallback_paths = (
                Path("/home/roma/dual_arm/src/leader_calibration.json"),
                Path("/home/roma/dual_arm/src/follower_calibration.json"),
                Path("src/leader_calibration.json"),
                Path("src/follower_calibration.json"),
                Path("leader_calibration.json"),
                Path("follower_calibration.json"),
            )
            cal_path = next((p for p in fallback_paths if p.exists()), None)

        if cal_path is None or not cal_path.exists():
            logger.warning("Leader calibration file NOT FOUND. Falling back to fixed values.")
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
        else:
            with open(cal_path, "r") as f:
                data = json.load(f)

            self.calibration = {
                name: MotorCalibration(**val)
                for name, val in data.items()
            }
            logger.info(f"Loaded leader calibration directly from {cal_path}")

        self.bus.calibration = self.calibration

        if self.config.skip_write_calibration:
            logger.info("Leader write_calibration skipped for fast start.")
            return

        try:
            self.bus.write_calibration(self.calibration, cache=True)
        except Exception as e:
            logger.warning(f"Leader calibration injected, but write_calibration failed ignored: {e!r}")

    @check_if_already_connected
    def connect(self, calibrate: bool = False) -> None:
        """Connect to the leader as fast as possible."""
        t0 = time.perf_counter()

        self.bus.connect()
        t_bus = time.perf_counter()

        if not self.is_calibrated and calibrate:
            logger.info("Leader calibration is missing or mismatched. Running calibration...")
            self.calibrate()
        elif not self.is_calibrated:
            logger.info("Leader calibration is missing. Loading from JSON.")
            self._load_calibration_from_json()
        t_cal = time.perf_counter()

        self.configure()
        t_cfg = time.perf_counter()

        logger.info(
            f"{self} connected. "
            f"timing: bus={t_bus - t0:.3f}s, "
            f"calib={t_cal - t_bus:.3f}s, "
            f"config={t_cfg - t_cal:.3f}s, "
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

        xl_motors = [n for n, m in self.bus.motors.items() if not self._is_ax12a(n)]
        if xl_motors:
            for motor in xl_motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
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
        leader는 텔레옵 입력을 읽기만 하면 되므로 가장 빠른 기본값은 토크 OFF만 보장.
        """
        if self.config.minimal_configure:
            self.bus.disable_torque()
            logger.info(f"{self} minimal configuration complete. Torque is OFF.")
            return

        # 리더 초기위치로 이동시에 적용
        if not self.config.skip_mode_write:
            # 모터 설정 값 변경
            with self.bus.torque_disabled():
                self.bus.sync_write("Operating_Mode", self._all_position_mode, normalize=False)
                self.bus.sync_write("Profile_Acceleration", self._all_acc_values, normalize=False)
                self.bus.sync_write("Profile_Velocity", self._all_vel_values, normalize=False)
        
        # 모터 목표 위치로 이동
        if not self.config.skip_home:
            self.bus.enable_torque()
            self.bus.sync_write("Goal_Position", self._all_home_positions, normalize=False)
            if self.config.home_wait > 0:
                time.sleep(self.config.home_wait)

        self.bus.disable_torque()
        logger.info(f"{self} fast configuration complete. Torque is OFF.")

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        if self.config.action_debug:
            start = time.perf_counter()

        raw = self.bus.sync_read("Present_Position", normalize=False)
        action = {f"{motor}.pos": float(val) for motor, val in raw.items()}

        if self.config.action_debug:
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")


SO100Leader: TypeAlias = SOLeader
SO101Leader: TypeAlias = SOLeader