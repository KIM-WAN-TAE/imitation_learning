#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import TypeAlias

from dual_arm.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = False

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


    # 감속비 매칭 (J1: 15, J2: 15, J3: 5, J4: 9)
    gear_ratios: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 15.0,
        "shoulder_lift": 15.0,
        "elbow_pitch": 9.0,
        "elbow_roll": 9.0,
        "wrist_yaw": 1.0,
        "wrist_roll": 1.0,
        "wrist_pitch": 1.0,
        "gripper": 3.0,
    })


    # 방향 매칭 (제공된 코드의 DIRECTION_MAP 반영)
    motor_directions: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": -1.0,
        "shoulder_lift": -1.0,
        "elbow_pitch": -1.0,
        "elbow_roll": -1.0,
        "wrist_yaw": 1.0,
        "wrist_roll": 1.0,
        "wrist_pitch": 1.0,
        "gripper": 1.0,
    })

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # --- 빠른 시작 및 최적화 옵션 (Fast-start & Optimization Options) ---
    
    # 캘리브레이션 파일 경로 (모터의 각도 오프셋 및 가동 범위 설정 파일)
    calibration_path: str = "/home/roma/dual_arm/src/follower_calibration.json"
    
    # 모터 EEPROM에 캘리브레이션 값 쓰기 생략 (이미 설정된 경우 시간을 단축하기 위해 True 권장)
    skip_write_calibration: bool = True
    
    # 카메라 초기화 생략 (하드웨어 테스트 시 시간을 아끼기 위해 사용)
    skip_cameras: bool = False
    
    # 시작 시 2048(원점) 위치로 이동하는 과정 생략
    skip_home: bool = True
    
    # 최소 구성 모드 (운영 모드나 프로파일 설정 없이 토크만 켬, 가장 빠른 시작)
    minimal_configure: bool = False
    
    # 운영 모드(Operating Mode) 및 프로파일(Profile) 정보 쓰기 생략
    skip_mode_write: bool = True
    
    # 원점 이동 후 대기 시간 (초 단위)
    home_wait: float = 0.02
    
    # # 모터 동작 프로파일 설정
    # work_profile_velocity: int = 1000   # 작업 시 속도
    # profile_acceleration: int = 1     # 가속도
    # home_profile_velocity: int = 300   # 원점 복귀 시 속도



    profile_accelerations: dict[str, int] = field(default_factory=lambda: {
        "shoulder_pan": 400,
        "shoulder_lift": 400,
        "elbow_pitch": 300,
        "elbow_roll": 300,
        "wrist_yaw": 120,
        "wrist_roll": 120,
        "wrist_pitch": 120,
        "gripper": 50,
    })


    work_profile_velocities: dict[str, int] = field(default_factory=lambda: {
        "shoulder_pan": 2800,
        "shoulder_lift": 2800,
        "elbow_pitch": 2000,
        "elbow_roll": 2000,
        "wrist_yaw": 550,
        "wrist_roll": 550,
        "wrist_pitch": 550,
        "gripper": 150,
    })


    home_profile_velocities: dict[str, int] = field(default_factory=lambda: {
        "shoulder_pan": 800,
        "shoulder_lift": 800,
        "elbow_pitch": 700,
        "elbow_roll": 700,
        "wrist_yaw": 300,
        "wrist_roll": 300,
        "wrist_pitch": 300,
        "gripper": 150,
    })


        # 관측(Observation) 데이터 설정
    camera_max_age_ms: int = 200       # 카메라 프레임의 최대 허용 지연 시간 (밀리초)
    obs_skip_cameras: bool = False     # 관측 데이터에서 카메라 이미지 제외 여부


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    pass


SO100FollowerConfig: TypeAlias = SOFollowerRobotConfig
SO101FollowerConfig: TypeAlias = SOFollowerRobotConfig

