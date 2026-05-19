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

from dataclasses import dataclass
from typing import TypeAlias

from ..config import TeleoperatorConfig


@dataclass
class SOLeaderConfig:
    """Base configuration class for SO Leader teleoperators."""

    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = False

    # --- 빠른 시작 및 최적화 옵션 (Fast-start & Optimization Options) ---
    
    # 캘리브레이션 파일 경로 (리더는 팔로워의 파일을 공유하거나 전용 파일을 사용 가능)
    calibration_path: str = "/home/roma/dual_arm/src/follower_calibration.json"
    
    # 모터 EEPROM에 캘리브레이션 값 쓰기 생략 (리더는 읽기 전용인 경우가 많아 True 권장)
    skip_write_calibration: bool = True
    
    # 최소 구성 모드 (토크를 끄고 읽기만 가능한 상태로 바로 시작)
    minimal_configure: bool = False
    
    # 운영 모드 및 프로파일 정보 쓰기 생략 (리더 모드 설정 시간을 줄임)
    skip_mode_write: bool = True
    
    # 시작 시 리더를 특정 위치로 이동시키는 과정 생략
    skip_home: bool = True
    
    # 원점 이동 시 대기 시간 (초 단위)
    home_wait: float = 2.0
    
    # 액션(Action) 데이터 읽기 시간 등의 디버그 로그 출력 여부
    action_debug: bool = False


@TeleoperatorConfig.register_subclass("so101_leader")
@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SOLeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    pass


SO100LeaderConfig: TypeAlias = SOLeaderTeleopConfig
SO101LeaderConfig: TypeAlias = SOLeaderTeleopConfig
