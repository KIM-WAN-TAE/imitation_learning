# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Records a dataset.
"""

import logging
import os
import select
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any
from collections import deque
from pynput import keyboard

from dual_arm.cameras import (  # noqa: F401
    CameraConfig,
)
from dual_arm.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from dual_arm.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from dual_arm.configs import parser
from dual_arm.configs.policies import PreTrainedConfig
from dual_arm.datasets.image_writer import safe_stop_image_writer
from dual_arm.datasets.lerobot_dataset import LeRobotDataset
from dual_arm.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from dual_arm.datasets.utils import build_dataset_frame, combine_feature_dicts
from dual_arm.datasets.video_utils import VideoEncodingManager
from dual_arm.policies.factory import make_policy, make_pre_post_processors
from dual_arm.policies.pretrained import PreTrainedPolicy
from dual_arm.policies.utils import make_robot_action
from dual_arm.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from dual_arm.processor.rename_processor import rename_stats
from dual_arm.robots import Robot, RobotConfig, make_robot_from_config
from dual_arm.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from dual_arm.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from dual_arm.utils.constants import ACTION, OBS_STR
from dual_arm.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from dual_arm.utils.import_utils import register_third_party_plugins
from dual_arm.utils.robot_utils import precise_sleep
from dual_arm.utils.utils import get_safe_torch_device, init_logging, log_say
from dual_arm.utils.visualization_utils import init_rerun, log_rerun_data
from dual_arm.scripts.calibrate_origin_keyboard import calibrate_origin

# ==========================================
# 설정 클래스 정의
# ==========================================

@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 20
    episode_time_s: int | float = 500
    reset_time_s: int | float = 10
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "libsvtav1"
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")

@dataclass
class StartupConfig:
    """하드웨어 초기화 및 최적화 설정"""
    skip_home_sync: bool = False
    fast_silent_mode: bool = True
    home_tolerance: int = 60
    sync_duration: float = 1.5
    sync_fps: int = 20
    sync_settle_time: float = 0.5
    parallel_connect: bool = True
    overlap_connect: bool = True
    connect_cameras: bool = True
    teleop_start_threshold: float = 3.0
    enable_gripper_offset: bool = True

@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    startup: StartupConfig = field(default_factory=StartupConfig)
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False

# ==========================================
# 상수 및 헬퍼 함수
# ==========================================

FOLLOWER_TARGET_POSITIONS = {
    "shoulder_pan": 2388, "shoulder_lift": 2048, "elbow_pitch": 2048,
    "elbow_roll": 2048, "wrist_yaw": 2048, "wrist_roll": 2048, "wrist_pitch": 2958, "gripper": 2048,
}

LEADER_TARGET_POSITIONS = {
    "shoulder_pan": 2728, "shoulder_lift": 2048, "elbow_pitch": 2048,
    "elbow_roll": 2958, "wrist_yaw": 2048, "wrist_roll": 2048, "wrist_pitch": 2958, "gripper": 2048,
}

def handle_overflow(val: int | float) -> int | float:
    if val > 2147483647:
        return val - 4294967296
    return val

def say_fast(text: str, play_sounds: bool, startup_cfg: StartupConfig):
    if startup_cfg.fast_silent_mode:
        logging.info(text)
    else:
        log_say(text, play_sounds)

class TeleopGate:
    def __init__(self, threshold: float, enable_gripper_offset: bool):
        self.threshold = threshold
        self.baseline = None
        self.enabled = False

    def process(self, act: dict, obs: dict) -> dict:
        if self.baseline is None:
            self.baseline = dict(act)
            return {k: v for k, v in obs.items() if k.endswith(".pos")}
        
        if not self.enabled:
            for k, v in act.items():
                if k != "gripper.pos" and k in self.baseline:
                    if abs(float(v) - float(self.baseline[k])) > self.threshold:
                        self.enabled = True
                        logging.info("[Teleop Gate] Leader moved. Teleop Enabled.")
                        break
            if not self.enabled:
                return {k: v for k, v in obs.items() if k.endswith(".pos")}
        return act

# ==========================================
# 동기화 및 제어 함수
# ==========================================

def move_bus_to_target(bus, target_positions: dict, name: str, startup_cfg: StartupConfig):
    raw = bus.sync_read("Present_Position", normalize=False)
    start_vals = {m: handle_overflow(raw.get(m, 2048)) for m in bus.motors}
    deltas = {m: (target_positions.get(m, 2048) - start_vals[m]) for m in bus.motors}

    steps = max(1, int(startup_cfg.sync_fps * startup_cfg.sync_duration))
    sleep_dt = 1.0 / max(1, startup_cfg.sync_fps)

    for i in range(1, steps + 1):
        alpha = i / steps
        goal = {m: int(start_vals[m] + deltas[m] * alpha) for m in bus.motors}
        bus.sync_write("Goal_Position", goal, normalize=False)
        time.sleep(sleep_dt)

def torque_off_if_possible(teleop: Teleoperator | None):
    if teleop and hasattr(teleop, "bus"):
        try:
            teleop.bus.disable_torque()
        except:
            pass

def fast_sync_to_target(robot, teleop, startup_cfg):
    if startup_cfg.skip_home_sync:
        torque_off_if_possible(teleop)
        return

    robot.bus.enable_torque()
    if teleop and hasattr(teleop, "bus"):
        teleop.bus.enable_torque()

    move_bus_to_target(robot.bus, FOLLOWER_TARGET_POSITIONS, "follower", startup_cfg)
    if teleop and hasattr(teleop, "bus"):
        move_bus_to_target(teleop.bus, LEADER_TARGET_POSITIONS, "leader", startup_cfg)
    
    time.sleep(startup_cfg.sync_settle_time)
    torque_off_if_possible(teleop)

def reset_follower_before_exit(robot: Robot, startup_cfg: StartupConfig):
    """종료 전 팔로워를 초기 위치로 이동하고 토크를 켭니다."""
    if not (robot and getattr(robot, "is_connected", False) and hasattr(robot, "bus")):
        return
    try:
        robot.bus.enable_torque()
        print("[Exit Sync] 복귀 중...")
        move_bus_to_target(robot.bus, FOLLOWER_TARGET_POSITIONS, "exit", startup_cfg)
        robot.bus.enable_torque()  # 토크 유지
        print("[Exit Sync] 완료 (토크 유지)")
    except:
        pass

# ==========================================
# 메인 루프 및 녹화
# ==========================================

@safe_stop_image_writer
def record_loop(
    robot,
    events,
    fps,
    teleop_proc,
    robot_act_proc,
    robot_obs_proc,
    startup_cfg,
    dataset=None,
    teleop=None,
    policy=None,
    preprocessor=None,
    postprocessor=None,
    control_time_s=None,
    single_task=None,
    display_data=False,
    display_compressed_images=False,
):
    gate = TeleopGate(startup_cfg.teleop_start_threshold, startup_cfg.enable_gripper_offset)
    target_dt = 1.0 / fps
    loop_count = 0
    start_t = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as executor:
        while (loop_count / fps) < control_time_s:
            loop_start = time.perf_counter()
            if events["stop_recording"]:
                break

            # 데이터 취득
            if policy is None and isinstance(teleop, Teleoperator):
                f_obs = executor.submit(robot.get_observation)
                f_act = executor.submit(teleop.get_action)
                obs = f_obs.result()
                act = f_act.result()
            else:
                obs = robot.get_observation()
                act = None  # 복합 텔레옵 로직 생략(필요시 추가)

            obs_proc = robot_obs_proc(obs)
            obs_frame = build_dataset_frame(dataset.features, obs_proc, prefix=OBS_STR) if dataset else None

            # 액션 처리
            if policy:
                action = predict_action(
                    obs_frame,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                )
                action = make_robot_action(action, dataset.features)
            elif act:
                action = teleop_proc((gate.process(act, obs), obs))
            else:
                continue

            robot_action_to_send = robot_act_proc((action, obs))
            sent_action = robot.send_action(robot_action_to_send)

            # send_action()이 None 또는 빈 dict를 반환하면 기존 action을 저장
            action_to_record = sent_action if sent_action else action

            # ==========================================
            # 데이터셋 저장 방식 수정 부분
            # - 기존 코드 구조는 유지
            # - obs_frame/action_frame이 None이면 프레임 저장을 건너뜀
            # - NoneType mapping 오류 방지
            # ==========================================
            if dataset is not None:
                if obs_frame is None:
                    logging.warning(
                        "[Record] obs_frame is None. Skip this frame. "
                        f"obs_proc keys={list(obs_proc.keys()) if isinstance(obs_proc, dict) else type(obs_proc)}"
                    )
                    continue

                action_frame = build_dataset_frame(dataset.features, action_to_record, prefix=ACTION)

                if action_frame is None:
                    logging.warning(
                        "[Record] action_frame is None. Skip this frame. "
                        f"action_to_record keys={list(action_to_record.keys()) if isinstance(action_to_record, dict) else type(action_to_record)}"
                    )
                    continue

                frame = {**obs_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)

            if display_data:
                log_rerun_data(
                    observation=obs_proc,
                    action=action_to_record,
                    compress_images=display_compressed_images,
                )

            loop_count += 1
            precise_sleep(max(0, target_dt - (time.perf_counter() - loop_start)))

@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop else None
    t_proc, r_act_proc, r_obs_proc = make_default_processors()

    robot.connect(connect_cameras=cfg.startup.connect_cameras)
    if teleop:
        teleop.connect()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            t_proc,
            create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            r_obs_proc,
            create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    # ==========================================
    # 데이터셋 생성 방식 수정 부분
    # - 참고 코드와 동일하게 image writer / encoding 옵션 반영
    # - 기존 record 흐름은 유지
    # ==========================================
    num_cameras = len(getattr(robot, "cameras", {}))

    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        vcodec=cfg.dataset.vcodec,
    )

    fast_sync_to_target(robot, teleop, cfg.startup)
    listener, events = init_keyboard_listener()

    try:
        with VideoEncodingManager(dataset):
            for i in range(cfg.dataset.num_episodes):
                if events["stop_recording"]:
                    break

                say_fast(f"Episode {i}", cfg.play_sounds, cfg.startup)

                record_loop(
                    robot,
                    events,
                    cfg.dataset.fps,
                    t_proc,
                    r_act_proc,
                    r_obs_proc,
                    cfg.startup,
                    dataset,
                    teleop,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=cfg.display_compressed_images,
                )

                # 현재 episode 저장
                try:
                    dataset.save_episode()
                    print(f"[Save] Episode {i} 저장 완료")
                except Exception as e:
                    print(f"[Save] Episode {i} 저장 실패: {e}")

    except KeyboardInterrupt:
        print("\n[Interrupt] Ctrl+C 감지됨. 현재 데이터를 저장하고 종료합니다.")
        try:
            dataset.save_episode()
            print("[Save] 현재 episode 저장 완료")
        except Exception as e:
            print(f"[Save] 현재 episode 저장 실패 또는 저장할 프레임 없음: {e}")

    finally:
        if dataset:
            try:
                print("[Finalize] dataset finalize 중...")
                dataset.finalize()
                print("[Finalize] 완료")
            except Exception as e:
                print(f"[Finalize] 오류 발생: {e}")

        if cfg.dataset.push_to_hub and dataset is not None:
            try:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            except Exception as e:
                print(f"[Hub] push_to_hub 실패: {e}")

        reset_follower_before_exit(robot, cfg.startup)

        if teleop:
            teleop.disconnect()

        if listener:
            listener.stop()

        if hasattr(robot, "disconnect"):
            robot.disconnect()
            
    return dataset

def main():
    if os.getenv("CALIBRATE_ORIGIN", "1") == "1":
        calibrate_origin()
    register_third_party_plugins()
    record()

if __name__ == "__main__":
    main()
