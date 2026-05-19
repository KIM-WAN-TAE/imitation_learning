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
Replays the actions of an episode from a dataset on a robot.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from dual_arm.configs import parser
from dual_arm.datasets.lerobot_dataset import LeRobotDataset
from dual_arm.processor import make_default_robot_action_processor
from dual_arm.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1,
)
from dual_arm.utils.constants import ACTION
from dual_arm.utils.import_utils import register_third_party_plugins
from dual_arm.utils.robot_utils import precise_sleep
from dual_arm.utils.utils import init_logging, log_say


@dataclass
class DatasetReplayConfig:
    repo_id: str
    episode: int
    root: str | Path | None = None
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    play_sounds: bool = True


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)

    input_path = Path(cfg.dataset.repo_id).expanduser().resolve()

    if input_path.exists() and input_path.is_dir():
        # 사용자가 dataset.repo_id에 로컬 폴더 전체 경로를 넣은 경우
        repo_id = input_path.name
        root = input_path
        logging.info(f"Loading local dataset directory: {root}")
    else:
        # 일반적인 HF repo_id 또는 root + repo_id 조합
        repo_id = cfg.dataset.repo_id
        root = Path(cfg.dataset.root).expanduser().resolve() if cfg.dataset.root else None
        logging.info(f"Loading dataset with repo_id={repo_id}, root={root}")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[cfg.dataset.episode],
    )

    # episodes=[...] 로 이미 선택했으므로 그대로 사용
    episode_frames = dataset.hf_dataset
    actions = episode_frames.select_columns(ACTION)

    replay_fps = cfg.dataset.fps if cfg.dataset.fps is not None else dataset.fps

    robot.connect()

    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)

        for idx in range(len(episode_frames)):
            start_t = time.perf_counter()

            action_array = actions[idx][ACTION]
            action = {
                name: action_array[i]
                for i, name in enumerate(dataset.features[ACTION]["names"])
            }

            robot_obs = robot.get_observation()
            processed_action = robot_action_processor((action, robot_obs))
            robot.send_action(processed_action)

            dt_s = time.perf_counter() - start_t
            precise_sleep(max(1 / replay_fps - dt_s, 0.0))
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()