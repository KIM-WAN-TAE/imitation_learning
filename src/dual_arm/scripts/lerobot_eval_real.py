#!/usr/bin/env python3

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
Evaluates a policy on a real robot without saving a dataset.
Supports multiple cameras (e.g., OpenCV + RealSense) via robot configuration.
"""

import logging
import select
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from pynput import keyboard

from dual_arm.configs import parser
from dual_arm.configs.policies import PreTrainedConfig
from dual_arm.datasets.utils import build_dataset_frame
from dual_arm.policies.factory import make_policy, make_pre_post_processors
from dual_arm.policies.pretrained import PreTrainedPolicy
from dual_arm.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from dual_arm.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from dual_arm.utils.constants import OBS_STR
from dual_arm.utils.control_utils import (
    init_keyboard_listener,
    predict_action,
)
from dual_arm.utils.import_utils import register_third_party_plugins
from dual_arm.utils.robot_utils import precise_sleep
from dual_arm.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from dual_arm.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class EvalRealConfig:
    num_episodes: int = 10
    episode_time_s: int | float = 60
    reset_time_s: int | float = 5
    fps: int = 15


@dataclass
class EvalRealPipelineConfig:
    robot: RobotConfig
    policy: PreTrainedConfig | None = None
    eval: EvalRealConfig = field(default_factory=EvalRealConfig)
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    seed: int = 42

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        
        if self.policy is None:
            raise ValueError("Policy is not configured. Please specify a pretrained policy with `--policy.path`.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def eval_loop(
    robot: Robot,
    events: dict,
    fps: int,
    robot_action_processor: RobotProcessorPipeline,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset_features: dict[str, Any],
    control_time_s: int | float,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    
    inference_times = []
    action_magnitudes = []

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip().lower()
            if line == 'q':
                events["stop_recording"] = True

        if events["stop_recording"]:
            break

        # 1. Get robot observation
        obs = robot.get_observation()
        if obs is None:
            continue
            
        obs_dict = {}
        # 1a. Handle Images
        for cam_key in robot.cameras:
            if cam_key in obs:
                obs_dict[cam_key] = obs[cam_key]

        # 1b. Handle State (Proprioception)
        # Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for i, joint in enumerate(joint_names):
            key = f"{joint}.pos"
            if key in obs:
                obs_dict[f"observation.state/{i}"] = obs[key]

        # 2. Build observation frame
        observation_frame = build_dataset_frame(dataset_features, obs_dict, prefix=OBS_STR)

        # 3. Predict action
        t_start_inf = time.perf_counter()
        with torch.inference_mode():
            # 3a. Prepare observation (convert to tensor, add batch dim)
            from dual_arm.policies.utils import prepare_observation_for_inference
            observation_tensor_dict = prepare_observation_for_inference(
                observation_frame, 
                get_safe_torch_device(policy.config.device), 
                robot_type=robot.robot_type
            )
            
            # 3b. Preprocess
            processed_obs = preprocessor(observation_tensor_dict)

            # 3c. Policy inference (ACT outputs a chunk of actions)
            action_chunk = policy.select_action(processed_obs)

            # 3d. Postprocess
            # PolicyAction is a TypeAlias for torch.Tensor.
            # Some pipelines expect a dict, some expect a Tensor.
            # We use the most direct way to get the unnormalized action.
            try:
                # Try passing as a direct tensor (PolicyAction)
                processed_action = postprocessor(action_chunk)
                # If it returned a dict (EnvTransition), extract the action
                if isinstance(processed_action, dict) and "action" in processed_action:
                    action_tensor = processed_action["action"]
                elif isinstance(processed_action, dict) and TransitionKey.ACTION.value in processed_action:
                    action_tensor = processed_action[TransitionKey.ACTION.value]
                else:
                    action_tensor = processed_action
            except Exception:
                # Fallback: manually find the normalizer step and unnormalize
                logging.warning("Standard postprocessing failed. Attempting manual unnormalization.")
                action_tensor = action_chunk
                for step in postprocessor.steps:
                    if hasattr(step, "unnormalize"):
                        action_tensor = step.unnormalize(action_tensor)
            
            # Take the first action from the chunk
            action_tensor = action_tensor[0] if action_tensor.ndim > 1 else action_tensor

        t_end_inf = time.perf_counter()
        inference_times.append(t_end_inf - t_start_inf)

        # 4. Convert to RobotAction
        # Map 'action/i' back to robot-specific joint names
        action_dict = {}
        # For ACT, we typically take the first action from the predicted chunk
        current_action = action_tensor[0] if action_tensor.ndim > 1 else action_tensor
        for i, joint in enumerate(joint_names):
            action_dict[f"{joint}.pos"] = current_action[i].item()
        
        move_mag = np.mean([abs(v) for v in action_dict.values()])
        action_magnitudes.append(move_mag)

        # 5. Send action to robot
        if len(inference_times) % 15 == 0:
            formatted_actions = {k: f"{v:.2f}" for k, v in action_dict.items()}
            logging.info(f"Target Angles: {formatted_actions}")

        robot.send_action(action_dict)

        if display_data:
            log_rerun_data(
                observation=observation_frame, action=action_dict, compress_images=display_compressed_images
            )

        if len(inference_times) % 30 == 0:
            avg_inf = np.mean(inference_times[-30:]) * 1000
            avg_mag = np.mean(action_magnitudes[-30:])
            logging.info(f"Diag: Inference={avg_inf:.1f}ms | Avg Raw Mag={avg_mag:.4f}")

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def main(cfg: EvalRealPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        init_rerun(session_name="evaluation", ip=cfg.display_ip, port=cfg.display_port)
    
    display_compressed_images = (
        True if (cfg.display_data and cfg.display_ip is not None) else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    _, robot_action_processor, _ = make_default_processors()

    # Load policy and pre/post-processors using local config to avoid Hub lookup
    from types import SimpleNamespace
    from dual_arm.configs.types import FeatureType
    
    all_features = {**cfg.policy.input_features, **cfg.policy.output_features}
    dummy_env_cfg = SimpleNamespace(
        features=all_features,
        features_map={key: key for key in all_features},
    )
    policy = make_policy(cfg.policy, env_cfg=dummy_env_cfg)
    
    # Construct dataset_features for the eval_loop
    dataset_features = {}
    for key, ft in all_features.items():
        if ft.type == FeatureType.VISUAL:
            dataset_features[key] = {"dtype": "image", "shape": ft.shape}
        else:
            dataset_features[key] = {
                "dtype": "float32", 
                "shape": ft.shape,
                "names": [f"{key}/{i}" for i in range(ft.shape[0])]
            }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
    )

    try:
        robot.connect()
        listener, events = init_keyboard_listener()

        print("\n" + "="*60)
        print(" [Evaluation] RUNNING IN RAW MODE (No Smoothing/No Clamping)")
        print(" Press 'q' + Enter or 'Esc' to stop.")

        recorded_episodes = 0
        while recorded_episodes < cfg.eval.num_episodes and not events["stop_recording"]:
            log_say(f"Evaluating episode {recorded_episodes + 1}", cfg.play_sounds)
            
            eval_loop(
                robot=robot,
                events=events,
                fps=cfg.eval.fps,
                robot_action_processor=robot_action_processor,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset_features=dataset_features,
                control_time_s=cfg.eval.episode_time_s,
                display_data=cfg.display_data,
                display_compressed_images=display_compressed_images,
            )

            if not events["stop_recording"] and (recorded_episodes < cfg.eval.num_episodes - 1):
                log_say("Reset", cfg.play_sounds)
                time.sleep(cfg.eval.reset_time_s)
            recorded_episodes += 1

    finally:
        if robot.is_connected:
            robot.disconnect()
        if listener:
            listener.stop()
        log_say("Evaluation finished", cfg.play_sounds)


if __name__ == "__main__":
    register_third_party_plugins()
    main()