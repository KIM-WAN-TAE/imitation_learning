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

########################################################################################
# Utilities
########################################################################################


import logging
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff

from dual_arm.datasets.lerobot_dataset import LeRobotDataset
from dual_arm.datasets.utils import DEFAULT_FEATURES
from dual_arm.policies.pretrained import PreTrainedPolicy
from dual_arm.policies.utils import prepare_observation_for_inference
from dual_arm.processor import PolicyAction, PolicyProcessorPipeline
from dual_arm.robots import Robot


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).
    """
    try:
        import pynput  # noqa
        return False
    except Exception:
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)
        action = policy.select_action(observation)
        action = postprocessor(action)
    return action


def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.
    """
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning("Headless environment detected. Keyboard inputs will not be available.")
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc or (hasattr(key, 'char') and key.char == 'q'):
                print(f"\nStop recording requested. (Key: {key})")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided.")
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided.")


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]
    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")
    if mismatches:
        raise ValueError("Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches))
