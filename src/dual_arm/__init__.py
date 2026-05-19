#!/usr/bin/env python

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
This file contains lists of available environments, dataset and policies to reflect the current state of LeRobot library.
We do not want to import all the dependencies, but instead we keep it lightweight to ensure fast access to these variables.

Example:
    ```python
        import dual_arm
        print(dual_arm.available_envs)
        print(dual_arm.available_tasks_per_env)
        print(dual_arm.available_datasets)
        print(dual_arm.available_datasets_per_env)
        print(dual_arm.available_real_world_datasets)
        print(dual_arm.available_policies)
        print(dual_arm.available_policies_per_env)
        print(dual_arm.available_robots)
        print(dual_arm.available_cameras)
        print(dual_arm.available_motors)
    ```

When implementing a new dataset loadable with LeRobotDataset follow these steps:
- Update `available_datasets_per_env` in `dual_arm/__init__.py`

When implementing a new environment (e.g. `gym_aloha`), follow these steps:
- Update `available_tasks_per_env` and `available_datasets_per_env` in `dual_arm/__init__.py`

When implementing a new policy class (e.g. `DiffusionPolicy`) follow these steps:
- Update `available_policies` and `available_policies_per_env`, in `dual_arm/__init__.py`
- Set the required `name` class attribute.
- Update variables in `tests/test_available.py` by importing your new Policy class
"""

import itertools

from dual_arm.__version__ import __version__  # noqa: F401

# TODO(rcadene): Improve policies and envs. As of now, an item in `available_policies`
# refers to a yaml file AND a modeling name. Same for `available_envs` which refers to
# a yaml file AND a environment name. The difference should be more obvious.
available_tasks_per_env = {
    "aloha": [
        "AlohaInsertion-v0",
        "AlohaTransferCube-v0",
    ],
    "pusht": ["PushT-v0"],
}
available_envs = list(available_tasks_per_env.keys())

available_datasets_per_env = {
    "aloha": [
        "dual_arm/aloha_sim_insertion_human",
        "dual_arm/aloha_sim_insertion_scripted",
        "dual_arm/aloha_sim_transfer_cube_human",
        "dual_arm/aloha_sim_transfer_cube_scripted",
        "dual_arm/aloha_sim_insertion_human_image",
        "dual_arm/aloha_sim_insertion_scripted_image",
        "dual_arm/aloha_sim_transfer_cube_human_image",
        "dual_arm/aloha_sim_transfer_cube_scripted_image",
    ],
    # TODO(alexander-soare): Add "dual_arm/pusht_keypoints". Right now we can't because this is too tightly
    # coupled with tests.
    "pusht": ["dual_arm/pusht", "dual_arm/pusht_image"],
}

available_real_world_datasets = [
    "dual_arm/aloha_mobile_cabinet",
    "dual_arm/aloha_mobile_chair",
    "dual_arm/aloha_mobile_elevator",
    "dual_arm/aloha_mobile_shrimp",
    "dual_arm/aloha_mobile_wash_pan",
    "dual_arm/aloha_mobile_wipe_wine",
    "dual_arm/aloha_static_battery",
    "dual_arm/aloha_static_candy",
    "dual_arm/aloha_static_coffee",
    "dual_arm/aloha_static_coffee_new",
    "dual_arm/aloha_static_cups_open",
    "dual_arm/aloha_static_fork_pick_up",
    "dual_arm/aloha_static_pingpong_test",
    "dual_arm/aloha_static_pro_pencil",
    "dual_arm/aloha_static_screw_driver",
    "dual_arm/aloha_static_tape",
    "dual_arm/aloha_static_thread_velcro",
    "dual_arm/aloha_static_towel",
    "dual_arm/aloha_static_vinh_cup",
    "dual_arm/aloha_static_vinh_cup_left",
    "dual_arm/aloha_static_ziploc_slide",
    "dual_arm/umi_cup_in_the_wild",
    "dual_arm/unitreeh1_fold_clothes",
    "dual_arm/unitreeh1_rearrange_objects",
    "dual_arm/unitreeh1_two_robot_greeting",
    "dual_arm/unitreeh1_warehouse",
    "dual_arm/nyu_rot_dataset",
    "dual_arm/utokyo_saytap",
    "dual_arm/imperialcollege_sawyer_wrist_cam",
    "dual_arm/utokyo_xarm_bimanual",
    "dual_arm/tokyo_u_lsmo",
    "dual_arm/utokyo_pr2_opening_fridge",
    "dual_arm/cmu_franka_exploration_dataset",
    "dual_arm/cmu_stretch",
    "dual_arm/asu_table_top",
    "dual_arm/utokyo_pr2_tabletop_manipulation",
    "dual_arm/utokyo_xarm_pick_and_place",
    "dual_arm/ucsd_kitchen_dataset",
    "dual_arm/austin_buds_dataset",
    "dual_arm/dlr_sara_grid_clamp",
    "dual_arm/conq_hose_manipulation",
    "dual_arm/columbia_cairlab_pusht_real",
    "dual_arm/dlr_sara_pour",
    "dual_arm/dlr_edan_shared_control",
    "dual_arm/ucsd_pick_and_place_dataset",
    "dual_arm/berkeley_cable_routing",
    "dual_arm/nyu_franka_play_dataset",
    "dual_arm/austin_sirius_dataset",
    "dual_arm/cmu_play_fusion",
    "dual_arm/berkeley_gnm_sac_son",
    "dual_arm/nyu_door_opening_surprising_effectiveness",
    "dual_arm/berkeley_fanuc_manipulation",
    "dual_arm/jaco_play",
    "dual_arm/viola",
    "dual_arm/kaist_nonprehensile",
    "dual_arm/berkeley_mvp",
    "dual_arm/uiuc_d3field",
    "dual_arm/berkeley_gnm_recon",
    "dual_arm/austin_sailor_dataset",
    "dual_arm/utaustin_mutex",
    "dual_arm/roboturk",
    "dual_arm/stanford_hydra_dataset",
    "dual_arm/berkeley_autolab_ur5",
    "dual_arm/stanford_robocook",
    "dual_arm/toto",
    "dual_arm/fmb",
    "dual_arm/droid_100",
    "dual_arm/berkeley_rpt",
    "dual_arm/stanford_kuka_multimodal_dataset",
    "dual_arm/iamlab_cmu_pickup_insert",
    "dual_arm/taco_play",
    "dual_arm/berkeley_gnm_cory_hall",
    "dual_arm/usc_cloth_sim",
]

available_datasets = sorted(
    set(itertools.chain(*available_datasets_per_env.values(), available_real_world_datasets))
)

# lists all available policies from `dual_arm/policies`
available_policies = ["act", "diffusion", "tdmpc", "vqbet"]

# lists all available robots from `dual_arm/robots`
available_robots = [
    "koch",
    "koch_bimanual",
    "aloha",
    "so100",
    "so101",
]

# lists all available cameras from `dual_arm/cameras`
available_cameras = [
    "opencv",
    "intelrealsense",
]

# lists all available motors from `dual_arm/motors`
available_motors = [
    "dynamixel",
    "feetech",
]

# keys and values refer to yaml files
available_policies_per_env = {
    "aloha": ["act"],
    "pusht": ["diffusion", "vqbet"],
    "koch_real": ["act_koch_real"],
    "aloha_real": ["act_aloha_real"],
}

env_task_pairs = [(env, task) for env, tasks in available_tasks_per_env.items() for task in tasks]
env_dataset_pairs = [
    (env, dataset) for env, datasets in available_datasets_per_env.items() for dataset in datasets
]
env_dataset_policy_triplets = [
    (env, dataset, policy)
    for env, datasets in available_datasets_per_env.items()
    for dataset in datasets
    for policy in available_policies_per_env[env]
]
