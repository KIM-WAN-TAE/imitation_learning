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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
# limitations under the License.

"""
Helper to recalibrate your device (robot or teleoperator).

Example:

dual_arm-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue
"""

import inspect
import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from dual_arm.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from dual_arm.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from dual_arm.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    so_follower,
)
from dual_arm.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    so_leader,
    unitree_g1,
)
from dual_arm.utils.import_utils import register_third_party_plugins
from dual_arm.utils.utils import init_logging


@dataclass
class CalibrateConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")
        self.device = self.robot if self.robot else self.teleop


def _call_connect_safely(device, *, calibrate: bool) -> None:
    """
    Call device.connect() with only the kwargs it supports.
    If possible, disable camera connection during calibration.
    """
    fn = getattr(device, "connect", None)
    if fn is None:
        raise AttributeError(f"{type(device).__name__} has no connect()")

    sig = inspect.signature(fn)
    supported = set(sig.parameters.keys())

    kwargs = {}
    if "calibrate" in supported:
        kwargs["calibrate"] = calibrate

    # If robot implementation supports disabling cameras, turn them off.
    for k in ("connect_cameras", "with_cameras", "enable_cameras", "use_cameras"):
        if k in supported:
            kwargs[k] = False
            break

    fn(**kwargs)


def _call_disconnect_safely(device) -> None:
    fn = getattr(device, "disconnect", None)
    if fn is None:
        return

    try:
        sig = inspect.signature(fn)
        supported = set(sig.parameters.keys())
        if "disable_torque" in supported:
            fn(disable_torque=True)
        else:
            fn()
    except Exception:
        logging.exception("disconnect() failed")


@draccus.wrap()
def calibrate(cfg: CalibrateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    elif isinstance(cfg.device, TeleoperatorConfig):
        device = make_teleoperator_from_config(cfg.device)
    else:
        raise ValueError("Invalid config: device must be a RobotConfig or TeleoperatorConfig")

    _call_connect_safely(device, calibrate=False)

    try:
        device.calibrate()
    finally:
        _call_disconnect_safely(device)


def main():
    register_third_party_plugins()
    calibrate()


if __name__ == "__main__":
    main()