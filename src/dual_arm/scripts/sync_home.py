#!/usr/bin/env python
import logging
import time
from dataclasses import dataclass

from dual_arm.configs import parser
from dual_arm.robots import make_robot_from_config, RobotConfig
from dual_arm.teleoperators import make_teleoperator_from_config, TeleoperatorConfig
from dual_arm.utils.utils import init_logging

# ==========================================
# [사용자 설정] 각 관절별 절대 원점 위치 (단위: Degree 또는 Norm)
# 기본적으로 0.0으로 설정되어 있으며, 필요 시 여기서 값을 수정하세요.
# ==========================================
TARGET_POSITIONS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
# ==========================================

@dataclass
class SyncHomeConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    fps: int = 50  # 동기화 이동 속도 (Hz)
    duration: float = 3.0  # 이동에 걸리는 시간 (초)

@parser.wrap()
def main(cfg: SyncHomeConfig):
    init_logging()
    logger = logging.getLogger(__name__)

    # 로봇 및 리더 인스턴스 생성
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    # 연결
    robot.connect()
    teleop.connect()

    print(f"\n[Sync] 시작 위치 동기화 프로세스를 시작합니다. (목표: {TARGET_POSITIONS})")
    print("[Sync] 주의: 리더와 팔로워 팔이 모두 움직입니다. 주변을 확인하세요.")

    try:
        # 1. 두 로봇 모두 토크 활성화
        robot.bus.enable_torque()
        teleop.bus.enable_torque()

        # 2. 현재 위치 읽기 (부드러운 이동을 위한 시작점)
        robot_obs = robot.get_observation()
        teleop_raw = teleop.get_action()

        # 3. 부드러운 이동 (Interpolation)
        steps = int(cfg.fps * cfg.duration)
        for i in range(steps):
            alpha = (i + 1) / steps
            
            # 팔로워 목표 계산 (Joint Space)
            f_goal = {}
            for motor in robot.bus.motors:
                start_val = robot_obs.get(f"{motor}.pos", 0.0)
                target_val = TARGET_POSITIONS.get(motor, 0.0)
                f_goal[f"{motor}.pos"] = start_val + (target_val - start_val) * alpha
            
            # 리더 목표 계산 (1:1이므로 Joint Space 직접 사용)
            l_goal = {}
            for motor in teleop.bus.motors:
                start_val = teleop_raw.get(f"{motor}.pos", 0.0)
                target_val = TARGET_POSITIONS.get(motor, 0.0)
                l_goal[motor] = start_val + (target_val - start_val) * alpha

            # 실제 전송
            robot.send_action(f_goal)
            teleop.bus.sync_write("Goal_Position", l_goal)
            
            time.sleep(1.0 / cfg.fps)

        print("[Sync] 목표 위치에 도달했습니다. 1초간 유지합니다.")
        time.sleep(1.0)

        # 4. 리더 토크 해제 (팔로워는 유지)
        teleop.bus.disable_torque()
        print("[Sync] 리더 토크 해제 완료. 이제 리더 팔을 자유롭게 움직여 조종하세요!")

    except KeyboardInterrupt:
        print("\n[Sync] 사용자에 의해 중단되었습니다.")
    finally:
        # 안전을 위해 연결 해제는 하되, 팔로워 토크는 유지하고 싶다면 disconnect 인자 조정 필요
        # 여기서는 스크립트 종료 후에도 팔로워가 멈춰있게 하기 위해 일반적인 disconnect 수행
        teleop.disconnect()
        # robot.disconnect() # 팔로워 토크를 계속 유지하려면 이 줄을 주석 처리하거나 토크 유지 옵션 사용

if __name__ == "__main__":
    main()
