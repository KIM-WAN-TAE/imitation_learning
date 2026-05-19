#!/usr/bin/env python3
import time
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.dynamixel import DynamixelMotorsBus
except ImportError:
    print("Error: LeRobot motor libraries not found. Ensure PYTHONPATH includes 'src'.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Manual Calibration Tool for Leader Arm")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Motor port")
    args = parser.parse_args()

    # SO-Leader 모터 정의 (ID 0-5)
    motors_def = {
        "shoulder_pan": Motor(0, "ax-12a", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(1, "xl430-w250", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(2, "ax-12a", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(3, "ax-12a", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(4, "ax-12a", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(5, "ax-12a", MotorNormMode.RANGE_0_100),
    }

    # 더미 캘리브레이션 (연결용)
    dummy_cal = {name: MotorCalibration(id=m.id, drive_mode=0, homing_offset=0, range_min=0, range_max=4095) for name, m in motors_def.items()}

    bus = DynamixelMotorsBus(port=args.port, motors=motors_def, calibration=dummy_cal)
    
    try:
        bus.connect(handshake=False)
        bus.disable_torque()
        print("[Connected] Torque is OFF. You can move the arm freely.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("" + "="*60)
    print(" 캘리브레이션 시작 (Leader Arm)")
    print(" 1. 각 관절을 가동 범위의 끝에서 끝까지 천천히 움직이세요.")
    print(" 2. 특히 그리퍼는 '완전 닫힘'과 '완전 열림' 상태를 확실히 거치세요.")
    print(" 3. 측정이 끝나면 Ctrl+C를 눌러 종료하세요.")
    print("="*60 + "")

    # 측정 데이터 저장용
    mins = {name: 9999 for name in motors_def}
    maxes = {name: -9999 for name in motors_def}

    try:
        while True:
            current_vals = {}
            # 노이즈 방지를 위해 개별적으로 읽음
            for name in motors_def:
                try:
                    val = bus.read("Present_Position", name, normalize=False, num_retry=2)
                    mins[name] = min(mins[name], val)
                    maxes[name] = max(maxes[name], val)
                    current_vals[name] = val
                except:
                    current_vals[name] = -1

            # 실시간 화면 출력
            print("" + " | ".join([f"{n[:4]}:{mins[n]:4}-{maxes[n]:4}" for n in motors_def]), end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("" + "="*60)
        print(" 캘리브레이션 결과 (이 값을 저에게 알려주세요!)")
        print("="*60)
        print(f"{'관절 이름':<15} | {'Min':>6} | {'Max':>6}")
        print("-" * 35)
        for name in motors_def:
            print(f"{name:<15} | {mins[name]:6d} | {maxes[name]:6d}")
        print("="*60)

    finally:
        bus.disconnect()

if __name__ == "__main__":
    main()
