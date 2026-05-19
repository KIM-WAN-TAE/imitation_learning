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
    parser = argparse.ArgumentParser(description="Manual Calibration Tool for Follower (Puppet) Arm")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Follower motor port")
    args = parser.parse_args()

    # ✅ 팔로워 전용 ID 설정 (10번~15번)
    motors_def = {
        "shoulder_pan": Motor(10, "ax-12a", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(11, "xl430-w250", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(12, "ax-12a", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(13, "ax-12a", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(14, "ax-12a", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(15, "ax-12a", MotorNormMode.RANGE_0_100),
    }

    dummy_cal = {name: MotorCalibration(id=m.id, drive_mode=0, homing_offset=0, range_min=0, range_max=4095) for name, m in motors_def.items()}

    print(f"Connecting to Follower on port {args.port} (IDs: 10-15)...")
    bus = DynamixelMotorsBus(port=args.port, motors=motors_def, calibration=dummy_cal)
    
    try:
        # handshake=False로 연결하여 누락된 모터가 있어도 실행 가능하게 함
        bus.connect(handshake=False)
        bus.disable_torque()
        print("\n[Connected] Follower Arm torque is OFF.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("\n" + "="*60)
    print(" 팔로워(Follower) 캘리브레이션 시작")
    print(" 1. 팔로워 암을 리더암의 가동 범위와 똑같이 손으로 움직여보세요.")
    print(" 2. 측정이 끝나면 Ctrl+C를 눌러 종료하세요.")
    print("="*60 + "\n")

    mins = {name: 9999 for name in motors_def}
    maxes = {name: -9999 for name in motors_def}

    try:
        while True:
            for name in motors_def:
                try:
                    val = bus.read("Present_Position", name, normalize=False, num_retry=2)
                    mins[name] = min(mins[name], val)
                    maxes[name] = max(maxes[name], val)
                except: pass

            # 화면에 실시간 범위 출력
            print("\r" + " | ".join([f"{n[:4]}:{mins[n]:4}-{maxes[n]:4}" for n in motors_def]), end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print(" 팔로워 캘리브레이션 결과 (ID 10-15)")
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
