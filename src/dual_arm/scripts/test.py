#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XL430 (ID=1)에서 Extended Position 값을 읽어 출력하는 예제
- 통신속도: 1000000
- 제어테이블 기준: Present Position = 132 (4 bytes)
- 프로토콜: 2.0

동작 순서:
1. Torque OFF
2. Operating Mode를 Extended Position Control Mode(4)로 설정
3. Torque ON
4. Present Position 반복 출력

실행 예시:
python read_xl430_extended_position.py --port /dev/ttyUSB0
"""

import time
import argparse

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# -----------------------------
# XL430 / Protocol 2.0 설정
# -----------------------------
PROTOCOL_VERSION = 2.0
DXL_ID = 1
BAUDRATE = 1000000

# XL430 제어테이블 주소
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1

# XL430 Operating Mode
EXTENDED_POSITION_CONTROL_MODE = 4


def check_comm_result(packet_handler, dxl_comm_result, dxl_error, context=""):
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(f"{context} 통신 실패: {packet_handler.getTxRxResult(dxl_comm_result)}")
    if dxl_error != 0:
        raise RuntimeError(f"{context} 패킷 에러: {packet_handler.getRxPacketError(dxl_error)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True, help="예: /dev/ttyUSB0")
    parser.add_argument("--id", type=int, default=DXL_ID, help="Dynamixel ID (기본값: 1)")
    parser.add_argument("--baudrate", type=int, default=BAUDRATE, help="통신속도 (기본값: 1000000)")
    parser.add_argument("--interval", type=float, default=0.1, help="출력 주기 초 단위 (기본값: 0.1)")
    args = parser.parse_args()

    port_handler = PortHandler(args.port)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise RuntimeError(f"포트를 열 수 없습니다: {args.port}")
    print(f"[INFO] Port opened: {args.port}")

    if not port_handler.setBaudRate(args.baudrate):
        raise RuntimeError(f"보드레이트 설정 실패: {args.baudrate}")
    print(f"[INFO] Baudrate set: {args.baudrate}")

    try:
        # 1) Torque OFF
        dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
            port_handler, args.id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )
        check_comm_result(packet_handler, dxl_comm_result, dxl_error, "Torque Disable")
        print("[INFO] Torque OFF")

        # 2) Extended Position Mode 설정
        dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
            port_handler, args.id, ADDR_OPERATING_MODE, EXTENDED_POSITION_CONTROL_MODE
        )
        check_comm_result(packet_handler, dxl_comm_result, dxl_error, "Set Extended Position Mode")
        print("[INFO] Operating mode set to Extended Position Control Mode (4)")

        # 3) Torque ON
        dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
            port_handler, args.id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
        )
        check_comm_result(packet_handler, dxl_comm_result, dxl_error, "Torque Enable")
        print("[INFO] Torque ON")

        # 4) Present Position 읽기
        print("[INFO] Present Position 읽기 시작 (Ctrl+C 종료)")
        while True:
            present_position, dxl_comm_result, dxl_error = packet_handler.read4ByteTxRx(
                port_handler, args.id, ADDR_PRESENT_POSITION
            )
            check_comm_result(packet_handler, dxl_comm_result, dxl_error, "Read Present Position")

            # unsigned 32-bit -> signed 32-bit 변환
            if present_position >= 0x80000000:
                present_position -= 0x100000000

            print(f"ID {args.id} Present Position (Extended): {present_position}")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[INFO] 종료합니다.")

    finally:
        port_handler.closePort()
        print("[INFO] Port closed")


if __name__ == "__main__":
    main()