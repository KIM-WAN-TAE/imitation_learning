import dynamixel_sdk as dxl
import sys


def set_fast_response(port):
    ph = dxl.PortHandler(port)
    pkt1 = dxl.PacketHandler(1.0)  # AX-12A
    pkt2 = dxl.PacketHandler(2.0)  # XH430 / XL430

    if not ph.openPort():
        print(f"❌ {port} 포트 열기 실패")
        return

    if not ph.setBaudRate(1000000):
        print(f"❌ {port} 보드레이트 설정 실패")
        ph.closePort()
        return

    # # 네 모터 리더 구성 기준
    # protocol_1_ids = [0, 5]      # AX-12A
    # protocol_2_ids = [1, 2, 3, 4]  # XH430-V350, XL430-W250

        # 네 모터 팔로워 구성 기준
    protocol_1_ids = [10, 15]      # AX-12A
    protocol_2_ids = [11, 12, 13, 14]  # XH430-V350, XL430-W250


    ADDR_RETURN_DELAY_TIME_P1 = 5
    ADDR_RETURN_DELAY_TIME_P2 = 9

    print(f"🚀 {port} 모터 응답 속도 최적화 시작...")

    for m_id in protocol_1_ids:
        comm_result, dxl_error = pkt1.write1ByteTxRx(
            ph, m_id, ADDR_RETURN_DELAY_TIME_P1, 0
        )
        if comm_result != dxl.COMM_SUCCESS:
            print(f"❌ ID {m_id} (P1): {pkt1.getTxRxResult(comm_result)}")
        elif dxl_error != 0:
            print(f"⚠️ ID {m_id} (P1): {pkt1.getRxPacketError(dxl_error)}")
        else:
            print(f"✅ ID {m_id} (AX-12A): Return Delay Time = 0 설정 완료")

    for m_id in protocol_2_ids:
        comm_result, dxl_error = pkt2.write1ByteTxRx(
            ph, m_id, ADDR_RETURN_DELAY_TIME_P2, 0
        )
        if comm_result != dxl.COMM_SUCCESS:
            print(f"❌ ID {m_id} (P2): {pkt2.getTxRxResult(comm_result)}")
        elif dxl_error != 0:
            print(f"⚠️ ID {m_id} (P2): {pkt2.getRxPacketError(dxl_error)}")
        else:
            print(f"✅ ID {m_id} (X-series): Return Delay Time = 0 설정 완료")

    ph.closePort()
    print("✨ 설정 종료")


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    set_fast_response(port)