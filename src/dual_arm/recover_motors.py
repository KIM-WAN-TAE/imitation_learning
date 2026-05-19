import dynamixel_sdk as dxl
import sys
import time


def recover(port, baudrate=1000000):
    # ID 리스트 (0~5, 10~15 모든 가능성 포함)
    target_ids = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]

    ph = dxl.PortHandler(port)
    # Protocol 1.0 (AX-12A)용 핸들러
    pkt = dxl.PacketHandler(1.0)

    if not ph.openPort():
        print(f"❌ 포트 {port}를 열 수 없습니다.")
        return

    if not ph.setBaudRate(baudrate):
        print(f"❌ 보드레이트 {baudrate} 설정 실패.")
        ph.closePort()
        return

    ADDR_STATUS_RETURN_LEVEL = 16
    print(f"\n🚀 {port} 포트 복구 시작 (Baudrate: {baudrate})")

    for m_id in target_ids:
        # 응답을 기다리지 않고 명령만 전송
        # Status Return Level을 2(모든 명령에 응답)로 강제 설정
        pkt.write1ByteTxOnly(ph, m_id, ADDR_STATUS_RETURN_LEVEL, 2)
        print(f"✅ ID {m_id} 설정 변경 명령 전송 완료")
        time.sleep(0.05)

    ph.closePort()
    print(f"✨ {port} 복구 프로세스 종료\n")


if __name__ == "__main__":
    # 포트 인자가 없으면 기본값으로 ttyUSB0 사용
    target_port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB1"
    recover(target_port)