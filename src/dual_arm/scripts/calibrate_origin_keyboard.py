import sys
import tty
import termios
import time
from dynamixel_sdk import *

# --- 통신 설정 ---
DXL_PORT     = '/dev/ttyUSB0'
DXL_BAUDRATE = 1000000

# --- 컨트롤 테이블 주소 ---
ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112

# --- Position PID Gain 주소: X-series Protocol 2.0 기준 ---
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84

PROTOCOL_VERSION = 2.0
TORQUE_ENABLE    = 1
TORQUE_DISABLE   = 0
OP_MODE_POSITION     = 3
OP_MODE_EXT_POSITION = 4

TICKS_PER_REV = 4096

# --- 1번 모터 PID 세팅값 ---
# 부하 때문에 목표 위치에 도달하지 못하는 상황 기준 예시값
# 너무 크면 진동/과부하가 생길 수 있으므로 조금씩 조정하세요.
MOTOR_1_P_GAIN = 2000
MOTOR_1_I_GAIN = 0
MOTOR_1_D_GAIN = 3600

# ─────────────────────────────────────────────────────────────
# 모터별 원점 위치 (엔코더 절대값)
# ─────────────────────────────────────────────────────────────
MOTOR_HOME = {
    # 오른팔 기어 모터
    1:  (-9900) % TICKS_PER_REV,
    2:  2048,
    3:  2048,
    4:  (-2048) % TICKS_PER_REV,

    # 오른팔 일반 모터
    5:  2048,
    6:  2048,
    7:  2048 + round(80 * (2048 * 2 / 360)),
    8:  2048,

    # # 왼팔 기어 모터
    # 11: (9900) % TICKS_PER_REV,
    # 12: 2048,
    # 13: 2048,
    # 14: (2048) % TICKS_PER_REV,

    # # 왼팔 일반 모터
    # 15: 2048,
    # 16: 2048,
    # 17: 2048 - round(80 * (2048 * 2 / 360)),
    # 18: 2048,
}

# 오른팔
RIGHT_GEARED = [1, 2, 3, 4]
RIGHT_NORMAL = [5, 6, 7, 8]

# 왼팔
LEFT_GEARED  = [11, 12, 13, 14]
LEFT_NORMAL  = [15, 16, 17, 18]

ALL_GEARED = RIGHT_GEARED + LEFT_GEARED
ALL_NORMAL = RIGHT_NORMAL + LEFT_NORMAL

JOG_STEP = 300

portHandler   = PortHandler(DXL_PORT)
packetHandler = PacketHandler(PROTOCOL_VERSION)


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def check_comm_result(comm_result, dxl_error, dxl_id, action_name):
    """
    통신 결과 확인용 함수
    """
    if comm_result != COMM_SUCCESS:
        print(f"[오류] {dxl_id}번 모터 {action_name} 실패: {packetHandler.getTxRxResult(comm_result)}")
        return False

    if dxl_error != 0:
        print(f"[오류] {dxl_id}번 모터 {action_name} 에러: {packetHandler.getRxPacketError(dxl_error)}")
        return False

    return True


def set_position_pid(dxl_id, p_gain, i_gain, d_gain):
    """
    Position PID Gain 설정 함수

    X-series Protocol 2.0 기준:
      D Gain: Address 80, 2 byte
      I Gain: Address 82, 2 byte
      P Gain: Address 84, 2 byte
    """

    comm_result, dxl_error = packetHandler.write2ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_POSITION_D_GAIN,
        d_gain
    )
    check_comm_result(comm_result, dxl_error, dxl_id, "D Gain 설정")

    comm_result, dxl_error = packetHandler.write2ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_POSITION_I_GAIN,
        i_gain
    )
    check_comm_result(comm_result, dxl_error, dxl_id, "I Gain 설정")

    comm_result, dxl_error = packetHandler.write2ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_POSITION_P_GAIN,
        p_gain
    )
    check_comm_result(comm_result, dxl_error, dxl_id, "P Gain 설정")

    print(f"[{dxl_id}번 모터] PID 설정 완료: P={p_gain}, I={i_gain}, D={d_gain}")


def set_motor_1_pid_if_needed(dxl_id):
    """
    1번 모터일 때만 PID 적용
    """
    if dxl_id == 1:
        set_position_pid(
            dxl_id,
            MOTOR_1_P_GAIN,
            MOTOR_1_I_GAIN,
            MOTOR_1_D_GAIN
        )


def setup():
    if not portHandler.openPort():
        print("다이나믹셀 포트를 열 수 없습니다.")
        quit()

    if not portHandler.setBaudRate(DXL_BAUDRATE):
        print("보드레이트 변경 실패.")
        quit()

    for dxl_id in ALL_GEARED:
        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_DISABLE
        )

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_OPERATING_MODE,
            OP_MODE_EXT_POSITION
        )

        # 1번 모터만 PID 세팅
        # Operating Mode 설정 후, Torque ON 전에 적용
        set_motor_1_pid_if_needed(dxl_id)

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_ENABLE
        )


def read_present_position(dxl_id):
    pos, _, _ = packetHandler.read4ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_PRESENT_POSITION
    )

    if pos > 2147483647:
        pos -= 4294967296

    return pos


def jog_motor(dxl_id, direction):
    current_pos = read_present_position(dxl_id)
    target_pos  = current_pos + (JOG_STEP * direction)
    write_pos   = target_pos & 0xFFFFFFFF

    packetHandler.write4ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_GOAL_POSITION,
        write_pos
    )

    print(
        f"\r[{dxl_id}번 모터] 현재위치: {current_pos} -> 목표위치: {target_pos}        ",
        end=""
    )


def reboot_and_home_geared(dxl_id):
    """
    기어 모터 reboot → EXT_POSITION 재설정 → 모터별 원점으로 이동

    reboot 시 멀티턴 카운터가 리셋되므로 MOTOR_HOME은 0~4095 범위 값이어야 함
    PID도 RAM 값이므로 reboot 후 다시 설정해야 함
    """
    home = MOTOR_HOME.get(dxl_id, 2048)

    print(f"\n\n[{dxl_id}번 모터] 재부팅 및 멀티턴 초기화를 진행합니다...")

    packetHandler.reboot(portHandler, dxl_id)
    time.sleep(1.0)

    packetHandler.write1ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_TORQUE_ENABLE,
        TORQUE_DISABLE
    )

    packetHandler.write1ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_OPERATING_MODE,
        OP_MODE_EXT_POSITION
    )

    # 1번 모터는 reboot 후 PID 재설정
    set_motor_1_pid_if_needed(dxl_id)

    packetHandler.write1ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_TORQUE_ENABLE,
        TORQUE_ENABLE
    )

    print(f"[{dxl_id}번 모터] 원점 위치 {home} ticks 으로 이동합니다.")

    packetHandler.write4ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_GOAL_POSITION,
        home
    )

    time.sleep(0.5)


def home_normal_motors(arm):
    motors = RIGHT_NORMAL if arm == 'right' else LEFT_NORMAL

    print(f"\n\n--- {'오른팔' if arm == 'right' else '왼팔'} 일반 모터({motors}) 재부팅 및 원점 복귀 ---")

    for dxl_id in motors:
        home = MOTOR_HOME.get(dxl_id, 2048)

        print(f"[{dxl_id}번 모터] 재부팅 중...")

        packetHandler.reboot(portHandler, dxl_id)
        time.sleep(1.0)

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_DISABLE
        )

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_OPERATING_MODE,
            OP_MODE_POSITION
        )

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_ENABLE
        )

        packetHandler.write4ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_PROFILE_VELOCITY,
            50
        )

        packetHandler.write4ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_GOAL_POSITION,
            home
        )

        packetHandler.write4ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_PROFILE_VELOCITY,
            0
        )

        print(f"[{dxl_id}번 모터] 원점 {home} ticks 으로 이동 명령 전송 완료.")


def go_all_home():
    """
    전체 모터 reboot → 모드 재설정 → GroupSyncWrite로 동시 원점 이동

    1번 모터는 reboot 후 PID가 초기화될 수 있으므로 다시 설정함
    """
    print("\n\n--- 전체 모터 리부트 및 원점 복귀 시작 ---")

    # 1. 전체 리부트
    for dxl_id in MOTOR_HOME:
        print(f"  [{dxl_id}번] 리부트 중...", end=" ", flush=True)
        packetHandler.reboot(portHandler, dxl_id)
        print("완료")

    print("  전체 리부트 완료. 0.5초 대기...")
    time.sleep(0.5)

    # 2. 모드 재설정
    for dxl_id in MOTOR_HOME:
        mode = OP_MODE_EXT_POSITION if dxl_id in ALL_GEARED else OP_MODE_POSITION

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_DISABLE
        )

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_OPERATING_MODE,
            mode
        )

        # 1번 모터만 PID 재설정
        set_motor_1_pid_if_needed(dxl_id)

        if dxl_id not in ALL_GEARED:
            packetHandler.write4ByteTxRx(
                portHandler,
                dxl_id,
                ADDR_PROFILE_VELOCITY,
                50
            )

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_ENABLE
        )

        print(
            f"  [{dxl_id}번] 모드={'EXT_POS' if mode == OP_MODE_EXT_POSITION else 'POSITION'} 설정 완료"
        )

    # 3. GroupSyncWrite로 동시 원점 이동
    groupSyncWrite = GroupSyncWrite(
        portHandler,
        packetHandler,
        ADDR_GOAL_POSITION,
        4
    )

    for dxl_id, home in MOTOR_HOME.items():
        home_val = home & 0xFFFFFFFF

        param = [
            DXL_LOBYTE(DXL_LOWORD(home_val)),
            DXL_HIBYTE(DXL_LOWORD(home_val)),
            DXL_LOBYTE(DXL_HIWORD(home_val)),
            DXL_HIBYTE(DXL_HIWORD(home_val)),
        ]

        if not groupSyncWrite.addParam(dxl_id, param):
            print(f"  [경고] 모터 {dxl_id} 파라미터 추가 실패")

    comm_result = groupSyncWrite.txPacket()

    if comm_result != COMM_SUCCESS:
        print(f"  [오류] SyncWrite 실패: {packetHandler.getTxRxResult(comm_result)}")
    else:
        print("\n  전체 모터 원점 이동 명령 전송 완료.")
        for dxl_id, home in sorted(MOTOR_HOME.items()):
            print(f"    모터 {dxl_id:>3}: {home} ticks")

    groupSyncWrite.clearParam()


def calibrate_origin():
    setup()

    # 시작 시 MOTOR_HOME 테이블 출력
    print("\n=== 모터별 원점 위치 테이블 ===")
    print(f"  {'ID':>4} | {'원점(ticks)':>11} | {'원점(°, 2048기준)':>18}")
    print(f"  {'─' * 4}-+-{'─' * 11}-+-{'─' * 18}")

    for mid, home in sorted(MOTOR_HOME.items()):
        deg = (home - 2048) * 360.0 / TICKS_PER_REV
        print(f"  {mid:>4} | {home:>11} | {deg:>+17.2f}°")

    print()

    print("\n=======================================================")
    print("      양팔 키보드 수동 원점 정렬 (Dual-Arm Homing)      ")
    print("=======================================================")
    print(" [ R ] : 오른팔 모드 (모터 1-8)   기본값")
    print(" [ L ] : 왼팔 모드  (모터 11-18)")
    print(" [ 1, 2, 3, 4 ] : 기어 모터 선택")
    print(" [ a ] / [ d ] : 선택한 모터 시계 반대 / 시계 방향으로 회전")
    print(" [ r ] : 선택한 모터 멀티턴 초기화(Reboot) → 모터별 원점으로 이동")
    print(" [ h ] : 현재 팔의 일반 모터 원점 복귀")
    print("    오른팔: 모터 5, 6, 7, 8  /  왼팔: 모터 15, 16, 17, 18")
    print(" [ q ] : 프로그램 종료")
    print(" [ z ] : 전체 모터 동시 원점 복귀 (MOTOR_HOME 기준)")
    print("=======================================================\n")

    arm_mode       = 'right'
    selected_motor = 1

    print(f"-> 현재 선택: 오른팔, 모터 {selected_motor}번")

    try:
        while True:
            key = getch()

            if key == '\x03':
                print("\n[Ctrl+C] 강제 종료됨")
                break

            if key == 'q':
                print("\n프로그램을 종료합니다.")
                break

            elif key == 'R':
                arm_mode       = 'right'
                selected_motor = 1
                print("\n-> 오른팔 모드 선택됨 (모터 1-8). 선택된 모터: 1번")

            elif key in ['L', 'l']:
                arm_mode       = 'left'
                selected_motor = 11
                print("\n-> 왼팔 모드 선택됨 (모터 11-18). 선택된 모터: 11번")

            elif key in ['1', '2', '3', '4']:
                idx = int(key)
                selected_motor = idx if arm_mode == 'right' else (idx + 10)
                print(f"\n-> {selected_motor}번 모터 선택됨")

            elif key == 'a':
                jog_motor(selected_motor, -1)

            elif key == 'd':
                jog_motor(selected_motor, 1)

            elif key == 'r':
                reboot_and_home_geared(selected_motor)

            elif key == 'h':
                home_normal_motors(arm_mode)

            elif key == 'z':
                go_all_home()

    except KeyboardInterrupt:
        print("\n강제 종료됨")
        portHandler.closePort()
        quit()

    return portHandler, packetHandler


if __name__ == '__main__':
    port, packet = calibrate_origin()
    port.closePort()
