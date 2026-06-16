import json
import logging
import os
import socket
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from threading import Thread
from typing import Optional


class RobotMode(str, Enum):
    IDLE = "idle"
    POLICY = "policy"
    RESET_HOME = "reset_home"
    EXIT = "exit"
    PACKAGE_START = "package_start"
    PACKAGE_COMPLETE = "package_complete"


class ObjectColor(str, Enum):
    NONE = "none"
    GREEN = "green"
    RED = "red"
    BLUE = "blue"


@dataclass
class CommandState:
    mode: RobotMode = RobotMode.IDLE
    color: ObjectColor = ObjectColor.NONE
    raw_text: str = ""
    version: int = 0

GREEN_POLICY_PATH = ("/home/roma/pt/outputs/triple_camera/act/green_0529_chunk100/checkpoints/010000/pretrained_model")
RED_POLICY_PATH = ("/home/roma/pt/outputs/triple_camera/act/red_0529_chunk100/checkpoints/010000/pretrained_model")
BLUE_POLICY_PATH = ("/home/roma/pt/outputs/triple_camera/act/blue_0529_chunk80/checkpoints/010000/pretrained_model")

POLICY_REGISTRY = {
    ObjectColor.GREEN: os.getenv("GREEN_POLICY_PATH", GREEN_POLICY_PATH),
    ObjectColor.RED: os.getenv("RED_POLICY_PATH", RED_POLICY_PATH),
    ObjectColor.BLUE: os.getenv("BLUE_POLICY_PATH", BLUE_POLICY_PATH),
}

CHUNK_REGISTRY = {
    ObjectColor.GREEN: int(os.getenv("GREEN_CHUNK_SIZE", "100")),
    ObjectColor.RED: int(os.getenv("RED_CHUNK_SIZE", "100")),
    ObjectColor.BLUE: int(os.getenv("BLUE_CHUNK_SIZE", "80")),
}

# COMMAND_HOST = os.getenv("COMMAND_HOST", "127.0.0.1")
# COMMAND_PORT = int(os.getenv("COMMAND_PORT", "8765"))
COMMAND_HOST = os.getenv("COMMAND_HOST", "192.168.0.3")
COMMAND_PORT = int(os.getenv("COMMAND_PORT", "8765"))

def parse_command(text: str) -> tuple[RobotMode, ObjectColor]:
    """
    Rule-based parser. Replace this function body with an sLLM JSON parser later.
    """
    normalized = text.strip().lower()

    if any(keyword in normalized for keyword in ("종료", "끝내", "끝", "exit", "quit", "q")):
        return RobotMode.EXIT, ObjectColor.NONE

    if any(keyword in normalized for keyword in ("포장시작", "포장 시작", "package start", "package_start")):
        return RobotMode.PACKAGE_START, ObjectColor.NONE

    if any(keyword in normalized for keyword in ("포장종료", "포장 종료", "포장완료", "package end", "package_end", "package complete", "package_complete")):
        return RobotMode.PACKAGE_COMPLETE, ObjectColor.NONE

    if any(keyword in normalized for keyword in ("초기화", "원위치", "홈", "home", "reset")):
        return RobotMode.RESET_HOME, ObjectColor.NONE

    if any(keyword in normalized for keyword in ("멈춰", "정지", "대기", "stop", "idle", "wait")):
        return RobotMode.IDLE, ObjectColor.NONE

    if any(keyword in normalized for keyword in ("초록", "녹색", "green")):
        return RobotMode.POLICY, ObjectColor.GREEN

    if any(keyword in normalized for keyword in ("빨강", "빨간", "적색", "red")):
        return RobotMode.POLICY, ObjectColor.RED

    if any(keyword in normalized for keyword in ("파랑", "파란", "청색", "blue")):
        return RobotMode.POLICY, ObjectColor.BLUE

    logging.warning("[Command] Unknown command: %s", text)
    return RobotMode.IDLE, ObjectColor.NONE


def command_input_loop(command_state: CommandState, state_lock: Lock, events: dict):
    print("[Command] 자연어 명령 대기: 초록/빨강/파랑 물체 잡아줘, 멈춰, 초기화, 종료")

    while not events.get("stop_recording", False):
        try:
            text = input("[명령 입력] ").strip()
        except EOFError:
            break

        if not text:
            continue

        mode, color = parse_command(text)

        with state_lock:
            command_state.mode = mode
            command_state.color = color
            command_state.raw_text = text
            command_state.version += 1

        logging.info("[Command] text=%r mode=%s color=%s", text, mode.value, color.value)

        if mode == RobotMode.EXIT:
            events["stop_recording"] = True
            break


def command_server_loop(command_state: CommandState, state_lock: Lock, events: dict, host: str, port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        server.settimeout(0.5)
        logging.info("[Command] Listening on %s:%s", host, port)
        print(f"[Command] 명령 수신 서버 대기 중: {host}:{port}")

        while not events.get("stop_recording", False):
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue

            with conn:
                logging.info("[Command] Client connected: %s", addr)
                for raw_line in conn.makefile("r", encoding="utf-8"):
                    if events.get("stop_recording", False):
                        break

                    try:
                        payload = json.loads(raw_line)
                        mode = RobotMode(payload["mode"])
                        color = ObjectColor(payload.get("color", ObjectColor.NONE.value))
                        raw_text = payload.get("raw_text", "")
                    except Exception as exc:
                        logging.warning("[Command] Invalid command payload: %r (%s)", raw_line, exc)
                        continue

                    with state_lock:
                        command_state.mode = mode
                        command_state.color = color
                        command_state.raw_text = raw_text
                        command_state.version += 1

                    logging.info("[Command] received mode=%s color=%s text=%r", mode.value, color.value, raw_text)

                    if mode == RobotMode.EXIT:
                        events["stop_recording"] = True
                        break


def start_command_server(
    command_state: CommandState,
    state_lock: Lock,
    events: dict,
    host: str = COMMAND_HOST,
    port: int = COMMAND_PORT,
) -> Thread:
    thread = Thread(
        target=command_server_loop,
        args=(command_state, state_lock, events, host, port),
        daemon=True,
    )
    thread.start()
    return thread


def send_command(text: str, host: str = COMMAND_HOST, port: int = COMMAND_PORT):
    mode, color = parse_command(text)
    payload = {
        "mode": mode.value,
        "color": color.value,
        "raw_text": text,
    }

    with socket.create_connection((host, port), timeout=2.0) as client:
        client.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

    return mode, color


def main():
    print(f"[Command] lerobot_record 명령 서버로 송신합니다: {COMMAND_HOST}:{COMMAND_PORT}")
    print("[Command] 예시: 초록 물체 잡아줘 / 빨간 물체 잡아줘 / 파란 물체 잡아줘 / 멈춰 / 초기화 / 종료")

    while True:
        try:
            text = input("[명령 입력] ").strip()
        except EOFError:
            break

        if not text:
            continue

        try:
            mode, color = send_command(text)
        except OSError as exc:
            print(f"[Command] 전송 실패: {exc}")
            print("[Command] lerobot_record.py가 먼저 실행되어 있는지 확인하세요.")
            continue

        print(f"[Command] 전송 완료: mode={mode.value}, color={color.value}")

        if mode == RobotMode.EXIT:
            break


def snapshot_command(command_state: Optional[CommandState], state_lock) -> tuple[RobotMode, ObjectColor, int]:
    if command_state is None or state_lock is None:
        return RobotMode.IDLE, ObjectColor.NONE, 0

    with state_lock:
        return command_state.mode, command_state.color, command_state.version



def set_command_idle(command_state: Optional[CommandState], state_lock):
    if command_state is None or state_lock is None:
        return

    with state_lock:
        command_state.mode = RobotMode.IDLE
        command_state.color = ObjectColor.NONE
        command_state.version += 1


def get_policy_path_for_color(target_color: ObjectColor) -> str | None:
    return POLICY_REGISTRY.get(target_color)


def get_chunk_size_for_color(target_color: ObjectColor) -> int | None:
    return CHUNK_REGISTRY.get(target_color)


if __name__ == "__main__":
    main()