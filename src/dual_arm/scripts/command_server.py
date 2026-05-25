import socket
import threading
import json

HOST = "0.0.0.0"
PORT = 8765


def normalize_command(data_text):
    """
    클라이언트가 'green' 문자열을 보내도 처리하고,
    {"mode": "policy", "color": "green", "raw_text": "green"} JSON을 보내도 처리
    """

    data_text = data_text.strip()

    try:
        data = json.loads(data_text)

        if isinstance(data, dict):
            # color가 있으면 color 우선 사용
            if data.get("color") not in [None, "", "none"]:
                return data.get("color").strip().lower()

            # color가 없으면 raw_text 사용
            if data.get("raw_text"):
                return data.get("raw_text").strip().lower()

            # mode만 있는 stop/home/exit 같은 경우
            if data.get("mode"):
                return data.get("mode").strip().lower()

    except json.JSONDecodeError:
        pass

    # JSON이 아니면 기존처럼 일반 문자열 명령으로 처리
    return data_text.lower()


def handle_command(command):
    command = normalize_command(command)

    if command == "green":
        print("[CMD] GREEN 명령 수신")
        # TODO: green 동작

    elif command == "red":
        print("[CMD] RED 명령 수신")
        # TODO: red 동작

    elif command == "blue":
        print("[CMD] BLUE 명령 수신")
        # TODO: blue 동작

    elif command == "stop":
        print("[CMD] STOP 명령 수신")
        # TODO: 정지 동작

    elif command == "home":
        print("[CMD] HOME 명령 수신")
        # TODO: 홈 위치 이동

    elif command == "exit":
        print("[CMD] EXIT 명령 수신")

    else:
        print(f"[WARN] 알 수 없는 명령: {command}")

    return "ok"


def handle_client(conn, addr):
    print(f"[+] 연결됨: {addr}")

    try:
        data = conn.recv(1024)

        if not data:
            return

        data_text = data.decode("utf-8").strip()
        print(f"[RECV] {data_text}")

        result = handle_command(data_text)

        conn.sendall(result.encode("utf-8"))

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        conn.close()
        print(f"[-] 연결 종료: {addr}")


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((HOST, PORT))
    server.listen(5)

    print(f"[*] Command Server 실행 중: {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()

        thread = threading.Thread(
            target=handle_client,
            args=(conn, addr),
            daemon=True
        )
        thread.start()


if __name__ == "__main__":
    start_server()