import os
import sys
from flask import Flask, render_template, request, jsonify

# 현재 디렉토리를 path에 추가하여 command_control 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from command_control import send_command, COMMAND_HOST, COMMAND_PORT
except ImportError:
    # 패키지 형태로 실행될 경우를 대비
    from dual_arm.scripts.command_control import send_command, COMMAND_HOST, COMMAND_PORT

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send')
def handle_command():
    cmd = request.args.get('command', '')
    
    # command_control.py의 parse_command가 인식할 수 있는 텍스트로 매핑
    mapping = {
        "green": "green",
        "red": "red",
        "blue": "blue",
        "package_start": "package_start",
        "package_complete": "package_complete",
        "stop": "stop",
        "home": "home",
        "exit": "exit",
    }
    
    if cmd not in mapping:
        return jsonify({"status": "error", "message": "잘못된 명령입니다."})
    
    try:
        # command_control.py의 send_command 함수를 호출하여 로봇 서버(8765 포트)로 전송
        mode, color = send_command(mapping[cmd], host=COMMAND_HOST, port=COMMAND_PORT)
        return jsonify({
            "status": "success", 
            "message": f"{cmd.upper()} 전송됨 (mode={mode.value}, color={color.value})"
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"로봇 연결 실패: {str(e)}. lerobot_record.py가 실행 중인지 확인하세요."
        })

if __name__ == '__main__':
    print(f"[*] Dual Arm Web UI 가동 중... http://0.0.0.0:5000")
    # debug=True는 개발용입니다. 배포 시에는 False로 변경하세요.
    app.run(host='0.0.0.0', port=5000, debug=True)
