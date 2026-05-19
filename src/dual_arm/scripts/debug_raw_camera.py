import cv2
import os
from pathlib import Path

def debug_raw_capture(cam_id="/dev/video0", num_frames=5):
    save_dir = Path("debug_raw_frames")
    save_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {cam_id}")
        return

    # 현재 카메라가 사용 중인 포맷 확인
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"Current Camera Codec: {codec}")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # PNG는 무손실 압축이므로 여기서 깨져 보인다면 카메라 입력 자체가 문제인 것임
        fname = save_dir / f"raw_{i}.png"
        cv2.imwrite(str(fname), frame)
        print(f"Saved: {fname}")

    cap.release()

if __name__ == "__main__":
    # 문제가 있는 카메라 경로로 수정해서 테스트
    for cam in ["/dev/video0", "/dev/video4", "/dev/video6"]:
        if os.path.exists(cam):
            print(f"\n--- Testing {cam} ---")
            debug_raw_capture(cam)
