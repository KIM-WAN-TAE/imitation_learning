import time
import logging
import numpy as np
import concurrent.futures
from dual_arm.cameras.opencv.camera_opencv import OpenCVCamera
from dual_arm.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from dual_arm.cameras.configs import ColorMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def capture_thread(camera, target_fps, duration):
    frames = 0
    start_time = time.perf_counter()
    end_time = start_time + duration
    while time.perf_counter() < end_time:
        try:
            camera.async_read(timeout_ms=500)
            frames += 1
        except:
            pass
        # 타겟 FPS에 맞춘 최소 대기
        time.sleep(1.0 / (target_fps * 1.2)) 
    return frames

def main():
    cams = ["/dev/video0", "/dev/video4", "/dev/video6"]
    fps = 30
    duration = 10
    
    instances = []
    for c in cams:
        cfg = OpenCVCameraConfig(index_or_path=c, color_mode=ColorMode.RGB, fps=fps)
        cam = OpenCVCamera(cfg)
        cam.connect()
        instances.append(cam)
        
    logger.info(f"Testing SIMULTANEOUS capture for {len(instances)} cameras...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(instances)) as executor:
        futures = [executor.submit(capture_thread, cam, fps, duration) for cam in instances]
        results = [f.result() for f in futures]
        
    for i, res in enumerate(results):
        expected = fps * duration
        loss = (1 - res/expected) * 100
        print(f"Camera {cams[i]}: Captured {res}/{expected} frames ({loss:.1f}% loss)")
        
    for cam in instances:
        cam.disconnect()

if __name__ == "__main__":
    main()
