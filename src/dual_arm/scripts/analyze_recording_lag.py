import time
import logging
import numpy as np
from dual_arm.cameras.opencv.camera_opencv import OpenCVCamera
from dual_arm.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from dual_arm.cameras.configs import ColorMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_recording_timing(cam_id="/dev/video0", target_fps=30, duration=10):
    config = OpenCVCameraConfig(index_or_path=cam_id, color_mode=ColorMode.RGB, fps=target_fps)
    camera = OpenCVCamera(config)
    camera.connect()
    
    logger.info(f"Starting 10s timing test for {cam_id} at {target_fps} FPS...")
    
    timestamps = []
    frames_captured = 0
    
    start_time = time.perf_counter()
    end_time = start_time + duration
    
    # LeRobot의 실제 녹화 루프와 유사한 구조
    while time.perf_counter() < end_time:
        loop_start = time.perf_counter()
        
        try:
            # async_read가 새로운 프레임이 올 때까지 기다리는지 확인
            camera.async_read(timeout_ms=1000)
            frames_captured += 1
            timestamps.append(time.perf_counter())
        except Exception as e:
            logger.warning(f"Frame missed: {e}")
            
        # Target FPS 유지를 위한 대기
        elapsed = time.perf_counter() - loop_start
        sleep_time = max(0, (1.0 / target_fps) - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    total_time = time.perf_counter() - start_time
    camera.disconnect()
    
    expected_frames = target_fps * duration
    missing_frames = expected_frames - frames_captured
    
    print("\n" + "="*50)
    print(f"TIMING ANALYSIS: {cam_id}")
    print("="*50)
    print(f"Target Duration: {duration}s")
    print(f"Actual Duration: {total_time:.2f}s")
    print(f"Expected Frames: {expected_frames}")
    print(f"Actual Frames Captured: {frames_captured}")
    print(f"Missing Frames: {missing_frames} ({ (missing_frames/expected_frames)*100:.1f}%)")
    
    if len(timestamps) > 1:
        intervals = np.diff(timestamps)
        print(f"Avg Interval: {np.mean(intervals)*1000:.2f}ms (Target: {1000/target_fps:.2f}ms)")
        print(f"Max Interval: {np.max(intervals)*1000:.2f}ms")
        print(f"Jitter (Std Dev): {np.std(intervals)*1000:.2f}ms")
    print("="*50)

if __name__ == "__main__":
    # 문제가 가장 심한 카메라부터 테스트
    test_recording_timing("/dev/video0", target_fps=30)
