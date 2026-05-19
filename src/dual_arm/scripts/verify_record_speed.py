import time
import logging
import numpy as np
from pathlib import Path
from dual_arm.cameras.opencv.camera_opencv import OpenCVCamera
from dual_arm.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from dual_arm.cameras.realsense.camera_realsense import RealSenseCamera
from dual_arm.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from dual_arm.cameras.configs import ColorMode
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_actual_record_test(fps=15, duration=10):
    # 1. 카메라 설정 (사용자 사양)
    cv_cfg = OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=fps, color_mode=ColorMode.RGB)
    rs_cfg = RealSenseCameraConfig(serial_number_or_name="048522072070", width=640, height=480, fps=fps, color_mode=ColorMode.RGB)
    
    laptop = OpenCVCamera(cv_cfg)
    realsense = RealSenseCamera(rs_cfg)
    
    try:
        logger.info("Connecting to cameras...")
        laptop.connect()
        realsense.connect()
        
        target_frames = fps * duration
        target_interval = 1.0 / fps
        
        frame_timestamps = []
        loop_durations = []
        
        logger.info(f"Target: {duration}s recording at {fps} FPS ({target_frames} frames)")
        
        start_time = time.perf_counter()
        # 실제 LeRobot 루프 구조 재현
        for i in range(target_frames):
            loop_start = time.perf_counter()
            
            # [CAPTURE]
            img_cv = laptop.async_read()
            img_rs = realsense.async_read()
            
            # [TIMING CHECK]
            now = time.perf_counter()
            frame_timestamps.append(now - start_time)
            
            # [SIMULATED PROCESSING/WRITE]
            # 실제 PNG 저장은 CPU 부하가 크므로 30ms 정도의 지연을 의도적으로 주어 부하 시뮬레이션
            time.sleep(0.03) 
            
            loop_end = time.perf_counter()
            loop_durations.append(loop_end - loop_start)
            
            # 다음 프레임까지 대기 (LeRobot의 정밀 제어 로직)
            elapsed = loop_end - loop_start
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if i % 15 == 0:
                logger.info(f"Progress: {i}/{target_frames} frames ({ (i/target_frames)*100:.1f}%)")

        total_actual_time = time.perf_counter() - start_time
        
        print("\n" + "="*60)
        print("RECORDING ACTUAL EXECUTION ANALYSIS")
        print("="*60)
        print(f"Total Wall-Clock Time: {total_actual_time:.2f}s (Target: {duration}s)")
        print(f"Total Frames Captured: {len(frame_timestamps)}")
        
        # 여기서 '3초만 저장되는 문제'의 원인 분석
        # 만약 저장 루프가 너무 느려서 10초 동안 45프레임만 찍혔다면?
        # 45프레임 / 15 FPS = 3초짜리 영상이 됨.
        
        avg_loop = np.mean(loop_durations) * 1000
        print(f"Average Loop Duration: {avg_loop:.2f}ms (Limit for 15FPS: {target_interval*1000:.2f}ms)")
        
        if total_actual_time > duration * 1.1:
            actual_fps = len(frame_timestamps) / total_actual_time
            print(f"\n[경고] 루프 지연이 심각합니다!")
            print(f"실제 초당 저장 속도: {actual_fps:.2f} FPS (목표 15 FPS 대비 {(actual_fps/fps)*100:.1f}% 성능)")
            print(f"결과물 비디오 시간: {len(frame_timestamps) / fps:.2f}s (이 수치가 3초에 가깝다면 병목이 원인)")
        else:
            print(f"\n[정상] 루프 시간이 안정적입니다.")
            print(f"비디오 예상 시간: {len(frame_timestamps) / fps:.2f}s")

        print("="*60)

    finally:
        laptop.disconnect()
        realsense.disconnect()

if __name__ == "__main__":
    run_actual_record_test()
