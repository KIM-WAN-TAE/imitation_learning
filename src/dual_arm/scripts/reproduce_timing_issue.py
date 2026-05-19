import time
import logging
import os
from pathlib import Path
import numpy as np
from dual_arm.cameras.opencv.camera_opencv import OpenCVCamera
from dual_arm.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from dual_arm.cameras.realsense.camera_realsense import RealSenseCamera
from dual_arm.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from dual_arm.cameras.configs import ColorMode
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dual_camera_timing(fps=15, duration=10):
    # 1. 카메라 설정 (사용자 요청 사양)
    cv_cfg = OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=fps, color_mode=ColorMode.RGB)
    rs_cfg = RealSenseCameraConfig(serial_number_or_name="048522072070", width=640, height=480, fps=fps, color_mode=ColorMode.RGB)
    
    laptop = OpenCVCamera(cv_cfg)
    realsense = RealSenseCamera(rs_cfg)
    
    try:
        logger.info("Connecting to cameras...")
        laptop.connect()
        realsense.connect()
        
        save_dir = Path("timing_test_frames")
        save_dir.mkdir(exist_ok=True)
        
        target_interval = 1.0 / fps
        frames_captured = 0
        latencies = []
        capture_times = []
        
        logger.info(f"Starting 10s test at {fps} FPS (Target interval: {target_interval*1000:.2f}ms)")
        start_test = time.perf_counter()
        end_test = start_test + duration
        
        while time.perf_counter() < end_test:
            loop_start = time.perf_counter()
            
            # [Step 1] 카메라 읽기 (병목 지점 1)
            t0 = time.perf_counter()
            img_laptop = laptop.async_read(timeout_ms=1000)
            t1 = time.perf_counter()
            img_rs = realsense.async_read(timeout_ms=1000)
            t2 = time.perf_counter()
            
            read_latency = t2 - t0
            
            # [Step 2] 저장 시뮬레이션 (병목 지점 2)
            # 실제 PNG 저장은 훨씬 느리므로, 최소한의 I/O 부하 시뮬레이션
            cv2.imwrite(str(save_dir / "tmp_laptop.png"), cv2.cvtColor(img_laptop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_dir / "tmp_rs.png"), cv2.cvtColor(img_rs, cv2.COLOR_RGB2BGR))
            t3 = time.perf_counter()
            
            write_latency = t3 - t2
            total_latency = t3 - loop_start
            
            latencies.append({
                "read": read_latency,
                "write": write_latency,
                "total": total_latency
            })
            
            frames_captured += 1
            
            # 다음 프레임까지 대기
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"Loop delayed! Total latency: {total_latency*1000:.2f}ms (Exceeded {target_interval*1000:.2f}ms)")

        actual_duration = time.perf_counter() - start_test
        
        print("\n" + "="*60)
        print("DUAL CAMERA TIMING ANALYSIS")
        print("="*60)
        print(f"Test Duration: {actual_duration:.2f}s")
        print(f"Target FPS: {fps}")
        print(f"Expected Frames: {fps * duration}")
        print(f"Actual Frames: {frames_captured}")
        print(f"Missing Frames: {(fps * duration) - frames_captured}")
        
        avg_read = np.mean([l['read'] for l in latencies]) * 1000
        avg_write = np.mean([l['write'] for l in latencies]) * 1000
        avg_total = np.mean([l['total'] for l in latencies]) * 1000
        
        print(f"Avg Read Latency: {avg_read:.2f}ms")
        print(f"Avg Write Latency: {avg_write:.2f}ms")
        print(f"Avg Total Loop Time: {avg_total:.2f}ms")
        
        if avg_total > target_interval * 1000:
            print(f"\n[결론] 루프 지연 발생! 평균 루프 시간이 목표({target_interval*1000:.2f}ms)를 초과합니다.")
            print("이 상태로 녹화하면 영상 저장 시간이 실제보다 짧게 기록됩니다.")
        else:
            print("\n[결론] 루프 시간은 안정적입니다. 다른 I/O 병목(디스크 쓰기 등)을 확인해야 합니다.")
        print("="*60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        laptop.disconnect()
        realsense.disconnect()

if __name__ == "__main__":
    test_dual_camera_timing()
