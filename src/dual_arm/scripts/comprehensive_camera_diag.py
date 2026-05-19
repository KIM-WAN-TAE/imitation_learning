import time
import logging
import os
import shutil
import numpy as np
import cv2
import psutil
from pathlib import Path
from dual_arm.cameras.opencv.camera_opencv import OpenCVCamera
from dual_arm.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from dual_arm.cameras.configs import ColorMode
from dual_arm.datasets.video_utils import encode_video_frames

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_system_resources():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"System Check: CPU Usage: {cpu_usage}%, Memory: {memory.percent}% free")
    if cpu_usage > 80:
        logger.warning("High CPU usage detected. This can cause frame drops and encoding issues.")

def analyze_image_quality(frame):
    # Calculate basic stats to see if the image is too dark or noisy
    mean = np.mean(frame)
    std = np.std(frame)
    # Convert to grayscale for entropy-like measure (sharpness/detail)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        "mean_brightness": mean,
        "std_dev": std,
        "sharpness_score": laplacian_var
    }

def run_diagnostic(cam_id, target_fps=15, num_frames=100):
    diag_dir = Path(f"diag_results_{cam_id.replace('/', '_')}")
    diag_dir.mkdir(exist_ok=True)
    img_dir = diag_dir / "frames"
    img_dir.mkdir(exist_ok=True)
    
    config = OpenCVCameraConfig(index_or_path=cam_id, color_mode=ColorMode.RGB, fps=target_fps)
    camera = OpenCVCamera(config)
    
    try:
        camera.connect()
        logger.info(f"Connected to {cam_id}. Starting capture of {num_frames} frames...")
        
        frames = []
        timestamps = []
        
        # 1. Capture Phase
        start_capture = time.perf_counter()
        for i in range(num_frames):
            t_loop = time.perf_counter()
            frame = camera.read()
            frames.append(frame)
            timestamps.append(time.perf_counter())
            
            # Save raw PNG to check if input is good
            cv2.imwrite(str(img_dir / f"frame-{i:06d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Simulate real loop timing
            elapsed = time.perf_counter() - t_loop
            sleep_time = max(0, (1.0 / target_fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        end_capture = time.perf_counter()
        actual_fps = num_frames / (end_capture - start_capture)
        
        # 2. Analyze Capture Performance
        logger.info(f"[{cam_id}] Actual Capture FPS: {actual_fps:.2f}")
        
        # 3. Analyze Image Content
        sample_frame = frames[num_frames//2]
        quality_stats = analyze_image_quality(sample_frame)
        logger.info(f"[{cam_id}] Image Stats: {quality_stats}")
        
        # 4. Encoding Simulation
        # Default LeRobot settings
        video_default = diag_dir / "video_default.mp4"
        logger.info(f"[{cam_id}] Encoding with default LeRobot settings (libsvtav1, crf=30, preset=12)...")
        encode_video_frames(img_dir, video_default, target_fps, vcodec="libsvtav1", crf=30, preset=12, overwrite=True)
        
        # High Quality settings
        video_hq = diag_dir / "video_hq.mp4"
        logger.info(f"[{cam_id}] Encoding with High Quality settings (h264, crf=18, slow)...")
        # Note: encode_video_frames doesn't expose preset for h264 easily in the wrapper, but let's try h264
        encode_video_frames(img_dir, video_hq, target_fps, vcodec="h264", crf=18, overwrite=True)

        logger.info(f"[{cam_id}] Default Video Size: {os.path.getsize(video_default) / 1024:.1f} KB")
        logger.info(f"[{cam_id}] HQ Video Size: {os.path.getsize(video_hq) / 1024:.1f} KB")
        
        camera.disconnect()
        return {
            "cam_id": cam_id,
            "actual_fps": actual_fps,
            "stats": quality_stats,
            "default_size": os.path.getsize(video_default),
            "hq_size": os.path.getsize(video_hq)
        }

    except Exception as e:
        logger.error(f"Error during diagnostic for {cam_id}: {e}")
        if camera.is_connected:
            camera.disconnect()
        return None

def main():
    check_system_resources()
    
    target_cams = ["/dev/video0", "/dev/video4", "/dev/video6"]
    results = []
    
    for cam in target_cams:
        if os.path.exists(cam):
            res = run_diagnostic(cam)
            if res:
                results.append(res)
        else:
            logger.warning(f"Camera {cam} not found.")

    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY")
    print("="*50)
    for r in results:
        print(f"Camera: {r['cam_id']}")
        print(f"  - Actual FPS: {r['actual_fps']:.2f}")
        print(f"  - Brightness: {r['stats']['mean_brightness']:.2f} (0-255)")
        print(f"  - Sharpness (Laplacian Var): {r['stats']['sharpness_score']:.2f}")
        print(f"  - Compression Ratio (HQ size / Default size): {r['hq_size'] / r['default_size']:.2f}x")
        if r['stats']['mean_brightness'] < 40:
            print("  ! WARNING: Image is very dark. This increases noise and hurts compression.")
        if r['stats']['sharpness_score'] < 100:
            print("  ! WARNING: Image is blurry. Check focus.")
        if r['actual_fps'] < 14:
            print("  ! WARNING: Capture FPS is lower than target (15). Possible USB bottleneck.")
    print("="*50)

if __name__ == "__main__":
    main()
