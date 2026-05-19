import time
import logging
from dual_arm.cameras.opencv.camera_opencv import OpenCVCamera
from dual_arm.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from dual_arm.cameras.realsense.camera_realsense import RealSenseCamera
from dual_arm.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from dual_arm.cameras.configs import ColorMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def measure_fps(camera, duration=5.0):
    logger.info(f"Measuring FPS for {camera} for {duration} seconds...")
    frame_count = 0
    start_time = time.perf_counter()
    end_time = start_time + duration
    
    while time.perf_counter() < end_time:
        try:
            # use async_read to avoid blocking indefinitely if a frame is missed
            camera.async_read(timeout_ms=500)
            frame_count += 1
        except Exception as e:
            logger.warning(f"Error reading frame: {e}")
            
    actual_duration = time.perf_counter() - start_time
    measured_fps = frame_count / actual_duration
    return measured_fps, frame_count, actual_duration

def main():
    # 1. Find OpenCV cameras
    opencv_cameras = OpenCVCamera.find_cameras()
    for cam_info in opencv_cameras:
        cam_id = cam_info["id"]
        config = OpenCVCameraConfig(index_or_path=cam_id, color_mode=ColorMode.RGB, fps=30)
        camera = OpenCVCamera(config)
        try:
            camera.connect()
            fps, count, dur = measure_fps(camera)
            logger.info(f"OpenCV Camera {cam_id}: Measured FPS = {fps:.2f} ({count} frames in {dur:.2f}s)")
            camera.disconnect()
        except Exception as e:
            logger.error(f"Failed to test OpenCV camera {cam_id}: {e}")

    # 2. Find RealSense cameras
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            cam_id = cam_info["id"]
            config = RealSenseCameraConfig(serial_number_or_name=cam_id, color_mode=ColorMode.RGB, fps=30)
            camera = RealSenseCamera(config)
            try:
                camera.connect()
                fps, count, dur = measure_fps(camera)
                logger.info(f"RealSense Camera {cam_id}: Measured FPS = {fps:.2f} ({count} frames in {dur:.2f}s)")
                camera.disconnect()
            except Exception as e:
                logger.error(f"Failed to test RealSense camera {cam_id}: {e}")
    except Exception as e:
        logger.warning(f"RealSense not available or error: {e}")

if __name__ == "__main__":
    main()
