"""
Video helper functions - Laravel-style
"""
import cv2
from typing import Tuple


def get_video_info(video_path: str) -> Tuple[int, int, int, int]:
    """Get video information (fps, total frames, width, height)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return fps, total_frames, width, height


def calculate_skip_frames(video_fps: int, target_fps: int) -> int:
    """Calculate how many frames to skip to achieve target FPS"""
    return max(1, video_fps // target_fps)
