"""
Image service for image processing operations - Laravel-style
"""
from apps.utils.image_helpers import image_to_base64, save_frame, bytes_to_numpy
from apps.utils.video_helpers import get_video_info, calculate_skip_frames
from apps.utils.validators import validate_image_file, validate_video_file


class ImageService:
    """Service for image operations"""

    def validate_image_file(self, content_type: str = None, filename: str = None) -> bool:
        """Validate if file is an image"""
        return validate_image_file(content_type, filename)

    def validate_video_file(self, content_type: str = None, filename: str = None) -> bool:
        """Validate if file is a video"""
        return validate_video_file(content_type, filename)

    def image_to_base64(self, image_np, quality: int = 85) -> str:
        """Convert numpy array image to base64 string"""
        return image_to_base64(image_np, quality)

    def save_frame(self, frame, frame_path: str, quality: int = 85) -> bool:
        """Save a frame to disk"""
        return save_frame(frame, frame_path, quality)

    def bytes_to_numpy(self, image_data: bytes):
        """Convert image bytes to numpy array"""
        return bytes_to_numpy(image_data)

    def get_video_info(self, video_path: str):
        """Get video information"""
        return get_video_info(video_path)

    def calculate_skip_frames(self, video_fps: int, target_fps: int) -> int:
        """Calculate how many frames to skip"""
        return calculate_skip_frames(video_fps, target_fps)
