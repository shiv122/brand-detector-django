"""
File validation helper functions - Laravel-style
"""
from typing import Optional
from apps.core.enums import FileType


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


def validate_image_file(content_type: Optional[str], filename: Optional[str]) -> bool:
    """Validate if file is an image"""
    # Check MIME type
    if content_type and content_type.startswith("image/"):
        return True

    # Check file extension
    if filename:
        file_extension = "." + filename.split(".")[-1].lower()
        return file_extension in IMAGE_EXTENSIONS

    return False


def validate_video_file(content_type: Optional[str], filename: Optional[str]) -> bool:
    """Validate if file is a video"""
    # Check MIME type
    if content_type and content_type.startswith("video/"):
        return True

    # Check file extension
    if filename:
        file_extension = "." + filename.split(".")[-1].lower()
        return file_extension in VIDEO_EXTENSIONS

    return False


def get_file_type(content_type: Optional[str], filename: Optional[str]) -> Optional[FileType]:
    """Get file type from content type or filename"""
    if validate_image_file(content_type, filename):
        return FileType.IMAGE
    elif validate_video_file(content_type, filename):
        return FileType.VIDEO
    return None
