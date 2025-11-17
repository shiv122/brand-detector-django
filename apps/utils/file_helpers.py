"""
File helper functions - Laravel-style
"""
from pathlib import Path
import secrets
import time


def generate_unique_filename(original_filename: str, prefix: str = "uploaded") -> str:
    """Generate a unique filename with timestamp and random suffix"""
    timestamp = int(time.time())
    random_suffix = secrets.token_hex(4)
    extension = Path(original_filename).suffix
    return f"{prefix}_{timestamp}_{random_suffix}{extension}"


def ensure_directory_exists(directory_path: str) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()


def is_valid_path(path: str) -> bool:
    """Check if path is valid"""
    try:
        Path(path)
        return True
    except Exception:
        return False
