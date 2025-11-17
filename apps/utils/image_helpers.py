"""
Image helper functions - Laravel-style
"""
import base64
import io
import cv2
import numpy as np
from PIL import Image
from typing import Optional


def image_to_base64(image_np: np.ndarray, quality: int = 85) -> str:
    """Convert numpy array image to base64 string"""
    try:
        # Convert BGR to RGB if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_np

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return ""


def save_frame(frame: np.ndarray, frame_path: str, quality: int = 85) -> bool:
    """Save a frame to disk"""
    try:
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        # Save using PIL for better quality control
        pil_image = Image.fromarray(frame_rgb)
        pil_image.save(frame_path, quality=quality, optimize=False)

        return True
    except Exception as e:
        print(f"Error saving frame {frame_path}: {str(e)}")
        return False


def bytes_to_numpy(image_data: bytes) -> Optional[np.ndarray]:
    """Convert image bytes to numpy array"""
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error converting bytes to numpy: {str(e)}")
        return None
