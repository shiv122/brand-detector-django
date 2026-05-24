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


def resize_for_ocr(
    frame: np.ndarray, max_dim: int = 1280
) -> np.ndarray:
    """Downscale so the longest edge is at most `max_dim`, preserving aspect.

    Smaller images send less bandwidth to the OCR endpoint and let the
    vision model attend to the same number of tokens worth of content
    without spending capacity on raw pixel resolution. No upscaling — if
    the source is already smaller than max_dim we return it unchanged.
    """
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_bytes_for_ocr(
    image_data: bytes, max_dim: int = 1280, quality: int = 90
) -> bytes:
    """Decode → resize (longest edge ≤ max_dim) → re-encode as JPEG.

    Returns the original bytes unchanged if decoding fails or the image
    is already small enough — never silently drops the OCR input.
    """
    arr = bytes_to_numpy(image_data)
    if arr is None:
        return image_data
    h, w = arr.shape[:2]
    if max(h, w) <= max_dim:
        return image_data
    resized = resize_for_ocr(arr, max_dim=max_dim)
    ok, encoded = cv2.imencode(
        ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        return image_data
    return encoded.tobytes()
