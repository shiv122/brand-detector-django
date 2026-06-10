"""
Shared service instances for API v1 controllers
This ensures services are only initialized once, preventing duplicate logs
"""
from django.conf import settings

from config.app_config import AppConfig
from apps.services.model.model_service import ModelService
from apps.services.image.image_service import ImageService
from apps.services.classification.classification_service import ClassificationService
from apps.services.counting.counting_service import CountingService
from apps.services.detection.detection_service import DetectionService
from apps.services.detection.remote_detection_client import RemoteDetectionClient
from apps.services.ocr.ocr_service import OcrService
from apps.services.storage import SpacesService

# Initialize services once - shared across all controllers
_config = AppConfig()
_model_service = ModelService(_config)
_image_service = ImageService()
_classification_service = ClassificationService(_config)
_counting_service = CountingService()
# Frames are uploaded to DigitalOcean Spaces; the public URL is what OCR and
# the dashboard use.
_spaces_service = SpacesService(_config)
# Shared OCR service: used on-demand by the OCR controller + RQ task, and
# (when enabled per request) by the detection path to OCR each frame during
# video processing.
_ocr_service = OcrService(_config)

# External detector service (its own GPU). When DETECTOR_HOST is set, detection +
# classification run there over HTTP instead of in-process; unset => local YOLO.
_detector_client = None
if getattr(settings, "DETECTOR_HOST", ""):
    _detector_client = RemoteDetectionClient(
        host=settings.DETECTOR_HOST,
        api_key=getattr(settings, "DETECTOR_API_KEY", ""),
        timeout_seconds=getattr(settings, "DETECTOR_TIMEOUT_SECONDS", 60.0),
        retries=getattr(settings, "DETECTOR_RETRIES", 3),
        retry_base_delay=getattr(settings, "DETECTOR_RETRY_BASE_DELAY", 1.0),
    )

_detection_service = DetectionService(
    _config,
    _model_service,
    _image_service,
    _classification_service,
    _counting_service,
    _spaces_service,
    _ocr_service,
    detector_client=_detector_client,
)

