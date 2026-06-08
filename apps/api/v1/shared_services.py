"""
Shared service instances for API v1 controllers
This ensures services are only initialized once, preventing duplicate logs
"""
from config.app_config import AppConfig
from apps.services.model.model_service import ModelService
from apps.services.image.image_service import ImageService
from apps.services.classification.classification_service import ClassificationService
from apps.services.counting.counting_service import CountingService
from apps.services.detection.detection_service import DetectionService
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
_detection_service = DetectionService(
    _config,
    _model_service,
    _image_service,
    _classification_service,
    _counting_service,
    _spaces_service,
    _ocr_service,
)

