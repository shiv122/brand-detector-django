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

# Initialize services once - shared across all controllers
_config = AppConfig()
_model_service = ModelService(_config)
_image_service = ImageService()
_classification_service = ClassificationService(_config)
_counting_service = CountingService()
_detection_service = DetectionService(
    _config,
    _model_service,
    _image_service,
    _classification_service,
    _counting_service,
)

