# Request validation classes - Laravel-style
from .detection_requests import (
    UpdateConfigRequest,
    SwitchWeightRequest,
    DetectImagesRequest,
    DetectVideoRequest,
)
from .classification_requests import (
    SwitchClassificationWeightRequest,
    ClassifyImagesRequest,
)

__all__ = [
    "UpdateConfigRequest",
    "SwitchWeightRequest",
    "DetectImagesRequest",
    "DetectVideoRequest",
    "SwitchClassificationWeightRequest",
    "ClassifyImagesRequest",
]
