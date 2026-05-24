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
from .ocr_requests import OcrRunRequest, SportPromptUpsertRequest

__all__ = [
    "UpdateConfigRequest",
    "SwitchWeightRequest",
    "DetectImagesRequest",
    "DetectVideoRequest",
    "SwitchClassificationWeightRequest",
    "ClassifyImagesRequest",
    "OcrRunRequest",
    "SportPromptUpsertRequest",
]
