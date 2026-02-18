"""
Classification service for image classification - Laravel-style
"""

from typing import List, Dict, Optional
from rest_framework.response import Response
from rest_framework import status
from config.app_config import AppConfig
from apps.services.model.classification_model_service import ClassificationModelService
from apps.services.image.image_service import ImageService


class ClassificationResult:
    """Simple data class for classification results"""

    def __init__(self, class_id: int, class_name: str, confidence: float):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }


class ClassificationService:
    """Service for image classification"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.model_service = ClassificationModelService(config)
        self.image_service = ImageService()
        # Models are preloaded in ClassificationModelService.__init__

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_service.is_loaded()

    def get_available_weights(self) -> List[Dict]:
        """Get list of available classification weights"""
        return self.model_service.get_available_weights()

    def get_current_weight(self) -> str:
        """Get the currently selected weight"""
        return self.model_service.get_current_model_name()

    def switch_weight(self, weight_name: str) -> bool:
        """Switch to a different weight"""
        return self.model_service.switch_model(weight_name)

    def classify_image(
        self, image_data: bytes, top_k: int = 5,
        weight_name: Optional[str] = None
    ) -> List[ClassificationResult]:
        """Classify an image and return top-k predictions"""
        if not weight_name and not self.is_model_loaded():
            raise RuntimeError("Classification model not loaded")

        results = self.model_service.classify_image(image_data, top_k, weight_name=weight_name)

        classification_results = [
            ClassificationResult(
                class_id=r["class_id"],
                class_name=r["class_name"],
                confidence=r["confidence"],
            )
            for r in results
        ]

        return classification_results

    def switch_weight_handler(self, data: dict) -> Response:
        """Handle weight switching (data is already validated)"""
        weight_name = data["weight_name"]
        success = self.switch_weight(weight_name)
        
        if success:
            return Response({"message": f"Switched to weight: {weight_name}"})

        return Response(
            {"error": f"Failed to switch to weight: {weight_name}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    def classify_images_handler(self, request, validated_data: dict = None) -> Response:
        """Handle image classification request (data is already validated)"""
        weight_name = None
        if validated_data:
            weight_name = validated_data.get("weight_name")
        elif request.data:
            weight_name = request.data.get("weight_name")

        if not weight_name and not self.is_model_loaded():
            return Response(
                {"error": "Classification model not loaded"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        files = request.FILES.getlist("files")
        if validated_data:
            top_k = int(validated_data.get("top_k", 5))
        else:
            top_k = int(request.data.get("top_k", 5))

        results = []
        for file in files:
            if not self.image_service.validate_image_file(file.content_type, file.name):
                results.append(
                    {
                        "classifications": [],
                        "error": f"File {file.name} is not a valid image",
                    }
                )
                continue

            try:
                contents = file.read()
                classifications = self.classify_image(contents, top_k, weight_name=weight_name)

                results.append(
                    {
                        "classifications": [cls.to_dict() for cls in classifications],
                        "top_prediction": (
                            classifications[0].to_dict() if classifications else None
                        ),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "classifications": [],
                        "error": str(e),
                    }
                )

        return Response({"results": results})
