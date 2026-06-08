"""
Detection request validation classes - Laravel-style
"""
from apps.api.v1.requests.base_request import BaseRequest


class UpdateConfigRequest(BaseRequest):
    """Request validation for updating configuration"""

    def rules(self):
        self._required("frames_per_second", "frames_per_second")
        self._required("confidence_threshold", "confidence_threshold")
        self._integer("frames_per_second", min_value=1, max_value=30)
        self._float("confidence_threshold", min_value=0.0, max_value=1.0)


class SwitchWeightRequest(BaseRequest):
    """Request validation for switching weight"""

    def rules(self):
        self._required("weight_name", "weight_name")
        self._string("weight_name", min_length=1, max_length=255)


class DetectImagesRequest(BaseRequest):
    """Request validation for image detection (YOLO + optional classification)."""

    def rules(self):
        if "confidence_threshold" in self.data:
            self._float("confidence_threshold", min_value=0.0, max_value=1.0)
        else:
            self.data["confidence_threshold"] = 0.5

        if "weight_name" in self.data:
            self._string("weight_name", min_length=1, max_length=255)

        if "enable_classification" in self.data:
            self._boolean("enable_classification")
        else:
            self.data["enable_classification"] = False

        if "classification_weight_name" in self.data:
            self._string("classification_weight_name", min_length=1, max_length=255)


class DetectVideoRequest(BaseRequest):
    """Request validation for video detection"""

    def rules(self):
        has_file = "file" in self.files and self.files["file"]
        has_file_url = "file_url" in self.data and self.data.get("file_url")

        if not has_file and not has_file_url:
            self._add_error("file", "Either file or file_url must be provided")

        if has_file and has_file_url:
            self._add_error("file", "Provide either file or file_url, not both")

        if has_file_url:
            self._url("file_url")

        if "frames_per_second" in self.data:
            self._integer("frames_per_second", min_value=1, max_value=30)
        else:
            self.data["frames_per_second"] = 2

        if "confidence_threshold" in self.data:
            self._float("confidence_threshold", min_value=0.0, max_value=1.0)
        else:
            self.data["confidence_threshold"] = 0.5

        if "create_video" in self.data:
            self._boolean("create_video")
        else:
            self.data["create_video"] = False

        if "enable_classification" in self.data:
            self._boolean("enable_classification")
        else:
            self.data["enable_classification"] = False

        if "weight_name" in self.data:
            self._string("weight_name", min_length=1, max_length=255)

        if "classification_weight_name" in self.data:
            self._string("classification_weight_name", min_length=1, max_length=255)
