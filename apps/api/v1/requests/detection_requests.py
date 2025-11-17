"""
Detection request validation classes - Laravel-style
"""
from typing import Dict, Any, Optional
from apps.api.v1.requests.base_request import BaseRequest


class UpdateConfigRequest(BaseRequest):
    """Request validation for updating configuration"""

    def rules(self):
        """Validation rules"""
        self._required("frames_per_second", "frames_per_second")
        self._required("confidence_threshold", "confidence_threshold")
        self._integer("frames_per_second", min_value=1, max_value=30)
        self._float("confidence_threshold", min_value=0.0, max_value=1.0)


class SwitchWeightRequest(BaseRequest):
    """Request validation for switching weight"""

    def rules(self):
        """Validation rules"""
        self._required("weight_name", "weight_name")
        self._string("weight_name", min_length=1, max_length=255)


class DetectImagesRequest(BaseRequest):
    """Request validation for image detection"""

    def rules(self):
        """Validation rules"""
        # Files are validated separately in controller
        if "confidence_threshold" in self.data:
            self._float("confidence_threshold", min_value=0.0, max_value=1.0)
        else:
            # Set default if not provided
            self.data["confidence_threshold"] = 0.5


class DetectVideoRequest(BaseRequest):
    """Request validation for video detection"""

    def rules(self):
        """Validation rules"""
        # Either file or file_url must be provided
        has_file = "file" in self.files and self.files["file"]
        has_file_url = "file_url" in self.data and self.data.get("file_url")

        if not has_file and not has_file_url:
            self._add_error("file", "Either file or file_url must be provided")

        if has_file and has_file_url:
            self._add_error("file", "Provide either file or file_url, not both")

        # Validate file_url if provided
        if has_file_url:
            self._url("file_url")

        # Validate optional parameters
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

