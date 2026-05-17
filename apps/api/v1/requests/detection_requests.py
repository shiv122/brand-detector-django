"""
Detection request validation classes - Laravel-style
"""
import json
from typing import Dict, Any, Optional
from apps.api.v1.requests.base_request import BaseRequest


_OCR_SCOPES = ["frame", "detection", "both"]


def _parse_ocr_fields(req: BaseRequest, video: bool = False):
    """Shared OCR field parsing for image + video detection requests."""
    if "enable_ocr" in req.data:
        req._boolean("enable_ocr")
    else:
        req.data["enable_ocr"] = False

    if "ocr_scope" in req.data and req.data["ocr_scope"]:
        req._in("ocr_scope", _OCR_SCOPES)
    else:
        req.data["ocr_scope"] = "detection"

    if "ocr_template_key" in req.data and req.data["ocr_template_key"]:
        req._string("ocr_template_key", min_length=1, max_length=128)
    else:
        req.data["ocr_template_key"] = "raw"

    if "ocr_custom_prompt" in req.data and req.data["ocr_custom_prompt"]:
        req._string("ocr_custom_prompt", max_length=8192)
    else:
        req.data["ocr_custom_prompt"] = ""

    # ocr_class_filter: optional list of class_names. Accept JSON array or
    # comma-separated string. Empty/missing = apply to all classes.
    raw_filter = req.data.get("ocr_class_filter")
    classes: list = []
    if raw_filter:
        if isinstance(raw_filter, list):
            classes = [str(c).strip() for c in raw_filter if str(c).strip()]
        elif isinstance(raw_filter, str):
            stripped = raw_filter.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        classes = [str(c).strip() for c in parsed if str(c).strip()]
                    else:
                        req._add_error(
                            "ocr_class_filter",
                            "ocr_class_filter must be a JSON array of strings",
                        )
                except json.JSONDecodeError:
                    req._add_error(
                        "ocr_class_filter",
                        "ocr_class_filter must be a JSON array or comma-separated string",
                    )
            else:
                classes = [c.strip() for c in stripped.split(",") if c.strip()]
    req.data["ocr_class_filter"] = classes

    if video:
        if "ocr_every_n_frames" in req.data:
            req._integer("ocr_every_n_frames", min_value=1, max_value=600)
        else:
            req.data["ocr_every_n_frames"] = 1


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

        if "weight_name" in self.data:
            self._string("weight_name", min_length=1, max_length=255)

        _parse_ocr_fields(self, video=False)


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

        if "weight_name" in self.data:
            self._string("weight_name", min_length=1, max_length=255)

        if "classification_weight_name" in self.data:
            self._string("classification_weight_name", min_length=1, max_length=255)

        _parse_ocr_fields(self, video=True)

