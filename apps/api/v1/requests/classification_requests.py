"""
Classification request validation classes - Laravel-style
"""
from apps.api.v1.requests.base_request import BaseRequest


class SwitchClassificationWeightRequest(BaseRequest):
    """Request validation for switching classification weight"""

    def rules(self):
        """Validation rules"""
        self._required("weight_name", "weight_name")
        self._string("weight_name", min_length=1, max_length=255)


class ClassifyImagesRequest(BaseRequest):
    """Request validation for image classification"""

    def rules(self):
        """Validation rules"""
        # Files are validated separately in controller
        if "top_k" in self.data:
            self._integer("top_k", min_value=1, max_value=20)
        else:
            # Set default if not provided
            self.data["top_k"] = 5

