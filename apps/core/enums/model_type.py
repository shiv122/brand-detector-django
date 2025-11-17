from enum import Enum


class ModelType(str, Enum):
    """Type of ML model"""

    DETECTION = "detection"
    CLASSIFICATION = "classification"

    def __str__(self):
        return self.value
