import os
from pathlib import Path
from django.conf import settings


class AppConfig:
    """Application configuration"""

    def __init__(self):
        # Get from Django settings or use defaults
        self.frames_per_second: int = getattr(settings, "DEFAULT_FPS", 1)
        self.confidence_threshold: float = getattr(settings, "DEFAULT_CONFIDENCE", 0.5)
        self.weights_dir: str = getattr(settings, "WEIGHTS_DIR", "weights")
        # Use STATIC_ROOT for saving files (where files are actually stored)
        # In DEBUG mode, Django serves from STATICFILES_DIRS, but we save to STATIC_ROOT
        # This ensures files are accessible after collectstatic
        self.static_dir: str = getattr(
            settings, "STATIC_ROOT", os.path.join(settings.BASE_DIR, "staticfiles")
        )
        self.frames_dir: str = str(Path(self.static_dir) / "frames")
        self.selected_weight: str = getattr(
            settings, "DEFAULT_WEIGHT", "best_cricket.pt"
        )
        self.selected_classification_weight: str = getattr(
            settings, "DEFAULT_CLASSIFICATION_WEIGHT", "best_cricket_classify.pt"
        )

    def get_weight_path(self) -> str:
        """Get the full path to the selected weight"""
        return str(Path(self.weights_dir) / self.selected_weight)

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "frames_per_second": self.frames_per_second,
            "confidence_threshold": self.confidence_threshold,
            "selected_weight": self.selected_weight,
            "selected_classification_weight": self.selected_classification_weight,
            "weights_dir": self.weights_dir,
            "static_dir": self.static_dir,
        }
