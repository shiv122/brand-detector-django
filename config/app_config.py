import os
from pathlib import Path
from django.conf import settings


class AppConfig:
    """Application configuration."""

    def __init__(self):
        self.frames_per_second: int = getattr(settings, "DEFAULT_FPS", 1)
        self.confidence_threshold: float = getattr(settings, "DEFAULT_CONFIDENCE", 0.5)
        self.weights_dir: str = getattr(settings, "WEIGHTS_DIR", "weights")
        self.static_dir: str = getattr(
            settings, "STATIC_ROOT", os.path.join(settings.BASE_DIR, "staticfiles")
        )
        self.frames_dir: str = str(Path(self.static_dir) / "frames")
        self.selected_weight: str = getattr(
            settings, "DEFAULT_WEIGHT", "cricket.pt"
        )
        self.selected_classification_weight: str = getattr(
            settings, "DEFAULT_CLASSIFICATION_WEIGHT", "cricket_classify.pt"
        )

        # Only "local" remains: GLM-OCR (remote Ollama HTTP) extracts text,
        # then the DeepSeek text API formats it into JSON.
        self.ocr_provider: str = str(
            getattr(settings, "OCR_PROVIDER", "local")
        ).strip().lower() or "local"

        # Remote Ollama serving glm-ocr — points at the GLM_OCR container.
        self.local_ocr_ollama_host: str = getattr(
            settings, "LOCAL_OCR_OLLAMA_HOST", "http://localhost:11434"
        )
        self.local_ocr_ollama_model: str = getattr(
            settings, "LOCAL_OCR_OLLAMA_MODEL", "glm-ocr"
        )
        self.local_ocr_ollama_timeout_seconds: float = float(
            getattr(settings, "LOCAL_OCR_OLLAMA_TIMEOUT_SECONDS", 180)
        )
        self.local_ocr_max_new_tokens: int = int(
            getattr(settings, "LOCAL_OCR_MAX_NEW_TOKENS", 2048)
        )
        self.local_ocr_extract_prompt: str = getattr(
            settings,
            "LOCAL_OCR_EXTRACT_PROMPT",
            (
                "Extract all visible text from this image, preserving the "
                "original layout, line breaks, and reading order. Return "
                "only the extracted text."
            ),
        )

        # DeepSeek text API — formats GLM-OCR text into JSON using the
        # sport prompt.
        self.deepseek_text_api_key: str = getattr(
            settings, "DEEPSEEK_TEXT_API_KEY", ""
        )
        self.deepseek_text_base_url: str = getattr(
            settings, "DEEPSEEK_TEXT_BASE_URL", "https://api.deepseek.com/v1"
        )
        self.deepseek_text_model: str = getattr(
            settings, "DEEPSEEK_TEXT_MODEL", "deepseek-chat"
        )
        self.deepseek_text_timeout_seconds: float = float(
            getattr(settings, "DEEPSEEK_TEXT_TIMEOUT_SECONDS", 60)
        )
        self.deepseek_text_max_tokens: int = int(
            getattr(settings, "DEEPSEEK_TEXT_MAX_TOKENS", 2048)
        )
        self.deepseek_text_temperature: float = float(
            getattr(settings, "DEEPSEEK_TEXT_TEMPERATURE", 0.0)
        )

    def get_weight_path(self) -> str:
        return str(Path(self.weights_dir) / self.selected_weight)

    def to_dict(self):
        return {
            "frames_per_second": self.frames_per_second,
            "confidence_threshold": self.confidence_threshold,
            "selected_weight": self.selected_weight,
            "selected_classification_weight": self.selected_classification_weight,
            "weights_dir": self.weights_dir,
            "static_dir": self.static_dir,
        }
