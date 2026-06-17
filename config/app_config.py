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

        # DigitalOcean Spaces (S3-compatible) for frame storage.
        self.spaces_endpoint: str = getattr(settings, "SPACES_ENDPOINT", "")
        self.spaces_region: str = getattr(settings, "SPACES_REGION", "")
        self.spaces_bucket: str = getattr(settings, "SPACES_BUCKET", "")
        self.spaces_access_key_id: str = getattr(settings, "SPACES_ACCESS_KEY_ID", "")
        self.spaces_secret_access_key: str = getattr(
            settings, "SPACES_SECRET_ACCESS_KEY", ""
        )
        self.spaces_public_base_url: str = getattr(
            settings, "SPACES_PUBLIC_BASE_URL", ""
        )
        self.spaces_frames_prefix: str = getattr(
            settings, "SPACES_FRAMES_PREFIX", "frames"
        )

        self.ocr_provider: str = str(
            getattr(settings, "OCR_PROVIDER", "local")
        ).strip().lower() or "local"

        # External GLM OCR box (its own GPU). Speaks the Ollama /api/chat
        # protocol; point GLM_OCR_HOST at the deployed service.
        self.glm_ocr_host: str = getattr(
            settings, "GLM_OCR_HOST", "http://localhost:11434"
        )
        self.glm_ocr_model: str = getattr(settings, "GLM_OCR_MODEL", "glm-ocr")
        # Per-call ceiling for one GLM OCR request, retried on transient
        # failures by the engine. Kept below RQ_OCR_TIMEOUT.
        self.glm_ocr_timeout_seconds: float = float(
            getattr(settings, "GLM_OCR_TIMEOUT_SECONDS", 120)
        )
        self.glm_ocr_max_new_tokens: int = int(
            getattr(settings, "GLM_OCR_MAX_NEW_TOKENS", 2048)
        )
        # When true, call the worker's /parse (layout-aware) path so the result
        # carries per-region bounding boxes ("blocks"); else the text-only /ocr.
        self.glm_ocr_boxes: bool = bool(getattr(settings, "GLM_OCR_BOXES", False))
        self.glm_ocr_extract_prompt: str = getattr(
            settings,
            "GLM_OCR_EXTRACT_PROMPT",
            (
                "Extract all visible text from this image, preserving the "
                "original layout, line breaks, and reading order. Return "
                "only the extracted text."
            ),
        )

        # Stage-2 formatter provider switch: "deepseek" | "gemini".
        self.text_formatter_provider: str = str(
            getattr(settings, "TEXT_FORMATTER_PROVIDER", "deepseek")
        ).strip().lower() or "deepseek"

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

        # Gemini text API — alternative stage-2 formatter (Google AI Studio).
        self.gemini_text_api_key: str = getattr(
            settings, "GEMINI_TEXT_API_KEY", ""
        )
        self.gemini_text_base_url: str = getattr(
            settings,
            "GEMINI_TEXT_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        )
        self.gemini_text_model: str = getattr(
            settings, "GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite"
        )
        self.gemini_text_timeout_seconds: float = float(
            getattr(settings, "GEMINI_TEXT_TIMEOUT_SECONDS", 60)
        )
        self.gemini_text_max_tokens: int = int(
            getattr(settings, "GEMINI_TEXT_MAX_TOKENS", 2048)
        )
        self.gemini_text_temperature: float = float(
            getattr(settings, "GEMINI_TEXT_TEMPERATURE", 0.0)
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
