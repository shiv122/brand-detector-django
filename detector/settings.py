"""
Django settings for detector project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("SECRET_KEY", "django-insecure-change-this-in-production")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third party
    "rest_framework",
    "corsheaders",
    "drf_spectacular",
    "django_rq",
    # Local apps
    "apps.core",
    "apps.api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # Fast static file serving
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# Disable APPEND_SLASH to match FastAPI behavior (no trailing slashes required)
APPEND_SLASH = False

ROOT_URLCONF = "detector.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "detector.wsgi.application"

# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
# STATICFILES_DIRS is for source files (before collectstatic)
# STATIC_ROOT is where files are collected to (and where we save frames/videos)
# Only include static directory if it exists to avoid warnings
static_dir = os.path.join(BASE_DIR, "static")
STATICFILES_DIRS = (
    [
        static_dir,
    ]
    if os.path.exists(static_dir)
    else []
)

# WhiteNoise configuration for fast static file serving
# WhiteNoise serves files directly from STATIC_ROOT with compression and caching
WHITENOISE_USE_FINDERS = False  # Don't use finders, serve directly from STATIC_ROOT
WHITENOISE_AUTOREFRESH = DEBUG  # Auto-refresh in development, cache in production
WHITENOISE_MAX_AGE = 31536000 if not DEBUG else 0  # Cache for 1 year in production
WHITENOISE_MIMETYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
}

# Media files
MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "media")

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# REST Framework settings
REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

# CORS settings
cors_origins = os.getenv(
    "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
)
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in cors_origins.split(",")
    if origin.strip() and origin.strip() != "*"
]

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_ALL_ORIGINS = (
    DEBUG or "*" in cors_origins
)  # Allow all in debug mode or if * is specified

# Application-specific settings
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DEFAULT_WEIGHT = os.getenv("DEFAULT_WEIGHT", "cricket.pt")
DEFAULT_CLASSIFICATION_WEIGHT = os.getenv(
    "DEFAULT_CLASSIFICATION_WEIGHT", "cricket_classify.pt"
)
DEFAULT_FPS = 1
DEFAULT_CONFIDENCE = 0.5

# OCR pipeline: GLM-OCR (remote Ollama) extracts text → DeepSeek text API
# formats it into JSON using the sport prompt. The only supported value is
# "local"; the env var stays so future providers can be slotted in.
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "local").strip().lower() or "local"

# GLM-OCR via remote Ollama (the GLM_OCR container).
LOCAL_OCR_OLLAMA_HOST = os.getenv("LOCAL_OCR_OLLAMA_HOST", "http://localhost:11434")
LOCAL_OCR_OLLAMA_MODEL = os.getenv("LOCAL_OCR_OLLAMA_MODEL", "glm-ocr")
LOCAL_OCR_OLLAMA_TIMEOUT_SECONDS = float(
    os.getenv("LOCAL_OCR_OLLAMA_TIMEOUT_SECONDS", "180")
)
LOCAL_OCR_MAX_NEW_TOKENS = int(os.getenv("LOCAL_OCR_MAX_NEW_TOKENS", "2048"))
LOCAL_OCR_EXTRACT_PROMPT = os.getenv(
    "LOCAL_OCR_EXTRACT_PROMPT",
    "Extract all visible text from this image, preserving the original "
    "layout, line breaks, and reading order. Return only the extracted text.",
)

# Stage-2 text formatter provider — "deepseek" (default) or "gemini".
# Both turn raw GLM-OCR text into the prompted JSON shape; gemini is faster.
TEXT_FORMATTER_PROVIDER = os.getenv("TEXT_FORMATTER_PROVIDER", "deepseek").strip().lower() or "deepseek"

# DeepSeek text API (api.deepseek.com) — formats extracted OCR text into JSON.
DEEPSEEK_TEXT_API_KEY = os.getenv("DEEPSEEK_TEXT_API_KEY", "")
DEEPSEEK_TEXT_BASE_URL = os.getenv(
    "DEEPSEEK_TEXT_BASE_URL", "https://api.deepseek.com/v1"
)
DEEPSEEK_TEXT_MODEL = os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-chat")
DEEPSEEK_TEXT_TIMEOUT_SECONDS = float(os.getenv("DEEPSEEK_TEXT_TIMEOUT_SECONDS", "60"))
DEEPSEEK_TEXT_MAX_TOKENS = int(os.getenv("DEEPSEEK_TEXT_MAX_TOKENS", "2048"))
DEEPSEEK_TEXT_TEMPERATURE = float(os.getenv("DEEPSEEK_TEXT_TEMPERATURE", "0.0"))

# Gemini text API (generativelanguage.googleapis.com) — alternative stage-2
# formatter. Default model is 2.5 Flash-Lite for lowest latency.
GEMINI_TEXT_API_KEY = os.getenv("GEMINI_TEXT_API_KEY", "")
GEMINI_TEXT_BASE_URL = os.getenv(
    "GEMINI_TEXT_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite")
GEMINI_TEXT_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TEXT_TIMEOUT_SECONDS", "60"))
GEMINI_TEXT_MAX_TOKENS = int(os.getenv("GEMINI_TEXT_MAX_TOKENS", "2048"))
GEMINI_TEXT_TEMPERATURE = float(os.getenv("GEMINI_TEXT_TEMPERATURE", "0.0"))

# Redis / django-rq — used to run OCR off the detection sync path so the
# frontend gets detections back instantly and OCR results stream in once
# the worker picks them up.
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
RQ_QUEUES = {
    "ocr": {
        "URL": REDIS_URL,
        "DEFAULT_TIMEOUT": int(os.getenv("RQ_OCR_TIMEOUT", "180")),
        "RESULT_TTL": int(os.getenv("RQ_RESULT_TTL", "3600")),
        "FAILURE_TTL": int(os.getenv("RQ_FAILURE_TTL", "3600")),
    },
}

# Spectacular (API Documentation) settings
SPECTACULAR_SETTINGS = {
    "TITLE": "Logo Detection API",
    "DESCRIPTION": "API for detecting logos in images and videos using YOLO",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
    "SCHEMA_PATH_PREFIX": "/api/",
}
