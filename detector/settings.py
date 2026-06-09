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

# Database — external Postgres OR MySQL/MariaDB in production (Coolify) via
# DATABASE_URL, falling back to sqlite for local dev when DATABASE_URL is unset.
# The engine is chosen from the URL scheme:
#   postgres:// | postgresql://   -> Postgres   (needs psycopg2, in the image)
#   mysql://    | mariadb://      -> MySQL       (needs PyMySQL, in the image)
# sqlite needs nothing, so `manage.py check` works locally without a driver.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL:
    from urllib.parse import urlparse, unquote

    _db = urlparse(DATABASE_URL)
    # Allow an explicit driver suffix too, e.g. mysql+pymysql://...
    _scheme = _db.scheme.split("+", 1)[0].lower()
    _ENGINES = {
        "postgres": "django.db.backends.postgresql",
        "postgresql": "django.db.backends.postgresql",
        "mysql": "django.db.backends.mysql",
        "mariadb": "django.db.backends.mysql",
    }
    _engine = _ENGINES.get(_scheme, "django.db.backends.postgresql")

    _db_config = {
        "ENGINE": _engine,
        "NAME": unquote(_db.path.lstrip("/")),
        "USER": unquote(_db.username) if _db.username else "",
        "PASSWORD": unquote(_db.password) if _db.password else "",
        "HOST": _db.hostname or "",
        "PORT": str(_db.port or ""),
        "CONN_MAX_AGE": int(os.getenv("DB_CONN_MAX_AGE", "60")),
    }

    if _engine == "django.db.backends.mysql":
        # PyMySQL is a pure-python MySQL driver (no system libs to build).
        # Register it as MySQLdb and spoof a version new enough for Django's
        # mysqlclient >= 1.4.3 check.
        import pymysql

        pymysql.version_info = (1, 4, 6, "final", 0)
        pymysql.install_as_MySQLdb()
        _db_config["OPTIONS"] = {"charset": "utf8mb4"}

    DATABASES = {"default": _db_config}
else:
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

# DigitalOcean Spaces (S3-compatible) — detection uploads each annotated frame
# here and stores its public URL on the Frame row; OCR sends that URL to the
# GLM box. When SPACES is unconfigured, detection falls back to local /static
# frame files (dev only).
SPACES_ENDPOINT = os.getenv("DO_ENDPOINT", "").strip()
SPACES_REGION = os.getenv("DO_DEFAULT_REGION", "").strip()
SPACES_BUCKET = os.getenv("DO_BUCKET", "").strip()
SPACES_ACCESS_KEY_ID = os.getenv("DO_ACCESS_KEY_ID", "").strip()
SPACES_SECRET_ACCESS_KEY = os.getenv("DO_SECRET_ACCESS_KEY", "").strip()
# Public base for object URLs. Defaults to the virtual-hosted Spaces domain
# derived from the endpoint + bucket; override to use a CDN domain.
SPACES_PUBLIC_BASE_URL = os.getenv("DO_PUBLIC_BASE_URL", "").strip()
# Key prefix under which frames are stored in the bucket.
SPACES_FRAMES_PREFIX = os.getenv("DO_FRAMES_PREFIX", "frames").strip().strip("/")

# OCR pipeline (runs out-of-band from detection, as an RQ job): the EXTERNAL
# GLM OCR box extracts text → optional DeepSeek/Gemini formatter shapes it into
# JSON. OCR_PROVIDER stays so future providers can be slotted in.
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "local").strip().lower() or "local"

# External GLM OCR service (its own GPU). FastAPI /ocr endpoint that fetches
# the image URL itself. Retry knobs are read directly from the env by the
# client (GLM_OCR_RETRIES / GLM_OCR_RETRY_BASE_DELAY).
GLM_OCR_HOST = os.getenv("GLM_OCR_HOST", "http://localhost:8080")
GLM_OCR_MODEL = os.getenv("GLM_OCR_MODEL", "glm-ocr")
GLM_OCR_TIMEOUT_SECONDS = float(os.getenv("GLM_OCR_TIMEOUT_SECONDS", "120"))
GLM_OCR_MAX_NEW_TOKENS = int(os.getenv("GLM_OCR_MAX_NEW_TOKENS", "2048"))
GLM_OCR_EXTRACT_PROMPT = os.getenv(
    "GLM_OCR_EXTRACT_PROMPT",
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

# Redis / django-rq — EXTERNAL Redis (Coolify) via REDIS_URL. Runs OCR as
# background jobs that call the external GLM OCR API; the durable copy of each
# result lives on Frame.ocr_summary, the Redis result is the realtime channel.
# DEFAULT_TIMEOUT must exceed one GLM call (timeout x retries) plus the
# optional formatter call so a slow job isn't killed mid-flight.
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

# Default RQ worker class: SimpleWorker (NO fork). The default rq Worker forks a
# "work-horse" per job; forking AFTER torch/YOLO has initialized CUDA or Apple
# MPS aborts that child with SIGABRT (signal 6) — "Work-horse terminated
# unexpectedly", which silently kills detection jobs. SimpleWorker runs the job
# in the worker process itself (model stays resident, nothing to fork-crash).
# Applies to `rqworker` and `rqworker-pool` unless --worker-class is passed.
RQ = {"WORKER_CLASS": os.getenv("RQ_WORKER_CLASS", "rq.SimpleWorker")}

RQ_QUEUES = {
    "ocr": {
        "URL": REDIS_URL,
        "DEFAULT_TIMEOUT": int(os.getenv("RQ_OCR_TIMEOUT", "600")),
        "RESULT_TTL": int(os.getenv("RQ_RESULT_TTL", "7200")),
        "FAILURE_TTL": int(os.getenv("RQ_FAILURE_TTL", "3600")),
    },
    # Long-running video detection jobs run here, OFF the request/SSE path, so
    # a closed browser / dropped connection / redeploy can't stop them. Kept on
    # a SEPARATE queue from `ocr` so a multi-hour video never blocks short OCR
    # calls. DEFAULT_TIMEOUT is a hard ceiling on one job; the worker also
    # heartbeats, and the reaper resumes anything that dies before finishing.
    "detection": {
        "URL": REDIS_URL,
        "DEFAULT_TIMEOUT": int(os.getenv("RQ_DETECTION_TIMEOUT", str(24 * 3600))),
        "RESULT_TTL": int(os.getenv("RQ_RESULT_TTL", "7200")),
        "FAILURE_TTL": int(os.getenv("RQ_FAILURE_TTL", "3600")),
    },
}

# How long (seconds) a PROCESSING session may go without a heartbeat before the
# reaper treats its worker as dead and re-enqueues it from processed_frames.
DETECTION_HEARTBEAT_STALE_SECONDS = int(
    os.getenv("DETECTION_HEARTBEAT_STALE_SECONDS", "120")
)
# Bound on automatic resume attempts so a genuinely poisonous video can't loop
# forever; past this it stays FAILED for a human to look at.
DETECTION_MAX_ATTEMPTS = int(os.getenv("DETECTION_MAX_ATTEMPTS", "5"))

# Logging — explicit config so the detection path emits real, timestamped logs
# (with tracebacks via logger.exception) to stdout, which supervisord/Docker
# capture. Replaces the bare print()s that made silent stops un-diagnosable.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}] {levelname} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {"handlers": ["console"], "level": "WARNING"},
    "loggers": {
        "apps": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
        "django.request": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
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
