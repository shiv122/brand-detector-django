# Detector backend — single image: detection (YOLO/torch) + Django API + the
# RQ worker that runs OCR jobs by calling the EXTERNAL GLM OCR box over HTTP.
#
# Three processes run under supervisord (see docker/supervisord.conf):
#   - rqworker  — OCR jobs (HTTP calls to GLM_OCR_HOST; no GPU/Ollama here)
#   - gunicorn  — app/API + SSE, loopback only
#   - nginx     — the only public process; serves /static/ + proxies to gunicorn
#
# Redis (queue) and Postgres (DB) are EXTERNAL services reached via REDIS_URL /
# DATABASE_URL; GLM OCR is an external HTTP API (GLM_OCR_HOST). Nothing
# data-bearing is baked into this image.
#
# Build + push (Coolify builds this directly; this is for manual pushes):
#   docker buildx build --platform linux/amd64 \
#       -t root200/detector-backend:latest \
#       --build-arg GIT_SHA=$(git rev-parse --short HEAD) --push .

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin

# System deps:
#   libgl1/libglib2.0-0 — opencv-python   libgomp1 — torch/ultralytics
#   tini/supervisor      — process mgmt    curl/ca-certificates — install uv
#   nginx                — public reverse proxy + /static/ file serving
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini supervisor \
        libgl1 libglib2.0-0 libgomp1 nginx \
    && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default

# uv (manages the venv).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv \
    && ln -sf /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /app

# Python deps. The uv cache lives in a buildkit cache mount (not in the image
# layer) so peak disk during install stays low and wheels never bloat the
# image. psycopg2-binary (Postgres / DATABASE_URL) and boto3 (DigitalOcean
# Spaces uploads) are runtime-only extras added on top of the locked set.
#
# triton is stripped: it's the lone pure-python compile-time dep YOLO inference
# never touches. Every other CUDA wheel (cudnn, cublas, nccl, ...) IS linked or
# dlopen'd from torch._C, so stripping any of those breaks `import torch`.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --python 3.11 \
 && uv pip install --python /app/.venv/bin/python psycopg2-binary==2.9.9 boto3==1.34.162 \
 && uv pip uninstall triton \
 && find /app/.venv -depth -type d -name '__pycache__' -exec rm -rf {} + \
 && find /app/.venv -depth -type d -name 'tests' -exec rm -rf {} + \
 && find /app/.venv -type f -name '*.pyc' -delete

# YOLO weights baked in so detection works offline.
COPY weights/ ./weights/

# nginx config template — entrypoint renders the listen port ($PORT) into it.
COPY docker/nginx.conf /etc/nginx/nginx.conf.template

# App code (changes most often; keep these COPYs late so the BUILD_TIME stamp
# below auto-busts whenever any of them changes).
COPY apps/ ./apps/
COPY config/ ./config/
COPY detector/ ./detector/
COPY main.py manage.py README.md ./

# Supervisor + entrypoint.
COPY docker/supervisord.conf /etc/supervisor/conf.d/detector.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Build identity, exposed at GET /. Stamped AFTER the COPY layers so any
# code/config change busts the cache and refreshes the timestamp. Pass
# --build-arg GIT_SHA=$(git rev-parse --short HEAD) to embed the commit.
ARG GIT_SHA=unknown
ENV BUILD_GIT_SHA=$GIT_SHA
RUN date -u +%Y-%m-%dT%H:%M:%SZ > /app/BUILD_TIME

# Runtime defaults. DATABASE_URL, REDIS_URL, GLM_OCR_HOST, the formatter API
# keys and the DO_* Spaces creds are supplied by the deploy environment
# (Coolify) — the values below are safe local-dev fallbacks, NOT where the real
# services live.
#   - no DATABASE_URL  -> falls back to sqlite (local dev only)
#   - GLM_OCR_HOST     -> the external GLM OCR box (its own GPU)
#   - RQ_WORKERS       -> bounds concurrent OCR calls into the GLM box
ENV DJANGO_SETTINGS_MODULE=detector.settings \
    ALLOWED_HOSTS=* \
    DEBUG=False \
    OCR_PROVIDER=local \
    GLM_OCR_HOST=http://glm-ocr:8080 \
    GLM_OCR_MODEL=glm-ocr \
    GLM_OCR_TIMEOUT_SECONDS=120 \
    GLM_OCR_RETRIES=3 \
    GLM_OCR_RETRY_BASE_DELAY=1.0 \
    REDIS_URL=redis://redis:6379/0 \
    PORT=8000 \
    GUNICORN_PORT=8001 \
    WEB_CONCURRENCY=4 \
    GUNICORN_THREADS=8 \
    GUNICORN_TIMEOUT=3600 \
    RQ_WORKERS=4 \
    RQ_OCR_TIMEOUT=600 \
    RQ_RESULT_TTL=7200 \
    RQ_FAILURE_TTL=3600 \
    RQ_OCR_JOB_RETRIES=3

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
