# Application image — thin layer on top of the heavy base.
#
# The base image (root200/detector-base:vX) holds CUDA + the entire Python
# venv + the Ollama binary + YOLO weights. Those rarely change, so we pin
# a tagged base and only rebuild this image when code/config changes. Net
# result: every iteration push is ~5 MB instead of 11 GB.
#
# Bump the FROM tag when Dockerfile.base is rebuilt for new deps or weights.

FROM root200/detector-base:v1

WORKDIR /app

# App code (changes most often; keep these COPYs late so the BUILD_TIME
# stamp below auto-busts whenever any of them changes).
COPY apps/ ./apps/
COPY config/ ./config/
COPY detector/ ./detector/
COPY main.py manage.py README.md ./

# Supervisor + entrypoint.
COPY docker/supervisord.conf /etc/supervisor/conf.d/detector.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Build identity, exposed at GET /. Stamped AFTER the COPY layers so any
# code/config change busts the cache and the timestamp refreshes; if
# nothing changed the cache hits and the stamp stays the same (correct:
# no deploy happened). Pass --build-arg GIT_SHA=$(git rev-parse --short HEAD)
# from your build command to embed the commit too.
ARG GIT_SHA=unknown
ENV BUILD_GIT_SHA=$GIT_SHA
RUN date -u +%Y-%m-%dT%H:%M:%SZ > /app/BUILD_TIME

# OCR / queue tuning. The timeout budget MUST stay ordered so a stalled job
# fails CLEANLY (and retries) instead of being killed mid-flight:
#   slot wait (150) < ollama x retries (120*3=360) + slot + format < RQ job (600)
# OLLAMA_NUM_PARALLEL / MAX_CONCURRENT / RQ_WORKERS are the throughput knobs —
# raise them only if the GPU has VRAM headroom (the vision model is large).
ENV DJANGO_SETTINGS_MODULE=detector.settings \
    ALLOWED_HOSTS=* \
    DEBUG=False \
    OCR_PROVIDER=local \
    LOCAL_OCR_OLLAMA_HOST=http://127.0.0.1:11434 \
    LOCAL_OCR_OLLAMA_MODEL=glm-ocr \
    LOCAL_OCR_OLLAMA_TIMEOUT_SECONDS=120 \
    REDIS_URL=redis://127.0.0.1:6379/0 \
    PORT=8000 \
    WEB_CONCURRENCY=4 \
    GUNICORN_TIMEOUT=3600 \
    RQ_WORKERS=12 \
    RQ_OCR_TIMEOUT=600 \
    RQ_RESULT_TTL=7200 \
    RQ_FAILURE_TTL=3600 \
    RQ_OCR_JOB_RETRIES=3 \
    OLLAMA_NUM_PARALLEL=2 \
    LOCAL_OCR_OLLAMA_MAX_CONCURRENT=3 \
    LOCAL_OCR_OLLAMA_SLOT_TIMEOUT_SECONDS=150 \
    LOCAL_OCR_OLLAMA_RETRIES=3 \
    LOCAL_OCR_OLLAMA_RETRY_BASE_DELAY=1.0

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
