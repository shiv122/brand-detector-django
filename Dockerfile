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

# nginx — fronts Gunicorn and serves /static/ (frames, videos, csv) off disk
# so image bytes never occupy a Python worker. Kept in this thin layer (not
# the base) for now; it's a stable ~6 MB layer that ships once and caches. On
# the next base rebuild, move this apt line into Dockerfile.base.
RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default
COPY docker/nginx.conf /etc/nginx/nginx.conf.template

# Runtime deps added to the base venv here (so we don't rebuild the 11 GB base
# just to add them): psycopg2 for Postgres (DATABASE_URL), boto3 for uploading
# frames to DigitalOcean Spaces. Fold into Dockerfile.base on the next rebuild.
RUN uv pip install --python /app/.venv/bin/python --no-cache \
        psycopg2-binary==2.9.9 boto3==1.34.162

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

# Runtime defaults. DATABASE_URL, REDIS_URL, GLM_OCR_HOST and the formatter
# API keys are supplied by the deploy environment (Coolify) — the values
# below are safe local-dev fallbacks, NOT where the real services live.
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
