# Django backend with GPU YOLO. CUDA+cuDNN base provides system CUDA libs;
# after uv sync we strip the Nvidia pip wheels torch bundles but YOLO
# inference never uses (multi-GPU, FFT, sparse, profiling, triton/compile).
# That cuts ~5–6 GB off the image while keeping GPU inference working.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin

# libgl/libglib for opencv-python, libgomp for torch/ultralytics, redis +
# supervisor + tini for the runtime, curl + ca-certs to install uv, zstd to
# unpack the Ollama release tarball.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini supervisor redis-server zstd \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv (manages the venv).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv \
    && ln -sf /root/.local/bin/uvx /usr/local/bin/uvx

# Ollama (the engine that serves the glm-ocr vision model). Installed from
# the GitHub release tarball — avoids the systemd unit the upstream
# installer would otherwise wire up.
ARG OLLAMA_VERSION=v0.24.0
RUN curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
        -o /tmp/ollama.tar.zst \
    && zstd -d /tmp/ollama.tar.zst -o /tmp/ollama.tar \
    && tar -C /usr -xf /tmp/ollama.tar \
    && rm /tmp/ollama.tar /tmp/ollama.tar.zst \
    && ollama --version

# Bake the glm-ocr model into the image so cold starts on vast.ai don't
# have to pull ~10GB from the Ollama registry. Mirrors the GLM_OCR/Dockerfile
# build step: spin Ollama up on loopback, pull the model, stop it. The
# downloaded blobs live in /root/.ollama and the supervisord ollama program
# picks them up at runtime.
ENV OLLAMA_MODELS=/root/.ollama/models \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_NUM_PARALLEL=1
RUN set -eux; \
    OLLAMA_HOST=127.0.0.1:11434 ollama serve > /tmp/ollama-build.log 2>&1 & \
    pid=$!; \
    for i in $(seq 1 30); do \
        if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi; \
        sleep 1; \
    done; \
    OLLAMA_HOST=127.0.0.1:11434 ollama pull glm-ocr; \
    OLLAMA_HOST=127.0.0.1:11434 ollama list; \
    kill $pid; wait $pid 2>/dev/null || true

WORKDIR /app

# Python deps. The uv cache lives in a buildkit cache mount (not in the
# image layer) so peak disk during install stays low and wheels never
# bloat the final image.
#
# Strip only what torch genuinely never loads at `import torch` time.
# Across torch 2.4+ the CUDA wheels (cupti, nvtx, cudnn, cublas, cufft,
# curand, cusparse, cusparselt, cusolver, nccl, nvshmem) ARE all linked
# or dlopen'd from torch._C — stripping any of them breaks import with
# "libXXX.so.N: cannot open shared object file". triton is the lone
# pure-python compile-time dep that YOLO inference never touches.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --python 3.11 \
 && uv pip uninstall triton \
 && find /app/.venv -depth -type d -name '__pycache__' -exec rm -rf {} + \
 && find /app/.venv -depth -type d -name 'tests' -exec rm -rf {} + \
 && find /app/.venv -type f -name '*.pyc' -delete

# YOLO weights baked in (detection works offline).
COPY weights/ ./weights/

# App code (changes most often; keep last).
COPY apps/ ./apps/
COPY config/ ./config/
COPY detector/ ./detector/
COPY main.py manage.py README.md ./

# Supervisor + entrypoint.
COPY docker/supervisord.conf /etc/supervisor/conf.d/detector.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Build identity, exposed at GET /. Stamped AFTER the COPY layers so any
# code/config change busts the cache and the timestamp refreshes; if nothing
# changed the cache hits and the stamp stays the same (correct: no deploy
# happened). Pass --build-arg GIT_SHA=$(git rev-parse --short HEAD) from
# your build command to embed the commit too.
ARG GIT_SHA=unknown
ENV BUILD_GIT_SHA=$GIT_SHA
RUN date -u +%Y-%m-%dT%H:%M:%SZ > /app/BUILD_TIME

ENV DJANGO_SETTINGS_MODULE=detector.settings \
    ALLOWED_HOSTS=* \
    DEBUG=False \
    OCR_PROVIDER=local \
    LOCAL_OCR_OLLAMA_HOST=http://127.0.0.1:11434 \
    LOCAL_OCR_OLLAMA_MODEL=glm-ocr \
    LOCAL_OCR_OLLAMA_TIMEOUT_SECONDS=180 \
    REDIS_URL=redis://127.0.0.1:6379/0 \
    PORT=8000 \
    WEB_CONCURRENCY=2 \
    RQ_WORKERS=12 \
    LOCAL_OCR_OLLAMA_MAX_CONCURRENT=2 \
    LOCAL_OCR_OLLAMA_SLOT_TIMEOUT_SECONDS=300 \
    LOCAL_OCR_OLLAMA_RETRIES=3 \
    LOCAL_OCR_OLLAMA_RETRY_BASE_DELAY=1.0

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
