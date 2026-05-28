#!/usr/bin/env bash
# Container PID 1 (under tini): prep Django state + ensure the GLM-OCR
# model is present, then hand control to supervisord which runs redis,
# ollama, the rq worker, and gunicorn.
set -euo pipefail

cd /app

echo "[entrypoint] migrating database..."
python manage.py migrate --noinput

echo "[entrypoint] collecting static files..."
python manage.py collectstatic --noinput >/dev/null 2>&1 || true

# Lazy-pull the glm-ocr model on first start. Skipped after the first
# successful pull because the blobs persist in $OLLAMA_MODELS (mount a
# volume there to survive container restarts on vast.ai).
echo "[entrypoint] ensuring glm-ocr model is present..."
OLLAMA_HOST=127.0.0.1:11434 /usr/bin/ollama serve >/tmp/ollama-prep.log 2>&1 &
ollama_pid=$!
ready=0
for i in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        ready=1; break
    fi
    sleep 1
done
if [ "$ready" != "1" ]; then
    echo "[entrypoint] WARNING: ollama did not become ready in 30s; continuing"
    echo "[entrypoint] ollama prep log:"
    tail -n 50 /tmp/ollama-prep.log || true
elif OLLAMA_HOST=127.0.0.1:11434 /usr/bin/ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -q "^glm-ocr"; then
    echo "[entrypoint] glm-ocr already on disk, skipping pull"
else
    echo "[entrypoint] pulling glm-ocr (~10GB, one-time, can take 5-15min)..."
    OLLAMA_HOST=127.0.0.1:11434 /usr/bin/ollama pull glm-ocr
    echo "[entrypoint] glm-ocr pull complete"
fi
kill "$ollama_pid" 2>/dev/null || true
wait "$ollama_pid" 2>/dev/null || true

echo "[entrypoint] starting supervisord (redis, ollama, rqworker, gunicorn)..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/detector.conf
