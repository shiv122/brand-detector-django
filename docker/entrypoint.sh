#!/usr/bin/env bash
# Container PID 1 (under tini): prep Django state, then hand control to
# supervisord which runs redis, ollama, the rq worker, and gunicorn.
set -euo pipefail

cd /app

echo "[entrypoint] migrating database..."
python manage.py migrate --noinput

echo "[entrypoint] collecting static files..."
python manage.py collectstatic --noinput >/dev/null 2>&1 || true

echo "[entrypoint] starting supervisord (redis, ollama, rqworker, gunicorn)..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/detector.conf
