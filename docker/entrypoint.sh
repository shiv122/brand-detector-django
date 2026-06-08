#!/usr/bin/env bash
# Container PID 1 (under tini): prep Django state, then hand control to
# supervisord which runs the rq worker, gunicorn, and nginx. Redis + Postgres
# are external (Coolify); GLM OCR is an external HTTP API.
set -euo pipefail

cd /app

echo "[entrypoint] migrating database..."
python manage.py migrate --noinput

echo "[entrypoint] collecting static files..."
python manage.py collectstatic --noinput >/dev/null 2>&1 || true

# Render nginx config — substitute the public listen port ($PORT). nginx
# doesn't expand env vars itself, so we do it here. Then fail fast if the
# config is invalid rather than crash-looping under supervisord.
echo "[entrypoint] rendering nginx config (listen ${PORT})..."
sed "s/__PORT__/${PORT}/g" /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf
nginx -t -c /etc/nginx/nginx.conf

echo "[entrypoint] starting supervisord (rqworker, gunicorn, nginx)..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/detector.conf
