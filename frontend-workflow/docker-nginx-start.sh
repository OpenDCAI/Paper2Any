#!/bin/sh
set -eu

LISTEN_PORT="${NGINX_LISTEN_PORT:-80}"
BACKEND_URL="${BACKEND_UPSTREAM_URL:-http://paper2any-backend:8000}"

sed \
  -e "s|__NGINX_LISTEN_PORT__|${LISTEN_PORT}|g" \
  -e "s|__BACKEND_UPSTREAM_URL__|${BACKEND_URL}|g" \
  /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf

exec nginx -g 'daemon off;'
