#!/bin/bash

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

source "$PROJECT_ROOT/deploy/app_config.sh"

PID_FILE="$PROJECT_ROOT/logs/frontend.pid"

find_port_listener_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u
    return 0
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null \
      | awk -v port=":$port" '$4 ~ port { print $NF }' \
      | grep -oE 'pid=[0-9]+' \
      | cut -d= -f2 \
      | sort -u
    return 0
  fi

  if command -v netstat >/dev/null 2>&1; then
    netstat -ltnp 2>/dev/null \
      | awk -v port=":$port" '$4 ~ port { split($7, parts, "/"); if (parts[1] ~ /^[0-9]+$/) print parts[1] }' \
      | sort -u
    return 0
  fi

  return 1
}

wait_for_port_release() {
  local port="$1"
  for _ in {1..10}; do
    if [ -z "$(find_port_listener_pids "$port" || true)" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    wait_for_port_release "$FRONTEND_PORT" || true
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

port_pids="$(find_port_listener_pids "$FRONTEND_PORT" || true)"
if [ -n "$port_pids" ]; then
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done <<< "$port_pids"
  wait_for_port_release "$FRONTEND_PORT" || true
fi

port_pids="$(find_port_listener_pids "$FRONTEND_PORT" || true)"
if [ -n "$port_pids" ]; then
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    kill -KILL "$pid" 2>/dev/null || true
  done <<< "$port_pids"
fi

pkill -f "$FRONTEND_DIR/node_modules/.bin/vite --host $FRONTEND_HOST --port $FRONTEND_PORT" 2>/dev/null || true
pkill -f "frontend-workflow/node_modules/.bin/vite --host $FRONTEND_HOST --port $FRONTEND_PORT" 2>/dev/null || true

exit 0
