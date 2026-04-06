#!/bin/bash

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

source "$PROJECT_ROOT/deploy/app_config.sh"

LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/frontend.pid"

mkdir -p "$LOG_DIR"

resolve_npm() {
  if [ -n "${FRONTEND_NPM:-}" ] && command -v "$FRONTEND_NPM" >/dev/null 2>&1; then
    return 0
  fi

  FRONTEND_NPM="$(command -v npm || true)"
  export FRONTEND_NPM
  if [ -z "$FRONTEND_NPM" ]; then
    echo "No npm executable found."
    return 1
  fi
}

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

wait_for_port_listener() {
  local port="$1"
  local attempts="${2:-20}"
  for _ in $(seq 1 "$attempts"); do
    if [ -n "$(find_port_listener_pids "$port" || true)" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

bash "$PROJECT_ROOT/deploy/stop_frontend.sh" >/dev/null 2>&1 || true
sleep 1

resolve_npm || exit 1

if [ ! -d "$FRONTEND_DIR" ]; then
  echo "Frontend directory not found: $FRONTEND_DIR"
  exit 1
fi

cd "$FRONTEND_DIR" || exit 1

start_cmd=("$FRONTEND_NPM" run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" --strictPort)

if command -v setsid >/dev/null 2>&1; then
  nohup setsid env \
    VITE_API_BASE_URL="${VITE_API_BASE_URL:-}" \
    "${start_cmd[@]}" >> "$LOG_DIR/frontend.log" 2>&1 < /dev/null &
else
  nohup env \
    VITE_API_BASE_URL="${VITE_API_BASE_URL:-}" \
    "${start_cmd[@]}" >> "$LOG_DIR/frontend.log" 2>&1 < /dev/null &
fi

echo $! > "$PID_FILE"
disown 2>/dev/null || true

if wait_for_port_listener "$FRONTEND_PORT" 20; then
  echo "Frontend started with PID: $(cat "$PID_FILE")"
  exit 0
fi

echo "Frontend failed to start. Check $LOG_DIR/frontend.log"
exit 1
