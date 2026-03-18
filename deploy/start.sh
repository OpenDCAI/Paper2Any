#!/bin/bash
# FastAPI 应用启动脚本

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/uvicorn.pid"
APP_PORT=8000
APP_WORKERS=16

mkdir -p "$LOG_DIR"

find_port_listener_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u
    return 0
  fi

  ss -ltnp 2>/dev/null \
    | awk -v port=":$port" '$4 ~ port { print $NF }' \
    | grep -oE 'pid=[0-9]+' \
    | cut -d= -f2 \
    | sort -u
}

# 启动前先做一次清理，避免残留 worker 占住端口。
"$PROJECT_ROOT/deploy/stop.sh" >/dev/null 2>&1 || true
sleep 1

existing_pids="$(find_port_listener_pids "$APP_PORT" || true)"
if [ -n "$existing_pids" ]; then
  echo "Port $APP_PORT is still occupied. Refusing to start."
  echo "$existing_pids" | sed 's/^/LISTEN_PID: /'
  exit 1
fi

nohup uvicorn fastapi_app.main:app --workers "$APP_WORKERS" --port "$APP_PORT" \
  --log-level info \
  >> "$LOG_DIR/app.log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
disown 2>/dev/null || true
sleep 2

if [ -n "$(find_port_listener_pids "$APP_PORT" || true)" ]; then
  echo "FastAPI app started with PID: $(cat "$PID_FILE")"
  exit 0
fi

echo "FastAPI app failed to start. Check $LOG_DIR/app.log"
exit 1
