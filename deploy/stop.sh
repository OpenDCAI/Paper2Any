#!/bin/bash
# FastAPI 应用停止脚本

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PID_FILES=(
  "logs/uvicorn.pid"
  "logs/gunicorn.pid"
  "logs/uvicorn-9012.pid"
  "logs/backend-9012.pid"
)

cleanup_pid_files() {
  for pidfile in "${PID_FILES[@]}"; do
    if [ -f "$pidfile" ]; then
      pid="$(cat "$pidfile" 2>/dev/null || true)"
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pidfile"
      fi
    fi
  done
}

stop_pid() {
  local pid="$1"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi

  echo "Stopping FastAPI app (PID: $pid)..."
  kill -TERM "$pid" 2>/dev/null || true

  for _ in {1..10}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "FastAPI app stopped successfully"
      return 0
    fi
    sleep 1
  done

  echo "Force killing FastAPI app (PID: $pid)..."
  kill -KILL "$pid" 2>/dev/null || true
  sleep 1
  ! kill -0 "$pid" 2>/dev/null
}

stop_from_pidfile() {
  local pidfile="$1"
  if [ ! -f "$pidfile" ]; then
    return 1
  fi

  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if stop_pid "$pid"; then
    rm -f "$pidfile"
    return 0
  fi

  rm -f "$pidfile"
  return 1
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

cleanup_pid_files

for pidfile in "${PID_FILES[@]}"; do
  if stop_from_pidfile "$pidfile"; then
    cleanup_pid_files
    exit 0
  fi
done

echo "PID file not found or stale. Trying port 9012..."
port_pids="$(find_port_listener_pids 9012 || true)"
if [ -n "$port_pids" ]; then
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"fastapi_app.main:app"* ]] || [[ "$cmdline" == *"$PROJECT_ROOT"* ]]; then
      if stop_pid "$pid"; then
        cleanup_pid_files
        exit 0
      fi
    fi
  done <<< "$port_pids"
fi

echo "Trying process-name fallback..."
pkill -f "uvicorn fastapi_app.main:app" 2>/dev/null || \
pkill -f "uvicorn .*fastapi_app.main:app" 2>/dev/null || \
pkill -f "gunicorn fastapi_app.main:app" 2>/dev/null || true

cleanup_pid_files
echo "No managed FastAPI backend process was found."
