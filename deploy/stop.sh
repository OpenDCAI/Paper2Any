#!/bin/bash
# FastAPI 应用停止脚本

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

source "$PROJECT_ROOT/deploy/app_config.sh"

PID_FILES=(
  "logs/uvicorn.pid"
  "logs/gunicorn.pid"
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

get_pgid() {
  local pid="$1"
  ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]'
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

stop_pid_and_group() {
  local pid="$1"
  local signal="${2:-TERM}"
  local pgid

  if [ -z "$pid" ]; then
    return 1
  fi

  pgid="$(get_pgid "$pid")"
  if [ -n "$pgid" ]; then
    kill "-$signal" -- "-$pgid" 2>/dev/null || true
  fi

  kill "-$signal" "$pid" 2>/dev/null || true
  return 0
}

stop_from_pidfile() {
  local pidfile="$1"
  if [ ! -f "$pidfile" ]; then
    return 1
  fi

  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [ -n "$pid" ]; then
    echo "Stopping FastAPI app from PID file $pidfile (PID: $pid)..."
    stop_pid_and_group "$pid" TERM
    if ! wait_for_port_release "$APP_PORT"; then
      stop_pid_and_group "$pid" KILL
      wait_for_port_release "$APP_PORT" || true
    fi
  fi

  if [ -z "$(find_port_listener_pids "$APP_PORT" || true)" ]; then
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

echo "PID file not found or stale. Trying port $APP_PORT..."
port_pids="$(find_port_listener_pids "$APP_PORT" || true)"
if [ -n "$port_pids" ]; then
  declare -A seen_pgids=()
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    pgid="$(get_pgid "$pid")"
    if [ -n "$pgid" ] && [ -z "${seen_pgids[$pgid]+x}" ]; then
      seen_pgids["$pgid"]=1
      echo "Stopping listener process group PGID: $pgid (from PID: $pid)..."
      kill -TERM -- "-$pgid" 2>/dev/null || true
    fi
  done <<< "$port_pids"

  if ! wait_for_port_release "$APP_PORT"; then
    while IFS= read -r pid; do
      [ -n "$pid" ] || continue
      pgid="$(get_pgid "$pid")"
      if [ -n "$pgid" ]; then
        kill -KILL -- "-$pgid" 2>/dev/null || true
      fi
      kill -KILL "$pid" 2>/dev/null || true
    done <<< "$port_pids"
    wait_for_port_release "$APP_PORT" || true
  fi
fi

echo "Trying process-name fallback..."
pkill -f "uvicorn fastapi_app.main:app" 2>/dev/null || \
pkill -f "uvicorn .*fastapi_app.main:app" 2>/dev/null || \
pkill -f "gunicorn fastapi_app.main:app" 2>/dev/null || true

cleanup_pid_files

if [ -z "$(find_port_listener_pids "$APP_PORT" || true)" ]; then
  echo "FastAPI app stopped successfully"
  exit 0
fi

echo "FastAPI app is still listening on port $APP_PORT."
find_port_listener_pids "$APP_PORT" | sed 's/^/LISTEN_PID: /'
exit 1
