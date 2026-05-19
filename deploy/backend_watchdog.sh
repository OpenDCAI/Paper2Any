#!/bin/bash
# FastAPI backend watchdog for port 8000.

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

source "$PROJECT_ROOT/deploy/app_config.sh"

LOG_DIR="$PROJECT_ROOT/logs"
UVICORN_PID_FILE="$LOG_DIR/uvicorn.pid"
WATCHDOG_PID_FILE="$LOG_DIR/backend_watchdog.pid"
WATCHDOG_LOG_FILE="$LOG_DIR/backend_watchdog.log"
WATCHDOG_LOCK_DIR="$LOG_DIR/backend_watchdog.lock"

WATCHDOG_INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-15}"
WATCHDOG_FAIL_THRESHOLD="${WATCHDOG_FAIL_THRESHOLD:-2}"
WATCHDOG_RESTART_COOLDOWN_SECONDS="${WATCHDOG_RESTART_COOLDOWN_SECONDS:-30}"
WATCHDOG_HEALTH_TIMEOUT_SECONDS="${WATCHDOG_HEALTH_TIMEOUT_SECONDS:-5}"
WATCHDOG_START_WAIT_SECONDS="${WATCHDOG_START_WAIT_SECONDS:-25}"

mkdir -p "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '%s | %s\n' "$(timestamp)" "$*"
}

health_url() {
  printf 'http://127.0.0.1:%s/health' "$APP_PORT"
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

backend_port_listening() {
  [[ -n "$(find_port_listener_pids "$APP_PORT" || true)" ]]
}

backend_health_ok() {
  local response
  response="$(
    curl -fsS \
      --max-time "$WATCHDOG_HEALTH_TIMEOUT_SECONDS" \
      "$(health_url)" \
      2>/dev/null || true
  )"
  [[ "$response" == *'"status":"ok"'* ]]
}

current_watchdog_pid() {
  if [[ -f "$WATCHDOG_PID_FILE" ]]; then
    cat "$WATCHDOG_PID_FILE" 2>/dev/null || true
  fi
}

backend_pid_from_pidfile() {
  if [[ -f "$UVICORN_PID_FILE" ]]; then
    cat "$UVICORN_PID_FILE" 2>/dev/null || true
  fi
}

backend_process_running() {
  local pid
  pid="$(backend_pid_from_pidfile)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  pgrep -f "uvicorn fastapi_app.main:app" >/dev/null 2>&1
}

watchdog_running() {
  local pid
  pid="$(current_watchdog_pid)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

write_pid() {
  echo "$1" > "$WATCHDOG_PID_FILE"
}

cleanup_pid_file() {
  if ! watchdog_running; then
    rm -f "$WATCHDOG_PID_FILE"
  fi
}

acquire_lock() {
  mkdir "$WATCHDOG_LOCK_DIR" 2>/dev/null
}

release_lock() {
  rmdir "$WATCHDOG_LOCK_DIR" 2>/dev/null || true
}

wait_for_backend() {
  local attempt
  for attempt in $(seq 1 "$WATCHDOG_START_WAIT_SECONDS"); do
    if backend_port_listening; then
      return 0
    fi
    sleep 1
  done
  return 1
}

restart_backend() {
  if ! acquire_lock; then
    log "[watchdog] another restart is already in progress"
    return 1
  fi

  log "[watchdog] backend unhealthy, running deploy/start.sh"
  if bash "$PROJECT_ROOT/deploy/start.sh" >> "$WATCHDOG_LOG_FILE" 2>&1; then
    if wait_for_backend; then
      log "[watchdog] backend recovered successfully"
      release_lock
      return 0
    fi
    log "[watchdog] deploy/start.sh returned success, but port $APP_PORT did not recover in time"
  else
    log "[watchdog] deploy/start.sh failed"
  fi

  release_lock
  return 1
}

run_once() {
  if backend_port_listening; then
    log "[watchdog] backend port $APP_PORT is listening"
    return 0
  fi

  if backend_process_running; then
    log "[watchdog] port $APP_PORT is down, but backend process is still alive"
    return 0
  fi

  restart_backend
}

run_loop() {
  local consecutive_failures=0

  trap 'rm -f "$WATCHDOG_PID_FILE"; release_lock; exit 0' INT TERM EXIT
  write_pid "$$"
  log "[watchdog] started, interval=${WATCHDOG_INTERVAL_SECONDS}s fail_threshold=${WATCHDOG_FAIL_THRESHOLD}"

  while true; do
    if backend_port_listening; then
      consecutive_failures=0
    else
      consecutive_failures=$((consecutive_failures + 1))
      if backend_process_running; then
        log "[watchdog] port $APP_PORT is down, but backend process is still alive (${consecutive_failures}/${WATCHDOG_FAIL_THRESHOLD})"
      else
        log "[watchdog] port $APP_PORT is down (${consecutive_failures}/${WATCHDOG_FAIL_THRESHOLD})"
      fi
      if (( consecutive_failures >= WATCHDOG_FAIL_THRESHOLD )); then
        restart_backend || true
        consecutive_failures=0
        sleep "$WATCHDOG_RESTART_COOLDOWN_SECONDS"
        continue
      fi
    fi

    sleep "$WATCHDOG_INTERVAL_SECONDS"
  done
}

start_watchdog() {
  cleanup_pid_file
  if watchdog_running; then
    log "[watchdog] already running with PID $(current_watchdog_pid)"
    exit 0
  fi

  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$0" run >> "$WATCHDOG_LOG_FILE" 2>&1 < /dev/null &
  else
    nohup bash "$0" run >> "$WATCHDOG_LOG_FILE" 2>&1 < /dev/null &
  fi
  sleep 1

  if watchdog_running; then
    log "[watchdog] started with PID $(current_watchdog_pid)"
    exit 0
  fi

  log "[watchdog] failed to start"
  exit 1
}

stop_watchdog() {
  local pid
  local waited=0
  pid="$(current_watchdog_pid)"
  if [[ -z "$pid" ]]; then
    log "[watchdog] not running"
    rm -f "$WATCHDOG_PID_FILE"
    exit 0
  fi

  if kill "$pid" 2>/dev/null; then
    while kill -0 "$pid" 2>/dev/null && (( waited < 5 )); do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
      sleep 1
    fi
    cleanup_pid_file
    release_lock
    if kill -0 "$pid" 2>/dev/null; then
      log "[watchdog] failed to stop PID $pid"
      exit 1
    fi
    log "[watchdog] stopped"
    exit 0
  fi

  log "[watchdog] failed to stop PID $pid"
  exit 1
}

status_watchdog() {
  cleanup_pid_file
  if watchdog_running; then
    log "[watchdog] running with PID $(current_watchdog_pid)"
  else
    log "[watchdog] not running"
  fi

  if backend_port_listening; then
    log "[watchdog] backend port $APP_PORT is listening"
  else
    log "[watchdog] backend port $APP_PORT is NOT listening"
  fi

  if backend_health_ok; then
    log "[watchdog] backend health OK on $(health_url)"
  elif backend_process_running; then
    log "[watchdog] backend health FAILED, but uvicorn process is still alive"
  else
    log "[watchdog] backend health FAILED on $(health_url)"
  fi

  log "[watchdog] log file: $WATCHDOG_LOG_FILE"
}

show_logs() {
  touch "$WATCHDOG_LOG_FILE"
  tail -n 80 "$WATCHDOG_LOG_FILE"
}

usage() {
  cat <<'EOF'
Usage:
  bash deploy/backend_watchdog.sh start
  bash deploy/backend_watchdog.sh stop
  bash deploy/backend_watchdog.sh status
  bash deploy/backend_watchdog.sh logs
  bash deploy/backend_watchdog.sh run
  bash deploy/backend_watchdog.sh run-once

Optional env vars:
  WATCHDOG_INTERVAL_SECONDS
  WATCHDOG_FAIL_THRESHOLD
  WATCHDOG_RESTART_COOLDOWN_SECONDS
  WATCHDOG_HEALTH_TIMEOUT_SECONDS
  WATCHDOG_START_WAIT_SECONDS
EOF
}

case "${1:-status}" in
  start)
    start_watchdog
    ;;
  stop)
    stop_watchdog
    ;;
  status)
    status_watchdog
    ;;
  logs)
    show_logs
    ;;
  run)
    run_loop
    ;;
  run-once)
    run_once
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${1:-}" >&2
    usage
    exit 1
    ;;
esac
