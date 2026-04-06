#!/bin/bash
# FastAPI 应用启动脚本

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

source "$PROJECT_ROOT/deploy/app_config.sh"

LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/uvicorn.pid"

mkdir -p "$LOG_DIR"

maybe_source_model_server_env() {
  local env_file="$PROJECT_ROOT/$MODEL_SERVER_ENV_FILE"
  if [ -f "$env_file" ]; then
    # shellcheck disable=SC1090
    source "$env_file"
  fi
}

validate_python() {
  local python_bin="$1"
  [ -n "$python_bin" ] || return 1
  [ -x "$python_bin" ] || return 1

  "$python_bin" - <<'PY' >/dev/null 2>&1
import cv2
import cairosvg
import fastapi
import moviepy.editor
import supabase
import torch
import uvicorn
PY
}

choose_first_existing() {
  local candidate
  for candidate in "$@"; do
    if [ -n "$candidate" ] && [ -e "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

prepare_runtime_env() {
  local legacy_sam3_checkpoint=""
  local legacy_sam3_bpe=""
  local legacy_sam3_home=""
  local legacy_rmbg=""
  local resolved_sam3_checkpoint=""
  local resolved_sam3_bpe=""
  local resolved_sam3_home=""
  local resolved_rmbg=""

  if [ -n "${PAPER2ANY_ASSET_ROOT:-}" ]; then
    legacy_sam3_checkpoint="$PAPER2ANY_ASSET_ROOT/models/sam3/sam3.pt"
    legacy_sam3_bpe="$PAPER2ANY_ASSET_ROOT/models/sam3/bpe_simple_vocab_16e6.txt.gz"
    legacy_sam3_home="$PAPER2ANY_ASSET_ROOT/sam3_src"
    legacy_rmbg="$PAPER2ANY_ASSET_ROOT/models/RMBG-2.0"
  fi

  resolved_sam3_checkpoint="$(
    choose_first_existing \
      "${SAM3_CHECKPOINT_PATH:-}" \
      "$PROJECT_ROOT/models/sam3/sam3.pt" \
      "$legacy_sam3_checkpoint" \
      || true
  )"
  resolved_sam3_bpe="$(
    choose_first_existing \
      "${SAM3_BPE_PATH:-}" \
      "$PROJECT_ROOT/models/sam3/bpe_simple_vocab_16e6.txt.gz" \
      "$legacy_sam3_bpe" \
      "$legacy_sam3_home/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
      || true
  )"
  resolved_sam3_home="$(
    choose_first_existing \
      "${SAM3_HOME:-}" \
      "$PROJECT_ROOT/models/sam3-official/sam3" \
      "$legacy_sam3_home" \
      || true
  )"
  resolved_rmbg="$(
    choose_first_existing \
      "${RMBG_MODEL_PATH:-}" \
      "$PROJECT_ROOT/models/RMBG-2.0" \
      "$legacy_rmbg" \
      || true
  )"

  if [ -n "$resolved_sam3_checkpoint" ]; then
    export SAM3_CHECKPOINT_PATH="$resolved_sam3_checkpoint"
    export PAPER2DRAWIO_SAM3_CHECKPOINT_PATH="$resolved_sam3_checkpoint"
  fi
  if [ -n "$resolved_sam3_bpe" ]; then
    export SAM3_BPE_PATH="$resolved_sam3_bpe"
    export PAPER2DRAWIO_SAM3_BPE_PATH="$resolved_sam3_bpe"
  fi
  if [ -n "$resolved_sam3_home" ]; then
    export SAM3_HOME="$resolved_sam3_home"
  fi
  if [ -n "$resolved_rmbg" ]; then
    export RMBG_MODEL_PATH="$resolved_rmbg"
  fi
  if [ -n "${SAM3_SERVER_URLS:-}" ]; then
    export SAM3_SERVER_URLS
  fi
}

resolve_python() {
  if validate_python "${APP_PYTHON:-}"; then
    export APP_PYTHON
    return 0
  fi

  if [ -n "$APP_CONDA_ENV" ] && [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$APP_CONDA_ENV" >/dev/null 2>&1 || {
      echo "Failed to activate conda env: $APP_CONDA_ENV"
      return 1
    }

    APP_PYTHON="$(command -v python || true)"
    if validate_python "$APP_PYTHON"; then
      export APP_PYTHON
      return 0
    fi

    echo "Conda env '$APP_CONDA_ENV' is present but missing required runtime deps, falling back."
    conda deactivate >/dev/null 2>&1 || true
  fi

  for candidate in \
    "${APP_FALLBACK_PYTHON:-}" \
    "$(command -v python3 || true)" \
    "$(command -v python || true)"; do
    if validate_python "$candidate"; then
      APP_PYTHON="$candidate"
      export APP_PYTHON
      return 0
    fi
  done

  echo "No usable python runtime found. Checked conda env '$APP_CONDA_ENV' and fallback interpreters."
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

# 启动前先做一次清理，避免残留 worker 占住端口。
"$PROJECT_ROOT/deploy/stop.sh" >/dev/null 2>&1 || true
sleep 1

existing_pids="$(find_port_listener_pids "$APP_PORT" || true)"
if [ -n "$existing_pids" ]; then
  echo "Port $APP_PORT is still occupied. Refusing to start."
  echo "$existing_pids" | sed 's/^/LISTEN_PID: /'
  exit 1
fi

maybe_source_model_server_env
resolve_python || exit 1
prepare_runtime_env

start_cmd=(
  "$APP_PYTHON" -m uvicorn fastapi_app.main:app
  --host "$APP_HOST"
  --workers "$APP_WORKERS"
  --port "$APP_PORT"
  --log-level info
)

# 使用 setsid + nohup 彻底脱离当前 shell，避免多 worker 被会话一起带走。
if command -v setsid >/dev/null 2>&1; then
  nohup setsid env \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    SAM3_SERVER_URLS="${SAM3_SERVER_URLS:-}" \
    SAM3_HOME="${SAM3_HOME:-}" \
    SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-}" \
    SAM3_BPE_PATH="${SAM3_BPE_PATH:-}" \
    PAPER2DRAWIO_SAM3_CHECKPOINT_PATH="${PAPER2DRAWIO_SAM3_CHECKPOINT_PATH:-}" \
    PAPER2DRAWIO_SAM3_BPE_PATH="${PAPER2DRAWIO_SAM3_BPE_PATH:-}" \
    RMBG_MODEL_PATH="${RMBG_MODEL_PATH:-}" \
    "${start_cmd[@]}" >> "$LOG_DIR/app.log" 2>&1 < /dev/null &
else
  nohup env \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    SAM3_SERVER_URLS="${SAM3_SERVER_URLS:-}" \
    SAM3_HOME="${SAM3_HOME:-}" \
    SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-}" \
    SAM3_BPE_PATH="${SAM3_BPE_PATH:-}" \
    PAPER2DRAWIO_SAM3_CHECKPOINT_PATH="${PAPER2DRAWIO_SAM3_CHECKPOINT_PATH:-}" \
    PAPER2DRAWIO_SAM3_BPE_PATH="${PAPER2DRAWIO_SAM3_BPE_PATH:-}" \
    RMBG_MODEL_PATH="${RMBG_MODEL_PATH:-}" \
    "${start_cmd[@]}" >> "$LOG_DIR/app.log" 2>&1 < /dev/null &
fi

echo $! > "$PID_FILE"
disown 2>/dev/null || true

if wait_for_port_listener "$APP_PORT" 20; then
  echo "FastAPI app started with PID: $(cat "$PID_FILE")"
  exit 0
fi

echo "FastAPI app failed to start. Check $LOG_DIR/app.log"
exit 1
