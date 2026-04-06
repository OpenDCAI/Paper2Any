#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

SAM3_GPU="${SAM3_GPU:-0}"
SAM3_PORT="${SAM3_PORT:-8001}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-$ROOT_DIR/models/sam3/sam3.pt}"
SAM3_BPE_PATH="${SAM3_BPE_PATH:-$ROOT_DIR/models/sam3/bpe_simple_vocab_16e6.txt.gz}"
SAM3_HOME="${SAM3_HOME:-$ROOT_DIR/models/sam3-official/sam3}"
SAM3_SERVER_URLS="${SAM3_SERVER_URLS:-http://127.0.0.1:${SAM3_PORT}}"

PAPER2ANY_PYTHON="${PAPER2ANY_PYTHON:-}"
if [[ -z "$PAPER2ANY_PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PAPER2ANY_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PAPER2ANY_PYTHON="$(command -v python)"
  else
    echo "[ERROR] python executable not found"
    exit 1
  fi
fi

kill_port() {
  local port="$1"
  local pid
  pid="$(lsof -t -i:"$port" 2>/dev/null || true)"
  if [[ -n "$pid" ]]; then
    echo "[WARN] Killing PID ${pid} on port ${port}"
    kill -9 $pid || true
  fi
}

safe_kill_port() {
  local port="$1"
  case "$port" in
    8001|8000|3000)
      kill_port "$port"
      ;;
    *)
      echo "[WARN] Port $port not in allowlist (8001/8000/3000), skip kill"
      ;;
  esac
}

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Logs: $LOG_DIR"
echo "[INFO] Python: $PAPER2ANY_PYTHON"
echo "[INFO] SAM3 GPU: $SAM3_GPU, SAM3 port: $SAM3_PORT"
echo "[INFO] Backend: $BACKEND_PORT, Frontend: $FRONTEND_PORT"

safe_kill_port "$SAM3_PORT"
safe_kill_port "$BACKEND_PORT"
safe_kill_port "$FRONTEND_PORT"

cd "$ROOT_DIR"

echo "[STEP] Start SAM3 service..."
env CUDA_VISIBLE_DEVICES="$SAM3_GPU" \
    SAM3_HOME="$SAM3_HOME" \
    SAM3_CHECKPOINT_PATH="$SAM3_CHECKPOINT_PATH" \
    SAM3_BPE_PATH="$SAM3_BPE_PATH" \
    nohup "$PAPER2ANY_PYTHON" -m dataflow_agent.toolkits.model_servers.sam3_server \
      --host 0.0.0.0 \
      --port "$SAM3_PORT" \
      --checkpoint "$SAM3_CHECKPOINT_PATH" \
      --bpe "$SAM3_BPE_PATH" \
      --device cuda \
      > "$LOG_DIR/local_sam3.log" 2>&1 &

sleep 2

echo "[STEP] Start backend..."
env SAM3_SERVER_URLS="$SAM3_SERVER_URLS" \
    nohup "$PAPER2ANY_PYTHON" -m uvicorn fastapi_app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" \
      > "$LOG_DIR/local_backend.log" 2>&1 &

sleep 2

echo "[STEP] Start frontend..."
cd "$ROOT_DIR/frontend-workflow"
env VITE_API_BASE_URL="http://127.0.0.1:${BACKEND_PORT}" \
    nohup npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" \
      > "$LOG_DIR/local_frontend.log" 2>&1 &

echo "[OK] Local dev stack started."
echo "      SAM3 health:    http://127.0.0.1:${SAM3_PORT}/health"
echo "      Backend health: http://127.0.0.1:${BACKEND_PORT}/health"
echo "      Frontend:       http://127.0.0.1:${FRONTEND_PORT}"
echo "      Logs:           $LOG_DIR/local_sam3.log"
echo "                      $LOG_DIR/local_backend.log"
echo "                      $LOG_DIR/local_frontend.log"
