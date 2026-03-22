#!/bin/bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
STATE_ENV_FILE="$LOG_DIR/model_servers.env"

PAPER2ANY_PYTHON="${PAPER2ANY_PYTHON:-/opt/conda/bin/python}"
PAPER2ANY_ASSET_ROOT="${PAPER2ANY_ASSET_ROOT:-/mnt/paper2any/lz/github-proj/Paper2Any}"

SAM3_ENABLED="${SAM3_ENABLED:-1}"
SAM3_GPUS_RAW="${SAM3_GPUS:-1}"
SAM3_MAX_INSTANCES="${SAM3_MAX_INSTANCES:-1}"
SAM3_START_PORT="${SAM3_START_PORT:-8021}"
SAM3_HOST="${SAM3_HOST:-127.0.0.1}"
SAM3_HOME="${SAM3_HOME:-}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-}"
SAM3_BPE_PATH="${SAM3_BPE_PATH:-}"

OCR_ENABLED="${OCR_ENABLED:-0}"
OCR_PORT="${OCR_PORT:-8003}"
OCR_HOST="${OCR_HOST:-127.0.0.1}"
OCR_WORKERS="${OCR_WORKERS:-1}"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERR]${NC} $1"; }
log_debug() { echo -e "${CYAN}[DBG]${NC} $1"; }

check_port() {
    local port=$1
    lsof -iTCP:"$port" -sTCP:LISTEN > /dev/null 2>&1
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

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

validate_python_runtime() {
    [ -x "$PAPER2ANY_PYTHON" ] || return 1
    "$PAPER2ANY_PYTHON" - <<'PY' >/dev/null 2>&1
import cv2
import fastapi
import torch
import uvicorn
PY
}

check_cuda_runtime() {
    "$PAPER2ANY_PYTHON" - <<'PY'
import sys
import torch

available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
print(f"torch.cuda.is_available={available}")
print(f"torch.cuda.device_count={count}")
if not available or count <= 0:
    sys.exit(1)
PY
}

kill_port() {
    local port=$1
    local pid
    pid=$(lsof -t -i:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        log_warn "Port $port is busy (PID: $pid). Killing..."
        kill -9 $pid 2>/dev/null || true
    fi
}

wait_for_http() {
    local url=$1
    local label=$2
    local timeout=${3:-120}
    local waited=0

    while [ "$waited" -lt "$timeout" ]; do
        if curl -fsS "$url" > /dev/null 2>&1; then
            log_success "$label is ready: $url"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done

    log_error "$label failed health check: $url within ${timeout}s"
    return 1
}

discover_available_gpus() {
    if [ "$SAM3_GPUS_RAW" != "auto" ]; then
        echo "$SAM3_GPUS_RAW" | tr ',' ' '
        return 0
    fi

    if "$PAPER2ANY_PYTHON" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
        "$PAPER2ANY_PYTHON" - <<'PY'
import torch

items = []
for idx in range(torch.cuda.device_count()):
    try:
        free_bytes, _ = torch.cuda.mem_get_info(idx)
    except Exception:
        free_bytes = 0
    items.append((free_bytes, idx))

for _, idx in sorted(items, reverse=True):
    print(idx)
PY
        return 0
    fi

    if command -v mx-smi >/dev/null 2>&1; then
        mx-smi -L | awk '
            /^GPU#[0-9]+/ && $0 ~ /Available \(/ {
                gsub("GPU#", "", $1)
                print $1
            }
        '
    fi
}

prepare_sam3_paths() {
    SAM3_CHECKPOINT_PATH="$(
        choose_first_existing \
            "${SAM3_CHECKPOINT_PATH:-}" \
            "$ROOT_DIR/models/sam3/sam3.pt" \
            "$PAPER2ANY_ASSET_ROOT/models/sam3/sam3.pt" \
            || true
    )"
    SAM3_BPE_PATH="$(
        choose_first_existing \
            "${SAM3_BPE_PATH:-}" \
            "$ROOT_DIR/models/sam3/bpe_simple_vocab_16e6.txt.gz" \
            "$PAPER2ANY_ASSET_ROOT/models/sam3/bpe_simple_vocab_16e6.txt.gz" \
            "$PAPER2ANY_ASSET_ROOT/sam3_src/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
            || true
    )"
    SAM3_HOME="$(
        choose_first_existing \
            "${SAM3_HOME:-}" \
            "$ROOT_DIR/models/sam3-official/sam3" \
            "$PAPER2ANY_ASSET_ROOT/sam3_src" \
            || true
    )"

    if [ -z "$SAM3_CHECKPOINT_PATH" ] || [ -z "$SAM3_BPE_PATH" ] || [ -z "$SAM3_HOME" ]; then
        log_error "SAM3 assets are incomplete."
        log_error "SAM3_HOME=$SAM3_HOME"
        log_error "SAM3_CHECKPOINT_PATH=$SAM3_CHECKPOINT_PATH"
        log_error "SAM3_BPE_PATH=$SAM3_BPE_PATH"
        exit 1
    fi
}

cleanup_ports() {
    local ports=({8020..8028} "$OCR_PORT")
    local port
    for port in "${ports[@]}"; do
        kill_port "$port"
    done
}

cleanup_processes() {
    pkill -9 -f "sam3_server" 2>/dev/null || true
    pkill -9 -f "ocr_server" 2>/dev/null || true
    pkill -9 -f "generic_lb.py --port 8020" 2>/dev/null || true
}

write_state_env() {
    local sam3_urls="$1"

    : > "$STATE_ENV_FILE"
    if [ -n "$sam3_urls" ]; then
        printf 'export SAM3_SERVER_URLS=%q\n' "$sam3_urls" >> "$STATE_ENV_FILE"
    fi
    printf 'export SAM3_HOME=%q\n' "$SAM3_HOME" >> "$STATE_ENV_FILE"
    printf 'export SAM3_CHECKPOINT_PATH=%q\n' "$SAM3_CHECKPOINT_PATH" >> "$STATE_ENV_FILE"
    printf 'export SAM3_BPE_PATH=%q\n' "$SAM3_BPE_PATH" >> "$STATE_ENV_FILE"
    printf 'export PAPER2DRAWIO_SAM3_CHECKPOINT_PATH=%q\n' "$SAM3_CHECKPOINT_PATH" >> "$STATE_ENV_FILE"
    printf 'export PAPER2DRAWIO_SAM3_BPE_PATH=%q\n' "$SAM3_BPE_PATH" >> "$STATE_ENV_FILE"
}

cd "$ROOT_DIR" || { log_error "Failed to cd to $ROOT_DIR"; exit 1; }
mkdir -p "$LOG_DIR"

echo -e "${CYAN}${BOLD}"
echo "  ____                         ____    _                  "
echo " |  _ \ __ _ _ __   ___ _ __  |___ \  / \   _ __  _   _ "
echo " | |_) / _\` | '_ \ / _ \ '__|   __) |/ _ \ | '_ \| | | |"
echo " |  __/ (_| | |_) |  __/ |     / __// ___ \| | | | |_| |"
echo " |_|   \__,_| .__/ \___|_|    |_____/_/   \_\_| |_|\__, |"
echo "            |_|                                    |___/ "
echo -e "${NC}"
echo -e "  Target: ${BOLD}MetaX SAM3 Local Service${NC}"
echo -e "  Log Dir: $LOG_DIR"
echo -e "  Python:  $PAPER2ANY_PYTHON"
echo "------------------------------------------------------------"

if ! validate_python_runtime; then
    log_error "Python runtime '$PAPER2ANY_PYTHON' is missing FastAPI/Torch/OpenCV runtime deps."
    exit 1
fi

log_info "Running CUDA preflight..."
if ! check_cuda_runtime; then
    log_error "Current Python runtime cannot access MetaX CUDA devices."
    exit 1
fi

prepare_sam3_paths

log_info "Cleaning stale local SAM3/OCR processes..."
cleanup_ports
cleanup_processes
sleep 1
log_success "Cleanup complete."

SAM3_GPU_IDS=()
while IFS= read -r gpu_id; do
    gpu_id="$(trim "$gpu_id")"
    [ -n "$gpu_id" ] || continue
    SAM3_GPU_IDS+=("$gpu_id")
done < <(discover_available_gpus)

if [ "$SAM3_ENABLED" = "1" ] && [ "${#SAM3_GPU_IDS[@]}" -eq 0 ]; then
    log_error "No available GPUs detected for SAM3."
    exit 1
fi

if [ "${#SAM3_GPU_IDS[@]}" -gt "$SAM3_MAX_INSTANCES" ]; then
    SAM3_GPU_IDS=("${SAM3_GPU_IDS[@]:0:$SAM3_MAX_INSTANCES}")
fi

echo "------------------------------------------------------------"
log_info "MinerU is intentionally not started here. This machine uses MinerU API only."
log_info "OCR server is disabled by default. This machine uses Ali Qwen-VL-OCR API."

SAM3_URLS=()
if [ "$SAM3_ENABLED" = "1" ]; then
    log_info "Launching SAM3 instances on MetaX GPUs: ${SAM3_GPU_IDS[*]}"
    for i in "${!SAM3_GPU_IDS[@]}"; do
        gpu_id=${SAM3_GPU_IDS[$i]}
        port=$((SAM3_START_PORT + i))
        log_info "Booting SAM3 on GPU $gpu_id @ Port $port..."

        env CUDA_VISIBLE_DEVICES="$gpu_id" \
            SAM3_HOME="$SAM3_HOME" \
            SAM3_CHECKPOINT_PATH="$SAM3_CHECKPOINT_PATH" \
            SAM3_BPE_PATH="$SAM3_BPE_PATH" \
            nohup "$PAPER2ANY_PYTHON" -m dataflow_agent.toolkits.model_servers.sam3_server \
                --host "$SAM3_HOST" \
                --port "$port" \
                --checkpoint "$SAM3_CHECKPOINT_PATH" \
                --bpe "$SAM3_BPE_PATH" \
                --device cuda \
                > "$LOG_DIR/sam3_gpu${gpu_id}.log" 2>&1 &

        SAM3_URLS+=("http://127.0.0.1:$port")
    done
fi

if [ "$OCR_ENABLED" = "1" ]; then
    echo "------------------------------------------------------------"
    log_info "Starting local OCR server..."
    CUDA_VISIBLE_DEVICES="" nohup "$PAPER2ANY_PYTHON" -m uvicorn dataflow_agent.toolkits.model_servers.ocr_server:app \
        --host "$OCR_HOST" \
        --port "$OCR_PORT" \
        --workers "$OCR_WORKERS" \
        > "$LOG_DIR/ocr_server.log" 2>&1 &
fi

echo "------------------------------------------------------------"
log_info "Validating started services..."

failed=0
for url in "${SAM3_URLS[@]}"; do
    wait_for_http "${url}/health" "SAM3 backend" 180 || failed=1
done

if [ "$OCR_ENABLED" = "1" ]; then
    wait_for_http "http://127.0.0.1:${OCR_PORT}/health" "OCR backend" 60 || failed=1
fi

if [ "$failed" -ne 0 ]; then
    log_error "Model server startup incomplete. Check logs under $LOG_DIR"
    exit 1
fi

SAM3_URLS_CSV=""
if [ "${#SAM3_URLS[@]}" -gt 0 ]; then
    SAM3_URLS_CSV="$(IFS=,; echo "${SAM3_URLS[*]}")"
fi
write_state_env "$SAM3_URLS_CSV"

echo "------------------------------------------------------------"
echo -e "${GREEN}${BOLD}MODEL SERVICES READY${NC}"
if [ -n "$SAM3_URLS_CSV" ]; then
    echo "SAM3_SERVER_URLS=$SAM3_URLS_CSV"
fi
if [ "$OCR_ENABLED" = "1" ]; then
    echo "OCR_URL=http://127.0.0.1:${OCR_PORT}"
fi
echo "Env file: $STATE_ENV_FILE"
echo -e "Monitor logs with: ${YELLOW}tail -f logs/*.log${NC}"
