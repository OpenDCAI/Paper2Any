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
STATE_ENV_FILE="${MODEL_SERVER_ENV_FILE:-$LOG_DIR/model_servers.env}"

PAPER2ANY_PYTHON="${PAPER2ANY_PYTHON:-${APP_PYTHON:-$(command -v python3 || command -v python || true)}}"
PAPER2ANY_ASSET_ROOT="${PAPER2ANY_ASSET_ROOT:-}"
DEPLOY_TARGET="${DEPLOY_TARGET:-generic}"
GPU_QUERY_TOOL="${GPU_QUERY_TOOL:-auto}"

SAM3_ENABLED="${SAM3_ENABLED:-1}"
SAM3_GPU_MODE="${SAM3_GPU_MODE:-auto}"
SAM3_GPUS_RAW="${SAM3_GPUS:-}"
SAM3_INSTANCES_PER_GPU="${SAM3_INSTANCES_PER_GPU:-1}"
SAM3_MAX_INSTANCES="${SAM3_MAX_INSTANCES:-0}"
SAM3_START_PORT="${SAM3_START_PORT:-8021}"
SAM3_HOST="${SAM3_HOST:-127.0.0.1}"
SAM3_STARTUP_STRATEGY="${SAM3_STARTUP_STRATEGY:-parallel}"
SAM3_STARTUP_STAGGER_SEC="${SAM3_STARTUP_STAGGER_SEC:-0}"
SAM3_INSTANCE_HEALTH_TIMEOUT="${SAM3_INSTANCE_HEALTH_TIMEOUT:-360}"
SAM3_HOME="${SAM3_HOME:-}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-}"
SAM3_BPE_PATH="${SAM3_BPE_PATH:-}"
MINERU_LOCAL_ENABLED="${MINERU_LOCAL_ENABLED:-0}"
DRIPPER_AUTOSTOP="${DRIPPER_AUTOSTOP:-0}"

OCR_ENABLED="${OCR_ENABLED:-0}"
OCR_PORT="${OCR_PORT:-8003}"
OCR_HOST="${OCR_HOST:-127.0.0.1}"
OCR_WORKERS="${OCR_WORKERS:-1}"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERR]${NC} $1"; }
log_debug() { echo -e "${CYAN}[DBG]${NC} $1"; }

find_port_listener_pids() {
    local port=$1
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

kill_port() {
    local port=$1
    while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        log_warn "Port $port is busy (PID: $pid). Killing..."
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        kill -KILL "$pid" 2>/dev/null || true
    done < <(find_port_listener_pids "$port" || true)
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

discover_available_gpus() {
    if [ "$SAM3_GPU_MODE" = "manual" ] && [ -n "$SAM3_GPUS_RAW" ]; then
        printf '%s\n' "$SAM3_GPUS_RAW" | tr ', ' '\n\n' | awk 'NF { print $1 }'
        return 0
    fi

    local tool="$GPU_QUERY_TOOL"
    if [ "$tool" = "auto" ]; then
        case "$DEPLOY_TARGET" in
            muxi) tool="mx-smi" ;;
            nv) tool="nvidia-smi" ;;
            *) tool="torch" ;;
        esac
    fi

    if [ "$tool" = "nvidia-smi" ] && command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits
        return 0
    fi

    if [ "$tool" = "mx-smi" ] && command -v mx-smi >/dev/null 2>&1; then
        mx-smi -L | awk '
            /^GPU#[0-9]+/ && $0 ~ /Available/ {
                gsub("GPU#", "", $1)
                print $1
            }
        '
        return 0
    fi

    "$PAPER2ANY_PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(1)

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
}

prepare_sam3_paths() {
    local legacy_sam3_checkpoint=""
    local legacy_sam3_bpe=""
    local legacy_sam3_home=""

    if [ -n "$PAPER2ANY_ASSET_ROOT" ]; then
        legacy_sam3_checkpoint="$PAPER2ANY_ASSET_ROOT/models/sam3/sam3.pt"
        legacy_sam3_bpe="$PAPER2ANY_ASSET_ROOT/models/sam3/bpe_simple_vocab_16e6.txt.gz"
        legacy_sam3_home="$PAPER2ANY_ASSET_ROOT/sam3_src"
    fi

    SAM3_CHECKPOINT_PATH="$(
        choose_first_existing \
            "${SAM3_CHECKPOINT_PATH:-}" \
            "$ROOT_DIR/models/sam3/sam3.pt" \
            "$legacy_sam3_checkpoint" \
            || true
    )"
    SAM3_BPE_PATH="$(
        choose_first_existing \
            "${SAM3_BPE_PATH:-}" \
            "$ROOT_DIR/models/sam3/bpe_simple_vocab_16e6.txt.gz" \
            "$legacy_sam3_bpe" \
            "$legacy_sam3_home/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
            || true
    )"
    SAM3_HOME="$(
        choose_first_existing \
            "${SAM3_HOME:-}" \
            "$ROOT_DIR/models/sam3-official/sam3" \
            "$legacy_sam3_home" \
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

build_sam3_launch_gpu_ids() {
    local gpu_id
    local replica

    SAM3_GPU_IDS=()

    if [ "$SAM3_GPU_MODE" = "manual" ] && [ -n "$SAM3_GPUS_RAW" ]; then
        while IFS= read -r gpu_id; do
            gpu_id="$(trim "$gpu_id")"
            [ -n "$gpu_id" ] || continue
            SAM3_GPU_IDS+=("$gpu_id")
        done < <(printf '%s\n' "$SAM3_GPUS_RAW" | tr ', ' '\n\n')
        return 0
    fi

    while IFS= read -r gpu_id; do
        gpu_id="$(trim "$gpu_id")"
        [ -n "$gpu_id" ] || continue

        for replica in $(seq 1 "$SAM3_INSTANCES_PER_GPU"); do
            if [ "$SAM3_MAX_INSTANCES" -gt 0 ] && [ "${#SAM3_GPU_IDS[@]}" -ge "$SAM3_MAX_INSTANCES" ]; then
                return 0
            fi
            SAM3_GPU_IDS+=("$gpu_id")
        done
    done < <(discover_available_gpus)
}

cleanup_ports() {
    local port
    for port in $(seq "$SAM3_START_PORT" "$((SAM3_START_PORT + 31))"); do
        kill_port "$port"
    done
    kill_port "$OCR_PORT"
    kill_port 8010
    kill_port 8020
}

cleanup_processes() {
    pkill -f "dataflow_agent.toolkits.model_servers.sam3_server" 2>/dev/null || true
    pkill -f "dataflow_agent.toolkits.model_servers.ocr_server" 2>/dev/null || true
    pkill -f "generic_lb.py --port 8010" 2>/dev/null || true
    pkill -f "generic_lb.py --port 8020" 2>/dev/null || true
    if [ "$DRIPPER_AUTOSTOP" = "1" ]; then
        pkill -f "python -m dripper.server" 2>/dev/null || true
    fi
}

write_state_env() {
    local sam3_urls="$1"

    mkdir -p "$LOG_DIR"
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

launch_sam3_instance() {
    local gpu_id="$1"
    local port="$2"
    local instance_id="$3"
    local log_file="$LOG_DIR/sam3_gpu${gpu_id}_inst${instance_id}_port${port}.log"

    log_info "Booting SAM3 on GPU $gpu_id @ Port $port..."

    if command -v setsid >/dev/null 2>&1; then
        nohup setsid env \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            SAM3_HOME="$SAM3_HOME" \
            SAM3_CHECKPOINT_PATH="$SAM3_CHECKPOINT_PATH" \
            SAM3_BPE_PATH="$SAM3_BPE_PATH" \
            "$PAPER2ANY_PYTHON" -m dataflow_agent.toolkits.model_servers.sam3_server \
                --host "$SAM3_HOST" \
                --port "$port" \
                --checkpoint "$SAM3_CHECKPOINT_PATH" \
                --bpe "$SAM3_BPE_PATH" \
                --device cuda \
                > "$log_file" 2>&1 < /dev/null &
    else
        nohup env \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            SAM3_HOME="$SAM3_HOME" \
            SAM3_CHECKPOINT_PATH="$SAM3_CHECKPOINT_PATH" \
            SAM3_BPE_PATH="$SAM3_BPE_PATH" \
            "$PAPER2ANY_PYTHON" -m dataflow_agent.toolkits.model_servers.sam3_server \
                --host "$SAM3_HOST" \
                --port "$port" \
                --checkpoint "$SAM3_CHECKPOINT_PATH" \
                --bpe "$SAM3_BPE_PATH" \
                --device cuda \
                > "$log_file" 2>&1 < /dev/null &
    fi

    if [ "$SAM3_STARTUP_STAGGER_SEC" -gt 0 ]; then
        sleep "$SAM3_STARTUP_STAGGER_SEC"
    fi
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
echo -e "  Target: ${BOLD}Unified Local Model Service${NC}"
echo -e "  Log Dir: $LOG_DIR"
echo -e "  Python:  $PAPER2ANY_PYTHON"
echo "------------------------------------------------------------"

if ! validate_python_runtime; then
    log_error "Python runtime '$PAPER2ANY_PYTHON' is missing FastAPI/Torch/OpenCV runtime deps."
    exit 1
fi

log_info "Running CUDA preflight..."
if ! check_cuda_runtime; then
    log_error "Current Python runtime cannot access a CUDA-compatible backend."
    exit 1
fi

prepare_sam3_paths

log_info "Cleaning stale local model processes..."
cleanup_ports
cleanup_processes
sleep 1
log_success "Cleanup complete."

build_sam3_launch_gpu_ids
if [ "$SAM3_ENABLED" = "1" ] && [ "${#SAM3_GPU_IDS[@]}" -eq 0 ]; then
    log_error "No available GPUs detected for SAM3."
    exit 1
fi

echo "------------------------------------------------------------"
if [ "$MINERU_LOCAL_ENABLED" = "0" ]; then
    log_info "MinerU local service is disabled. This deployment expects MinerU API."
fi
if [ "$OCR_ENABLED" = "0" ]; then
    log_info "OCR local service is disabled. This deployment expects remote OCR API."
fi

SAM3_URLS=()
if [ "$SAM3_ENABLED" = "1" ]; then
    log_info "Launching SAM3 instances on GPUs: ${SAM3_GPU_IDS[*]}"
    for i in "${!SAM3_GPU_IDS[@]}"; do
        gpu_id=${SAM3_GPU_IDS[$i]}
        port=$((SAM3_START_PORT + i))
        instance_id=$((i + 1))
        sam3_url="http://127.0.0.1:$port"

        launch_sam3_instance "$gpu_id" "$port" "$instance_id"
        SAM3_URLS+=("$sam3_url")

        if [ "$SAM3_STARTUP_STRATEGY" = "sequential" ]; then
            wait_for_http "${sam3_url}/health" "SAM3 backend" "$SAM3_INSTANCE_HEALTH_TIMEOUT" || {
                log_error "SAM3 sequential startup failed on GPU $gpu_id port $port"
                exit 1
            }
        fi
    done
fi

if [ "$OCR_ENABLED" = "1" ]; then
    log_info "Starting local OCR server..."
    if command -v setsid >/dev/null 2>&1; then
        nohup setsid env \
            CUDA_VISIBLE_DEVICES="" \
            "$PAPER2ANY_PYTHON" -m uvicorn dataflow_agent.toolkits.model_servers.ocr_server:app \
            --host "$OCR_HOST" \
            --port "$OCR_PORT" \
            --workers "$OCR_WORKERS" \
            > "$LOG_DIR/ocr_server.log" 2>&1 < /dev/null &
    else
        nohup env \
            CUDA_VISIBLE_DEVICES="" \
            "$PAPER2ANY_PYTHON" -m uvicorn dataflow_agent.toolkits.model_servers.ocr_server:app \
            --host "$OCR_HOST" \
            --port "$OCR_PORT" \
            --workers "$OCR_WORKERS" \
            > "$LOG_DIR/ocr_server.log" 2>&1 < /dev/null &
    fi
fi

echo "------------------------------------------------------------"
log_info "Validating started services..."

failed=0
if [ "$SAM3_ENABLED" = "1" ]; then
    if [ "$SAM3_STARTUP_STRATEGY" = "parallel" ]; then
        for url in "${SAM3_URLS[@]}"; do
            wait_for_http "${url}/health" "SAM3 backend" "$SAM3_INSTANCE_HEALTH_TIMEOUT" || failed=1
        done
    else
        for url in "${SAM3_URLS[@]}"; do
            curl -fsS "${url}/health" > /dev/null 2>&1 || failed=1
        done
    fi
fi

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
