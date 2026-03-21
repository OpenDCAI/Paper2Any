#!/bin/bash

set -euo pipefail

# ==============================================================================
#  MinerU & SAM Production Launcher v2.0
#  "One GPU, One Instance, Maximum Power"
# ==============================================================================

# ------------------------------------------------------------------------------
#  🎨 Colors & Styles
# ------------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ------------------------------------------------------------------------------
#  ⚙️ Configuration
# ------------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PAPER2ANY_PYTHON="${PAPER2ANY_PYTHON:-$(command -v python3)}"

# MinerU Config
MINERU_MODEL="models/MinerU2.5-2509-1.2B"
MINERU_GPU_UTIL=0.85
MINERU_MAX_SEQS=64
MINERU_GPUS=(1 2 3)
MINERU_START_PORT=8011

# SAM3 Config
SAM3_GPUS=(4 5)
SAM3_START_PORT=8021
SAM3_CHECKPOINT_PATH="$ROOT_DIR/models/sam3/sam3.pt"
SAM3_BPE_PATH="$ROOT_DIR/models/sam3/bpe_simple_vocab_16e6.txt.gz"

# ------------------------------------------------------------------------------
#  🛠️ Helper Functions
# ------------------------------------------------------------------------------

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERR]${NC} $1"; }
log_debug() { echo -e "${CYAN}[DBG]${NC} $1"; }

# A cool spinner for waiting
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

check_port() {
    local port=$1
    lsof -i:$port > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        return 0 # Port is in use
    else
        return 1 # Port is free
    fi
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

wait_for_port() {
    local port=$1
    local label=$2
    local timeout=${3:-120}
    local waited=0

    while [ "$waited" -lt "$timeout" ]; do
        if check_port "$port"; then
            log_success "$label is listening on :$port"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done

    log_error "$label failed to bind :$port within ${timeout}s"
    return 1
}

cleanup_cluster_ports() {
    local ports=({8010..8024} 8003)
    for port in "${ports[@]}"; do
        kill_port "$port"
    done
}

check_cuda_runtime() {
    "$PAPER2ANY_PYTHON" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"TORCH_IMPORT_ERROR: {exc}")
    sys.exit(1)

available = torch.cuda.is_available()
count = 0
try:
    count = torch.cuda.device_count()
except Exception:
    count = 0

print(f"torch.cuda.is_available={available}")
print(f"torch.cuda.device_count={count}")
if not available or count <= 0:
    sys.exit(1)
PY
}

# ------------------------------------------------------------------------------
#  🚀 Main Execution Flow
# ------------------------------------------------------------------------------

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
echo -e "  Target: ${BOLD}High Concurrency / Single Instance Mode${NC}"
echo -e "  Log Dir: $LOG_DIR"
echo -e "  Python:  $PAPER2ANY_PYTHON"
echo "------------------------------------------------------------"

log_info "Running CUDA preflight..."
if ! check_cuda_runtime; then
    log_error "Current Python environment cannot access CUDA. Activate a CUDA-capable env before starting model servers."
    exit 1
fi

# --- Step 1: Deep Cleanup ---
log_info "Initiating deep cleanup sequence..."

# Kill specific ports
PORTS_TO_CLEAN=({8010..8024} 8003)
for port in "${PORTS_TO_CLEAN[@]}"; do
    kill_port $port
done

# Nuke process names
log_info "Nuking vLLM and worker processes..."
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "sam_server" 2>/dev/null || true
pkill -9 -f "sam3_server" 2>/dev/null || true
pkill -9 -f "ocr_server" 2>/dev/null || true
pkill -9 -f "generic_lb.py --port 8010" 2>/dev/null || true
pkill -9 -f "generic_lb.py --port 8020" 2>/dev/null || true

sleep 2
log_success "Cleanup complete. System is clean."

# --- Step 2: Launch MinerU (vLLM) ---
echo "------------------------------------------------------------"
log_info "Launching MinerU Cluster (vLLM)"
log_info "Config: Util=$MINERU_GPU_UTIL | MaxSeqs=$MINERU_MAX_SEQS"

MINERU_BACKENDS=""

for i in "${!MINERU_GPUS[@]}"; do
    gpu_id=${MINERU_GPUS[$i]}
    port=$((MINERU_START_PORT + i))
    
    log_info "Booting instance on GPU $gpu_id @ Port $port..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup "$PAPER2ANY_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MINERU_MODEL" \
        --served-model-name "mineru" \
        --host 127.0.0.1 \
        --port $port \
        --logits-processors mineru_vl_utils:MinerULogitsProcessor \
        --gpu-memory-utilization $MINERU_GPU_UTIL \
        --max-num-seqs $MINERU_MAX_SEQS \
        --trust-remote-code \
        --enforce-eager \
        > "$LOG_DIR/mineru_gpu${gpu_id}.log" 2>&1 &
        
    MINERU_BACKENDS+="http://127.0.0.1:$port "
done

# --- Step 3: Launch SAM ---
echo "------------------------------------------------------------"
log_info "Launching SAM3 Cluster"

SAM3_BACKENDS=""

for i in "${!SAM3_GPUS[@]}"; do
    gpu_id=${SAM3_GPUS[$i]}
    port=$((SAM3_START_PORT + i))
    
    log_info "Booting SAM3 on GPU $gpu_id @ Port $port..."
    
    env CUDA_VISIBLE_DEVICES=$gpu_id nohup "$PAPER2ANY_PYTHON" -m dataflow_agent.toolkits.model_servers.sam3_server \
        --port $port \
        --host 0.0.0.0 \
        --checkpoint "$SAM3_CHECKPOINT_PATH" \
        --bpe "$SAM3_BPE_PATH" \
        --device cuda \
        > "$LOG_DIR/sam_${gpu_id}.log" 2>&1 &
        
    SAM3_BACKENDS+="http://127.0.0.1:$port "
done

# --- Step 4: Launch OCR ---
echo "------------------------------------------------------------"
log_info "Starting OCR Service (CPU)..."
CUDA_VISIBLE_DEVICES="" nohup "$PAPER2ANY_PYTHON" -m uvicorn dataflow_agent.toolkits.model_servers.ocr_server:app \
    --port 8003 --host 0.0.0.0 --workers 4 \
    > "$LOG_DIR/ocr_server.log" 2>&1 &
log_success "OCR Service running on :8003"

# --- Step 5: Validate model backends ---
echo "------------------------------------------------------------"
log_info "Validating model backends..."

failed=0
for i in "${!MINERU_GPUS[@]}"; do
    port=$((MINERU_START_PORT + i))
    wait_for_port "$port" "MinerU backend" 240 || failed=1
done

for i in "${!SAM3_GPUS[@]}"; do
    port=$((SAM3_START_PORT + i))
    wait_for_port "$port" "SAM3 backend" 120 || failed=1
done

wait_for_port 8003 "OCR service" 30 || failed=1

if [ "$failed" -ne 0 ]; then
    log_error "Model server startup incomplete. Check logs under $LOG_DIR"
    cleanup_cluster_ports
    exit 1
fi

# --- Step 6: Launch Load Balancers ---
echo "------------------------------------------------------------"
log_info "Initializing Load Balancers..."

# MinerU LB
nohup "$PAPER2ANY_PYTHON" dataflow_agent/toolkits/model_servers/generic_lb.py \
    --port 8010 \
    --name "MinerU LB" \
    --backends $MINERU_BACKENDS \
    > "$LOG_DIR/mineru_lb.log" 2>&1 &
log_success "MinerU LB running on :8010 -> [ $MINERU_BACKENDS]"

# SAM3 LB
nohup "$PAPER2ANY_PYTHON" dataflow_agent/toolkits/model_servers/generic_lb.py \
    --port 8020 \
    --name "SAM3 LB" \
    --backends $SAM3_BACKENDS \
    > "$LOG_DIR/sam_lb.log" 2>&1 &
log_success "SAM3 LB running on :8020 -> [ $SAM3_BACKENDS]"

# --- Step 7: Validate load balancers ---
echo "------------------------------------------------------------"
log_info "Validating load balancers..."

failed=0
wait_for_port 8010 "MinerU LB" 30 || failed=1
wait_for_port 8020 "SAM3 LB" 30 || failed=1

if [ "$failed" -ne 0 ]; then
    log_error "Load balancer startup incomplete. Check logs under $LOG_DIR"
    cleanup_cluster_ports
    exit 1
fi

# --- Final Check ---
echo "------------------------------------------------------------"
echo -e "${GREEN}${BOLD}ALL SYSTEMS GO!${NC}"
echo -e "Monitor logs with: ${YELLOW}tail -f logs/*.log${NC}"
echo ""
