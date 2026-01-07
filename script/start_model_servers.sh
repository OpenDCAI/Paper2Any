#!/bin/bash

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
ROOT_DIR="$(dirname "$0")/.."
LOG_DIR="$ROOT_DIR/logs"

# MinerU Config
MINERU_MODEL="models/MinerU2.5-2509-1.2B"
MINERU_GPU_UTIL=0.85
MINERU_MAX_SEQS=64
MINERU_GPUS=(7 1 2 3)
MINERU_START_PORT=8011

# SAM Config
SAM_GPUS=(4 5 6)
SAM_START_PORT=8021

# ------------------------------------------------------------------------------
#  🛠️ Helper Functions
# ------------------------------------------------------------------------------

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERR]${NC} $1"; }

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
    lsof -i:$port > /dev/null
    if [ $? -eq 0 ]; then
        return 0 # Port is in use
    else
        return 1 # Port is free
    fi
}

kill_port() {
    local port=$1
    local pid=$(lsof -t -i:$port)
    if [ ! -z "$pid" ]; then
        log_warn "Port $port is busy (PID: $pid). Killing..."
        kill -9 $pid 2>/dev/null
    fi
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
echo "------------------------------------------------------------"

# --- Step 1: Deep Cleanup ---
log_info "Initiating deep cleanup sequence..."

# Kill specific ports
PORTS_TO_CLEAN=({8010..8024} 8003)
for port in "${PORTS_TO_CLEAN[@]}"; do
    kill_port $port
done

# Nuke process names
log_info "Nuking vLLM and worker processes..."
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "sam_server" 2>/dev/null
pkill -9 -f "ocr_server" 2>/dev/null

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
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 -m vllm.entrypoints.openai.api_server \
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
log_info "Launching SAM Cluster"

SAM_BACKENDS=""

for i in "${!SAM_GPUS[@]}"; do
    gpu_id=${SAM_GPUS[$i]}
    port=$((SAM_START_PORT + i))
    
    log_info "Booting SAM on GPU $gpu_id @ Port $port..."
    
    env CUDA_VISIBLE_DEVICES=$gpu_id nohup uvicorn dataflow_agent.toolkits.model_servers.sam_server:app \
        --port $port --host 0.0.0.0 \
        > "$LOG_DIR/sam_${gpu_id}.log" 2>&1 &
        
    SAM_BACKENDS+="http://127.0.0.1:$port "
done

# --- Step 4: Launch Load Balancers ---
echo "------------------------------------------------------------"
log_info "Initializing Load Balancers..."

# MinerU LB
nohup python3 dataflow_agent/toolkits/model_servers/generic_lb.py \
    --port 8010 \
    --name "MinerU LB" \
    --backends $MINERU_BACKENDS \
    > "$LOG_DIR/mineru_lb.log" 2>&1 &
log_success "MinerU LB running on :8010 -> [ $MINERU_BACKENDS]"

# SAM LB
nohup python3 dataflow_agent/toolkits/model_servers/generic_lb.py \
    --port 8020 \
    --name "SAM LB" \
    --backends $SAM_BACKENDS \
    > "$LOG_DIR/sam_lb.log" 2>&1 &
log_success "SAM LB running on :8020 -> [ $SAM_BACKENDS]"

# --- Step 5: Launch OCR ---
echo "------------------------------------------------------------"
log_info "Starting OCR Service (CPU)..."
CUDA_VISIBLE_DEVICES="" nohup uvicorn dataflow_agent.toolkits.model_servers.ocr_server:app \
    --port 8003 --host 0.0.0.0 --workers 4 \
    > "$LOG_DIR/ocr_server.log" 2>&1 &
log_success "OCR Service running on :8003"

# --- Final Check ---
echo "------------------------------------------------------------"
echo -e "${GREEN}${BOLD}ALL SYSTEMS GO!${NC}"
echo -e "Monitor logs with: ${YELLOW}tail -f logs/*.log${NC}"
echo ""