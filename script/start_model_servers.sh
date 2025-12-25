#!/bin/bash

# start_model_servers.sh
# 启动 SAM, OCR, 和 MinerU (vLLM) 模型服务
# 假设运行在 dev/DataFlow-Agent 目录下

# 确保在正确目录
cd "$(dirname "$0")/.."
echo "Current directory: $(pwd)"

# 创建日志目录
mkdir -p logs

# ==================================================================================
# 0. Cleanup Old Processes
# ==================================================================================
echo "Cleaning up old processes..."

# Cleanup MinerU (Ports 8010-8018)
for port in {8010..8018}; do
    pid=$(lsof -t -i:$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done

# Cleanup SAM (Ports 8020-8024)
for port in {8020..8024}; do
    pid=$(lsof -t -i:$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done

# Cleanup OCR (Port 8003)
pid=$(lsof -t -i:8003)
if [ ! -z "$pid" ]; then
    echo "Killing process on port 8003 (PID: $pid)"
    kill -9 $pid 2>/dev/null
fi

sleep 2
echo "Cleanup complete."

# ==================================================================================
# 1. MinerU (vLLM) Services (GPU 0 & 4, 4 instances each)
# ==================================================================================
MINERU_MODEL_PATH="models/MinerU2.5-2509-1.2B"
MINERU_GPU_UTIL=0.2

start_mineru_instance() {
    local gpu_id=$1
    local port=$2
    local instance_id=$3
    
    echo "Starting MinerU Backend $instance_id on GPU $gpu_id (Port $port)..."
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MINERU_MODEL_PATH" \
        --served-model-name "mineru" \
        --host 127.0.0.1 \
        --port $port \
        --logits-processors mineru_vl_utils:MinerULogitsProcessor \
        --gpu-memory-utilization $MINERU_GPU_UTIL \
        --trust-remote-code \
        --enforce-eager \
        > logs/mineru_backend_${instance_id}.log 2>&1 &
    
    local pid=$!
    echo "MinerU Backend $instance_id started with PID $pid"
}

# GPU 0 Instances (Ports 8011-8014)
start_mineru_instance 0 8011 1
sleep 10
start_mineru_instance 0 8012 2
sleep 10
start_mineru_instance 0 8013 3
sleep 10
start_mineru_instance 0 8014 4
sleep 10

# GPU 4 Instances (Ports 8015-8018)
start_mineru_instance 4 8015 5
sleep 10
start_mineru_instance 4 8016 6
sleep 10
start_mineru_instance 4 8017 7
sleep 10
start_mineru_instance 4 8018 8
sleep 10

# MinerU LB (Port 8010)
MINERU_BACKENDS=""
for port in {8011..8018}; do
    MINERU_BACKENDS="$MINERU_BACKENDS http://127.0.0.1:$port"
done

echo "Starting MinerU Load Balancer (Port 8010)..."
nohup python3 dataflow_agent/toolkits/model_servers/generic_lb.py \
    --port 8010 \
    --name "MinerU LB" \
    --backends $MINERU_BACKENDS \
    > logs/mineru_lb.log 2>&1 &
echo "MinerU Load Balancer started"

# ==================================================================================
# 2. SAM Services (GPU 2 & 3, 1 instances each to avoid OOM)
# ==================================================================================
start_sam_instance() {
    local gpu_id=$1
    local port=$2
    local instance_id=$3
    
    echo "Starting SAM Backend $instance_id on GPU $gpu_id (Port $port)..."
    # Explicitly using env to set CUDA_VISIBLE_DEVICES
    env CUDA_VISIBLE_DEVICES=$gpu_id nohup uvicorn dataflow_agent.toolkits.model_servers.sam_server:app \
        --port $port --host 0.0.0.0 \
        > logs/sam_backend_${instance_id}.log 2>&1 &
        
    local pid=$!
    echo "SAM Backend $instance_id started with PID $pid"
}

# GPU 2 Instance (Port 8021)
start_sam_instance 2 8021 1

# GPU 3 Instance (Port 8022)
start_sam_instance 3 8022 2

# SAM LB (Port 8020)
SAM_BACKENDS=""
for port in {8021..8022}; do
    SAM_BACKENDS="$SAM_BACKENDS http://127.0.0.1:$port"
done

echo "Starting SAM Load Balancer (Port 8020)..."
nohup python3 dataflow_agent/toolkits/model_servers/generic_lb.py \
    --port 8020 \
    --name "SAM LB" \
    --backends $SAM_BACKENDS \
    > logs/sam_lb.log 2>&1 &
echo "SAM Load Balancer started"

# ==================================================================================
# 3. OCR Service (CPU)
# ==================================================================================

# OCR LB (Port 8003) - PaddleOCR handles multi-processing internally via --workers
# So we don't need a separate LB for it, uvicorn workers are sufficient for CPU bound tasks
echo "Starting OCR Server on CPU (Port 8003)..."
CUDA_VISIBLE_DEVICES="" nohup uvicorn dataflow_agent.toolkits.model_servers.ocr_server:app --port 8003 --host 0.0.0.0 --workers 4 > logs/ocr_server.log 2>&1 &
OCR_PID=$!
echo "OCR Server started with PID $OCR_PID"

# ==================================================================================
# Summary
# ==================================================================================
echo ""
echo "Model Servers Summary:"
echo "----------------------"
echo "MinerU LB: http://localhost:8010"
echo "  - Backends: 8 instances on GPU 0 & 4 (Ports 8011-8018)"
echo "SAM LB:    http://localhost:8020"
echo "  - Backends: 2 instances on GPU 2 & 3 (Ports 8021-8022)"
echo "OCR:       http://localhost:8003 (CPU, 4 workers)"
echo ""
echo "Logs are in logs/"
