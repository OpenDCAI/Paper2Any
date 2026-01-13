#!/usr/bin/env bash
set -euo pipefail

# 启动模型服务
bash /data/ziyi/Paper2Any/script/start_model_servers.sh

# 准备conda环境
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 启动后端
(conda activate p2a && cd /data/ziyi/Paper2Any/fastapi_app && uvicorn main:app --host 0.0.0.0 --port 8000 --reload) &
BACK_PID=$!

# 启动前端
(cd /data/ziyi/Paper2Any/frontend-workflow && npm run dev) &
FRONT_PID=$!

cleanup() {
  echo "Stopping servers..."
  kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait "$BACK_PID" "$FRONT_PID"
