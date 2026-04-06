#!/bin/bash
# 启动 8 个 EchoMimic API 实例：卡 4 上 2 个，卡 5/6 各 3 个。
# 端口：8040,8041 (GPU 4), 8050,8051,8052 (GPU 5), 8060,8061,8062 (GPU 6)
# 使用前请确保已安装 echomimic 环境依赖，并修改 ECHOMIMIC_PYTHON 等路径（或通过环境变量传入）。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 子进程必须用 echomimic 环境（含 diffusers 等），不能用 p2v
export ECHOMIMIC_PYTHON="${ECHOMIMIC_PYTHON:-/root/miniconda3/envs/echomimic/bin/python}"
export ECHOMIMIC_CWD="${ECHOMIMIC_CWD:-/data/users/ligang/EchoMimic}"
export ECHOMIMIC_INFER_TIMEOUT="${ECHOMIMIC_INFER_TIMEOUT:-900}"

# (GPU_ID, PORT) 共 8 个
INSTANCES=( "4:8040" "4:8041" "5:8050" "5:8051" "5:8052" "6:8060" "6:8061" "6:8062" )
PID_FILE="${PID_FILE:-$PROJECT_ROOT/.echomimic_api_pids}"

start() {
  echo "Starting 8 EchoMimic API instances..."
  > "$PID_FILE"
  for entry in "${INSTANCES[@]}"; do
    gpu="${entry%%:*}"
    port="${entry##*:}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PORT="$port"
    nohup "$ECHOMIMIC_PYTHON" -u "$SCRIPT_DIR/echomimic_api.py" >> "${SCRIPT_DIR}/echomimic_api_${port}.log" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "  GPU $gpu port $port PID $!"
  done
  echo "Done. PIDs saved to $PID_FILE"
}

stop() {
  echo "Stopping EchoMimic API instances..."
  # 1) 按 PID 文件发 SIGTERM，再 SIGKILL 确保退出
  if [ -f "$PID_FILE" ]; then
    while read -r pid; do
      [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done < "$PID_FILE"
    sleep 2
    while read -r pid; do
      [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi
  # 2) 兜底：按端口杀仍在监听的进程（解决 PID 文件错/丢 或 子进程未退的情况）
  for entry in "${INSTANCES[@]}"; do
    port="${entry##*:}"
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      echo "  Killing process(es) on port $port: $pids"
      echo "$pids" | xargs -r kill -9 2>/dev/null || true
    fi
  done
  echo "Done."
}

case "${1:-}" in
  start) start ;;
  stop)  stop ;;
  *)    echo "Usage: $0 start|stop"; exit 1 ;;
esac
