#!/bin/bash
# FastAPI 应用启动脚本

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 确保日志目录存在
mkdir -p logs

# 若已存在旧进程，先尝试清理，避免重复启动
if [ -f logs/uvicorn.pid ]; then
  old_pid=$(cat logs/uvicorn.pid 2>/dev/null)
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    kill "$old_pid" 2>/dev/null || true
    sleep 1
  fi
fi

# 使用 nohup + stdin 重定向彻底脱离当前 shell
nohup uvicorn fastapi_app.main:app --workers 2 --port 9012 \
  --log-level info \
  >> logs/app.log 2>&1 < /dev/null &

echo $! > logs/uvicorn.pid
disown 2>/dev/null || true
echo "FastAPI app started with PID: $(cat logs/uvicorn.pid)"
