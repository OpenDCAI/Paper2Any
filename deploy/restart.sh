#!/bin/bash
# FastAPI 应用重启脚本

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Restarting FastAPI app..."
"$PROJECT_ROOT/deploy/stop.sh" || true
sleep 2
"$PROJECT_ROOT/deploy/start.sh"
