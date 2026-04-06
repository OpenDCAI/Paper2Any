#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

bash "$PROJECT_ROOT/deploy/stop_frontend.sh" || true
bash "$PROJECT_ROOT/deploy/stop.sh" || true
bash "$PROJECT_ROOT/script/stop_model_servers.sh" || true
