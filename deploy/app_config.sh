#!/bin/bash

# Shared FastAPI runtime config for deploy scripts.
# Environment variables can override these defaults.

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEPLOY_DIR/.." && pwd)"

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_WORKERS="${APP_WORKERS:-1}"
APP_CONDA_ENV="${APP_CONDA_ENV:-}"
APP_PYTHON="${APP_PYTHON:-/opt/conda/bin/python}"
APP_FALLBACK_PYTHON="${APP_FALLBACK_PYTHON:-/opt/conda/bin/python}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"

# Keep the legacy external repo as a fallback only.
PAPER2ANY_ASSET_ROOT="${PAPER2ANY_ASSET_ROOT:-/mnt/paper2any/lz/github-proj/Paper2Any}"
MODEL_SERVER_ENV_FILE="${MODEL_SERVER_ENV_FILE:-logs/model_servers.env}"

SAM3_SERVER_URLS="${SAM3_SERVER_URLS:-http://127.0.0.1:8021}"
# Leave model paths empty by default so deploy/start.sh can prefer repo-local models first.
SAM3_HOME="${SAM3_HOME:-}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-}"
SAM3_BPE_PATH="${SAM3_BPE_PATH:-}"
RMBG_MODEL_PATH="${RMBG_MODEL_PATH:-}"
