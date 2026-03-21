#!/bin/bash

# Shared FastAPI runtime config for deploy scripts.
# Environment variables can override these defaults.

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_WORKERS="${APP_WORKERS:-1}"
APP_CONDA_ENV="${APP_CONDA_ENV:-}"
APP_PYTHON="${APP_PYTHON:-/opt/conda/bin/python}"
APP_FALLBACK_PYTHON="${APP_FALLBACK_PYTHON:-/opt/conda/bin/python}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"

PAPER2ANY_ASSET_ROOT="${PAPER2ANY_ASSET_ROOT:-/mnt/paper2any/lz/github-proj/Paper2Any}"
MODEL_SERVER_ENV_FILE="${MODEL_SERVER_ENV_FILE:-logs/model_servers.env}"

SAM3_SERVER_URLS="${SAM3_SERVER_URLS:-http://127.0.0.1:8021}"
SAM3_HOME="${SAM3_HOME:-$PAPER2ANY_ASSET_ROOT/sam3_src}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-$PAPER2ANY_ASSET_ROOT/models/sam3/sam3.pt}"
SAM3_BPE_PATH="${SAM3_BPE_PATH:-$PAPER2ANY_ASSET_ROOT/models/sam3/bpe_simple_vocab_16e6.txt.gz}"
RMBG_MODEL_PATH="${RMBG_MODEL_PATH:-}"
