#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
LOCAL_SAM3_DIR="$MODELS_DIR/sam3"
LOCAL_SAM3_HOME="$MODELS_DIR/sam3-official/sam3"
LOCAL_RMBG_DIR="$MODELS_DIR/RMBG-2.0"

PAPER2ANY_PYTHON="${PAPER2ANY_PYTHON:-${APP_PYTHON:-$(command -v python3 || command -v python || true)}}"
PAPER2ANY_ASSET_ROOT="${PAPER2ANY_ASSET_ROOT:-}"

LEGACY_SAM3_DIR=""
LEGACY_SAM3_HOME=""
if [ -n "$PAPER2ANY_ASSET_ROOT" ]; then
    LEGACY_SAM3_DIR="$PAPER2ANY_ASSET_ROOT/models/sam3"
    LEGACY_SAM3_HOME="$PAPER2ANY_ASSET_ROOT/sam3_src"
fi

log_info() { echo "[INFO] $1"; }
log_warn() { echo "[WARN] $1"; }
log_ok() { echo "[OK] $1"; }

mkdir -p "$LOCAL_SAM3_DIR" "$LOCAL_RMBG_DIR" "$(dirname "$LOCAL_SAM3_HOME")"

copy_file_if_missing() {
    local src="$1"
    local dst="$2"

    if [ -f "$dst" ]; then
        log_ok "Exists: $dst"
        return 0
    fi
    if [ -z "$src" ] || [ ! -f "$src" ]; then
        log_warn "Missing source file: $src"
        return 1
    fi

    cp -f "$src" "$dst"
    log_ok "Copied: $dst"
}

copy_tree_if_missing() {
    local src="$1"
    local dst="$2"
    local existing_payload

    if [ -d "$dst" ] && [ -f "$dst/sam3/__init__.py" ]; then
        log_ok "Exists: $dst"
        return 0
    fi
    if [ -z "$src" ] || [ ! -d "$src" ]; then
        log_warn "Missing source directory: $src"
        return 1
    fi

    existing_payload="$(find "$dst" -mindepth 1 ! -name '.gitkeep' -print -quit 2>/dev/null || true)"
    if [ -n "$existing_payload" ]; then
        log_warn "Target already contains files, skip copying: $dst"
        return 1
    fi

    rm -rf "$dst"
    cp -a "$src" "$dst"
    log_ok "Copied: $dst"
}

download_rmbg_if_missing() {
    if [ -f "$LOCAL_RMBG_DIR/config.json" ] && [ -f "$LOCAL_RMBG_DIR/model.safetensors" ]; then
        log_ok "Exists: $LOCAL_RMBG_DIR"
        return 0
    fi

    if [ -z "$PAPER2ANY_PYTHON" ]; then
        log_warn "Cannot download RMBG-2.0 because no python runtime is configured."
        return 1
    fi

    log_info "Downloading RMBG-2.0 into $LOCAL_RMBG_DIR"
    "$PAPER2ANY_PYTHON" - <<PY
from modelscope import snapshot_download
snapshot_download(
    "AI-ModelScope/RMBG-2.0",
    local_dir=r"${LOCAL_RMBG_DIR}",
    allow_patterns=["*.json", "*.py", "*.safetensors"],
)
PY

    if [ ! -f "$LOCAL_RMBG_DIR/config.json" ] || [ ! -f "$LOCAL_RMBG_DIR/model.safetensors" ]; then
        log_warn "RMBG download finished but required files are missing: $LOCAL_RMBG_DIR"
        return 1
    fi

    log_ok "Downloaded RMBG-2.0"
}

cleanup_rmbg_extras() {
    rm -rf "$LOCAL_RMBG_DIR/onnx" "$LOCAL_RMBG_DIR/._____temp"
}

copy_file_if_missing "${LEGACY_SAM3_DIR:+$LEGACY_SAM3_DIR/sam3.pt}" "$LOCAL_SAM3_DIR/sam3.pt" || true
copy_file_if_missing "${LEGACY_SAM3_DIR:+$LEGACY_SAM3_DIR/bpe_simple_vocab_16e6.txt.gz}" "$LOCAL_SAM3_DIR/bpe_simple_vocab_16e6.txt.gz" || true
copy_tree_if_missing "$LEGACY_SAM3_HOME" "$LOCAL_SAM3_HOME" || true
download_rmbg_if_missing || true
cleanup_rmbg_extras

cat <<EOF
[DONE] Local model layout prepared:
  SAM3 checkpoint: $LOCAL_SAM3_DIR/sam3.pt
  SAM3 bpe:        $LOCAL_SAM3_DIR/bpe_simple_vocab_16e6.txt.gz
  SAM3 source:     $LOCAL_SAM3_HOME
  RMBG:            $LOCAL_RMBG_DIR
EOF
