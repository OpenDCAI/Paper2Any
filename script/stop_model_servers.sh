#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
STATE_ENV_FILE="${MODEL_SERVER_ENV_FILE:-$LOG_DIR/model_servers.env}"
SAM3_START_PORT="${SAM3_START_PORT:-8021}"
OCR_PORT="${OCR_PORT:-8003}"

find_port_listener_pids() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u
        return 0
    fi

    if command -v ss >/dev/null 2>&1; then
        ss -ltnp 2>/dev/null \
            | awk -v port=":$port" '$4 ~ port { print $NF }' \
            | grep -oE 'pid=[0-9]+' \
            | cut -d= -f2 \
            | sort -u
        return 0
    fi

    if command -v netstat >/dev/null 2>&1; then
        netstat -ltnp 2>/dev/null \
            | awk -v port=":$port" '$4 ~ port { split($7, parts, "/"); if (parts[1] ~ /^[0-9]+$/) print parts[1] }' \
            | sort -u
        return 0
    fi

    return 1
}

kill_port() {
    local port="$1"
    local pid
    while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        kill -KILL "$pid" 2>/dev/null || true
    done < <(find_port_listener_pids "$port" || true)
}

pkill -f "dataflow_agent.toolkits.model_servers.sam3_server" 2>/dev/null || true
pkill -f "dataflow_agent.toolkits.model_servers.ocr_server" 2>/dev/null || true
pkill -f "generic_lb.py --port 8010" 2>/dev/null || true
pkill -f "generic_lb.py --port 8020" 2>/dev/null || true

for port in $(seq "$SAM3_START_PORT" "$((SAM3_START_PORT + 31))"); do
    kill_port "$port"
done
kill_port "$OCR_PORT"
kill_port 8010
kill_port 8020

rm -f "$STATE_ENV_FILE"
