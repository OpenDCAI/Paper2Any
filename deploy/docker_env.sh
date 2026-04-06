#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

load_env_file() {
  local file="$1"
  if [ -f "$file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$file"
    set +a
  fi
}

# Compose-level defaults first.
load_env_file "$PROJECT_ROOT/deploy/docker.env.example"

# Backend runtime vars used by env_file and optional SAM3 defaults.
load_env_file "$PROJECT_ROOT/fastapi_app/.env"

# Frontend public defaults for static build.
load_env_file "$PROJECT_ROOT/frontend-workflow/.env"

# Optional machine profile for local SAM3 path defaults.
if [ -f "$PROJECT_ROOT/deploy/profiles/nv.env" ]; then
  load_env_file "$PROJECT_ROOT/deploy/profiles/nv.env"
elif [ -f "$PROJECT_ROOT/deploy/profiles/muxi.env" ]; then
  load_env_file "$PROJECT_ROOT/deploy/profiles/muxi.env"
fi

# Local docker overrides win last.
load_env_file "$PROJECT_ROOT/deploy/docker.env"

COMPOSE_FILE_ARGS=(-f "$PROJECT_ROOT/docker-compose.yml")
if [ "${DOCKER_HOST_NETWORK:-0}" = "1" ]; then
  COMPOSE_FILE_ARGS+=(-f "$PROJECT_ROOT/docker-compose.host.yml")
fi
