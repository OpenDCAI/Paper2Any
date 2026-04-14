#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# shellcheck disable=SC1091
source "$PROJECT_ROOT/deploy/docker_env.sh"

extra_args=()
if [ "${DOCKER_WITH_SAM3:-0}" = "1" ]; then
  extra_args+=(--profile sam3)
fi

docker compose "${COMPOSE_FILE_ARGS[@]}" "${extra_args[@]}" up -d --build "$@"
