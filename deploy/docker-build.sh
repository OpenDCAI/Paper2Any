#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# shellcheck disable=SC1091
source "$PROJECT_ROOT/deploy/docker_env.sh"

docker compose "${COMPOSE_FILE_ARGS[@]}" build "$@"
