#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE_DIR="$PROJECT_ROOT/deploy/profiles"
PROFILE="$PROFILE_DIR/muxi.env"
PROFILE_EXAMPLE="$PROFILE_DIR/muxi.env.example"

if [ -f "$PROFILE" ]; then
  ACTIVE_PROFILE="$PROFILE"
elif [ -f "$PROFILE_EXAMPLE" ]; then
  ACTIVE_PROFILE="$PROFILE_EXAMPLE"
else
  echo "Profile not found: $PROFILE or $PROFILE_EXAMPLE"
  exit 1
fi

set -a
source "$ACTIVE_PROFILE"
set +a

echo "Using deploy profile: $ACTIVE_PROFILE"

bash "$PROJECT_ROOT/script/prepare_local_models.sh"
bash "$PROJECT_ROOT/script/start_model_servers.sh"
bash "$PROJECT_ROOT/deploy/start.sh"
bash "$PROJECT_ROOT/deploy/start_frontend.sh"
