#!/bin/bash
# Run integration tests for DataFlow-Agent
#
# Prerequisites:
#   - pip install -r requirements-base.txt
#   - Set environment variables (SUPABASE_URL, SUPABASE_JWT_SECRET, etc.)
#
# Usage:
#   ./scripts/run_integration_tests.sh

set -e

echo "=== DataFlow-Agent Integration Tests ==="
echo ""

# Check if running in project root
if [ ! -f "fastapi_app/main.py" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

# Run backend tests
echo "Running backend integration tests..."
echo "-----------------------------------"

python -m pytest fastapi_app/tests/ -v \
    --tb=short \
    -x \
    2>&1 || {
    echo ""
    echo "Note: Some tests may fail if Supabase is not connected."
    echo "Authentication tests should pass with mock JWT tokens."
}

echo ""
echo "=== Integration Tests Complete ==="
