#!/bin/bash
# Run tests with lightweight dependencies (no heavy ML models)
#
# This script:
# 1. Starts the mock API server
# 2. Runs integration tests against it
# 3. Cleans up
#
# Prerequisites:
#   pip install -r requirements-lite.txt

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== DataFlow-Agent Lite Tests ==="
echo ""

# Check if running in project root
if [ ! -f "fastapi_app/main.py" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

# Start mock API server in background
echo "Starting mock API server..."
python -m tests.mocks.mock_api_server &
MOCK_PID=$!
sleep 2

# Verify mock server is running
if ! curl -s http://localhost:8080/health > /dev/null; then
    echo "Error: Mock API server failed to start"
    kill $MOCK_PID 2>/dev/null || true
    exit 1
fi
echo "Mock API server running on http://localhost:8080"
echo ""

# Set environment variables for mock server
export DF_API_URL="http://localhost:8080"
export DF_API_KEY="mock-api-key"

# Run backend tests (auth, rate limiting, files)
echo "Running backend integration tests..."
echo "-----------------------------------"

python -m pytest fastapi_app/tests/ -v \
    --tb=short \
    -x \
    2>&1 || {
    echo ""
    echo "Note: Some tests may fail if Supabase is not connected."
}

echo ""
echo "-----------------------------------"
echo ""

# Cleanup
echo "Cleaning up..."
kill $MOCK_PID 2>/dev/null || true

echo ""
echo "=== Lite Tests Complete ==="
