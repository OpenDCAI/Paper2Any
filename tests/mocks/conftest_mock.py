"""
Pytest fixtures for mock API server.

These fixtures allow tests to run against a mock OpenAI-compatible API server
instead of making real API calls.

Usage in tests:
    def test_something(mock_api_url):
        # mock_api_url will be "http://localhost:8080"
        # Configure your LLM client to use this URL
        pass
"""

import os
import subprocess
import sys
import time

import pytest
import requests


MOCK_SERVER_PORT = 8080
MOCK_SERVER_URL = f"http://localhost:{MOCK_SERVER_PORT}"


def wait_for_server(url: str, timeout: int = 10) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=1)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def mock_api_server():
    """
    Start mock API server for the test session.

    Returns the server URL. Server is automatically stopped after tests.
    """
    # Start the mock server as a subprocess
    server_process = subprocess.Popen(
        [sys.executable, "-m", "tests.mocks.mock_api_server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Project root
    )

    # Wait for server to be ready
    if not wait_for_server(MOCK_SERVER_URL):
        server_process.kill()
        pytest.fail("Mock API server failed to start")

    yield MOCK_SERVER_URL

    # Cleanup
    server_process.terminate()
    server_process.wait(timeout=5)


@pytest.fixture
def mock_api_url(mock_api_server):
    """
    Get the mock API server URL.

    Use this fixture in tests that need to make API calls.
    """
    return mock_api_server


@pytest.fixture
def mock_env_vars(mock_api_server, monkeypatch):
    """
    Set environment variables to use mock API server.

    This configures DF_API_URL and DF_API_KEY for the mock server.
    """
    monkeypatch.setenv("DF_API_URL", mock_api_server)
    monkeypatch.setenv("DF_API_KEY", "mock-api-key-for-testing")
    return mock_api_server
