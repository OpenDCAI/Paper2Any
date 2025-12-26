"""
Pytest configuration and fixtures for FastAPI integration tests.
"""

import os
import pytest
from datetime import datetime, timezone
from jose import jwt
from httpx import AsyncClient, ASGITransport

# Set test environment variables before importing app
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-jwt-secret-at-least-32-chars-long")
os.environ.setdefault("DAILY_WORKFLOW_LIMIT", "10")


@pytest.fixture
def jwt_secret() -> str:
    """Return the JWT secret used for testing."""
    return os.environ["SUPABASE_JWT_SECRET"]


@pytest.fixture
def test_user_id() -> str:
    """Return a test user ID (UUID format like Supabase uses)."""
    return "12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def valid_jwt_token(jwt_secret: str, test_user_id: str) -> str:
    """
    Generate a valid JWT token for testing.

    This mimics Supabase's JWT structure with 'sub' claim for user_id
    and 'aud' (audience) set to 'authenticated'.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": test_user_id,
        "email": "test@example.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + 3600,  # 1 hour from now
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def expired_jwt_token(jwt_secret: str, test_user_id: str) -> str:
    """Generate an expired JWT token for testing 401 responses."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": test_user_id,
        "email": "test@example.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": int(now.timestamp()) - 7200,  # 2 hours ago
        "exp": int(now.timestamp()) - 3600,  # Expired 1 hour ago
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def invalid_signature_token(test_user_id: str) -> str:
    """Generate a token signed with wrong secret."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": test_user_id,
        "email": "test@example.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + 3600,
    }
    return jwt.encode(payload, "wrong-secret-key-that-is-long-enough", algorithm="HS256")


@pytest.fixture
def auth_headers(valid_jwt_token: str) -> dict:
    """Return headers with valid Authorization Bearer token."""
    return {"Authorization": f"Bearer {valid_jwt_token}"}


@pytest.fixture
async def async_client():
    """Create an async HTTP client for testing FastAPI endpoints.

    Uses main_lite.py to avoid heavy ML dependencies.
    For full integration tests, use the main app directly.
    """
    try:
        # Try lite app first (no heavy ML deps)
        from fastapi_app.main_lite import app
    except ImportError:
        # Fall back to full app if lite not available
        from fastapi_app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client
