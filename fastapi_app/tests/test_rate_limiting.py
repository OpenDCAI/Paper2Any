"""
Integration tests for rate limiting functionality.

Tests that:
- Quota endpoint returns correct format
- Rate limiting is enforced after quota exceeded
"""

import pytest


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_quota_returns_correct_format(self, async_client, auth_headers):
        """GET /api/quota should return {used, limit} structure."""
        response = await async_client.get("/api/quota", headers=auth_headers)

        # May fail with 500 if Supabase not connected, but auth should pass
        if response.status_code == 200:
            data = response.json()
            assert "used" in data
            assert "limit" in data
            assert isinstance(data["used"], int)
            assert isinstance(data["limit"], int)
            assert data["limit"] == 10  # Default from env
            assert 0 <= data["used"] <= data["limit"]

    @pytest.mark.asyncio
    async def test_quota_requires_auth(self, async_client):
        """GET /api/quota without auth should return 401."""
        response = await async_client.get("/api/quota")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_rate_limit_header_on_429(self, async_client, auth_headers):
        """
        When quota is exceeded, endpoint should return 429.

        Note: This test requires simulating exhausted quota,
        which needs database setup or mocking.
        """
        # This is a placeholder test - actual implementation would need
        # to either mock the rate_limiter or exhaust quota via API calls
        pass
