"""
Simple API key authentication for ML services.

Usage:
    # Server side - add middleware
    app.add_middleware(APIKeyMiddleware, api_key="your-secret-key")

    # Or use dependency injection
    @app.post("/predict")
    async def predict(api_key: str = Depends(verify_api_key)):
        ...

    # Client side - add header
    headers = {"X-API-Key": "your-secret-key"}
"""

import os
from typing import Optional

from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# Environment variable for API key
ML_SERVICE_API_KEY_ENV = "ML_SERVICE_API_KEY"

# Header name for API key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key_from_env() -> Optional[str]:
    """Get API key from environment variable."""
    return os.environ.get(ML_SERVICE_API_KEY_ENV)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    """
    FastAPI dependency to verify API key.

    Raises HTTPException 401 if key is missing or invalid.
    """
    expected_key = get_api_key_from_env()

    # If no key configured, allow all requests (dev mode)
    if not expected_key:
        return "dev-mode"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return api_key


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API key authentication.

    Skips auth for /health and /docs endpoints.
    """

    def __init__(self, app, api_key: Optional[str] = None):
        super().__init__(app)
        self.api_key = api_key or get_api_key_from_env()

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health check and docs
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)

        # If no key configured, allow all (dev mode)
        if not self.api_key:
            return await call_next(request)

        # Check API key
        provided_key = request.headers.get("X-API-Key")
        if not provided_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key. Provide X-API-Key header."},
            )

        if provided_key != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key."},
            )

        return await call_next(request)
