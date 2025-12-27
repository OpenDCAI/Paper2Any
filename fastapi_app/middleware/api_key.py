"""
Simple API Key middleware for workflow endpoints.

This provides basic protection for the backend API.
The frontend sends this key with every workflow request.

Usage (as middleware in main.py):
    from fastapi_app.middleware.api_key import APIKeyMiddleware
    app.add_middleware(APIKeyMiddleware)

Usage (as dependency in router):
    from fastapi_app.middleware.api_key import verify_api_key
    @router.post("/workflow")
    async def workflow(_: None = Depends(verify_api_key)):
        ...
"""

from fastapi import HTTPException, Header, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Hardcoded API key - frontend uses this to call backend
# This is not meant for security against determined attackers,
# just to prevent casual misuse of the API
API_KEY = "df-internal-2024-workflow-key"

# Paths that don't require API key
EXCLUDED_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}

# Path prefixes that don't require API key
EXCLUDED_PREFIXES = (
    "/outputs/",  # Static files
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that verifies API key for /api/* routes.

    Excludes health check, docs, and static file routes.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip excluded paths
        if path in EXCLUDED_PATHS:
            return await call_next(request)

        # Skip excluded prefixes
        if path.startswith(EXCLUDED_PREFIXES):
            return await call_next(request)

        # Only check API key for /api/* and /paper2video/* routes
        if path.startswith("/api/") or path.startswith("/paper2video/"):
            api_key = request.headers.get("X-API-Key")

            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API key required"},
                )

            if api_key != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid API key"},
                )

        return await call_next(request)


async def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"),
) -> None:
    """
    Verify the API key in request header (for use as Depends).

    Raises:
        HTTPException 401 if key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
