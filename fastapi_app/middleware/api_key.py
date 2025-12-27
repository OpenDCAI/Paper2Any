"""
Simple API Key middleware for workflow endpoints.

This provides basic protection for the backend API.
The frontend sends this key with every workflow request.

Usage:
    from fastapi_app.middleware.api_key import verify_api_key

    @router.post("/workflow")
    async def workflow(_: None = Depends(verify_api_key)):
        ...
"""

from fastapi import HTTPException, Header, status

# Hardcoded API key - frontend uses this to call backend
# This is not meant for security against determined attackers,
# just to prevent casual misuse of the API
API_KEY = "df-internal-2024-workflow-key"


async def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"),
) -> None:
    """
    Verify the API key in request header.

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
