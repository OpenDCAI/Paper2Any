"""
Common utilities for ML services.

Provides:
- API key authentication middleware
- Unified request/response schemas
- Client base class
"""

from .auth import verify_api_key, api_key_header, APIKeyMiddleware
from .schemas import (
    BaseMLRequest,
    BaseMLResponse,
    ImageInput,
    ErrorResponse,
    HealthResponse,
)
from .client import MLServiceClient, MLServiceClientPool

__all__ = [
    "verify_api_key",
    "api_key_header",
    "APIKeyMiddleware",
    "BaseMLRequest",
    "BaseMLResponse",
    "ImageInput",
    "ErrorResponse",
    "HealthResponse",
    "MLServiceClient",
    "MLServiceClientPool",
]
