"""
Base client for ML services.

Provides common HTTP client functionality with:
- API key authentication
- Retry logic
- Timeout handling
- Health checks
"""

import httpx
import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel

from .schemas import BaseMLResponse, ErrorResponse, HealthResponse

T = TypeVar("T", bound=BaseMLResponse)


class MLServiceClient:
    """
    Base client for ML services.

    Usage:
        client = MLServiceClient(
            base_url="http://localhost:8001",
            api_key="your-secret-key"
        )
        response = await client.post("/predict", request_data, ResponseModel)
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers including API key if set."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> HealthResponse:
        """Check service health."""
        client = await self._get_client()
        try:
            response = await client.get("/health")
            response.raise_for_status()
            return HealthResponse(**response.json())
        except Exception as e:
            return HealthResponse(
                status="unhealthy",
                service="unknown",
                model_loaded=False,
            )

    async def is_healthy(self) -> bool:
        """Quick health check returning boolean."""
        health = await self.health_check()
        return health.status == "healthy"

    async def post(
        self,
        endpoint: str,
        data: Dict[str, Any],
        response_model: Type[T],
        retries: Optional[int] = None,
    ) -> T:
        """
        POST request with retry logic.

        Args:
            endpoint: API endpoint (e.g., "/predict")
            data: Request data dict
            response_model: Pydantic model for response
            retries: Override max retries

        Returns:
            Parsed response

        Raises:
            httpx.HTTPError: On request failure after retries
        """
        client = await self._get_client()
        max_retries = retries if retries is not None else self.max_retries
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await client.post(endpoint, json=data)
                response.raise_for_status()
                return response_model(**response.json())

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    error_data = e.response.json() if e.response.content else {}
                    raise ValueError(error_data.get("detail", str(e)))
                last_error = e

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e

            # Exponential backoff
            if attempt < max_retries:
                await asyncio.sleep(2**attempt)

        raise last_error or Exception("Request failed")

    async def post_file(
        self,
        endpoint: str,
        file_path: str,
        field_name: str = "file",
        extra_data: Optional[Dict[str, Any]] = None,
        response_model: Type[T] = None,
    ) -> T:
        """
        POST with file upload (multipart/form-data).

        Args:
            endpoint: API endpoint
            file_path: Path to file
            field_name: Form field name for file
            extra_data: Additional form fields
            response_model: Pydantic model for response
        """
        client = await self._get_client()

        with open(file_path, "rb") as f:
            files = {field_name: f}
            data = extra_data or {}
            response = await client.post(endpoint, files=files, data=data)
            response.raise_for_status()

            if response_model:
                return response_model(**response.json())
            return response.json()

    # Sync convenience methods

    def post_sync(
        self,
        endpoint: str,
        data: Dict[str, Any],
        response_model: Type[T],
    ) -> T:
        """Synchronous POST request."""
        return asyncio.get_event_loop().run_until_complete(
            self.post(endpoint, data, response_model)
        )

    def health_check_sync(self) -> HealthResponse:
        """Synchronous health check."""
        return asyncio.get_event_loop().run_until_complete(self.health_check())


class MLServiceClientPool:
    """
    Client pool for load balancing across multiple service instances.

    Usage:
        pool = MLServiceClientPool(
            urls=["http://localhost:8001", "http://localhost:8002"],
            api_key="secret"
        )
        client = await pool.get_client()  # Round-robin
    """

    def __init__(
        self,
        urls: List[str],
        api_key: Optional[str] = None,
        **client_kwargs,
    ):
        self.clients = [
            MLServiceClient(url, api_key=api_key, **client_kwargs)
            for url in urls
        ]
        self._index = 0
        self._lock = asyncio.Lock()

    async def get_client(self) -> MLServiceClient:
        """Get next client (round-robin)."""
        async with self._lock:
            client = self.clients[self._index]
            self._index = (self._index + 1) % len(self.clients)
            return client

    async def get_healthy_client(self) -> Optional[MLServiceClient]:
        """Get a healthy client, or None if all unhealthy."""
        for client in self.clients:
            if await client.is_healthy():
                return client
        return None

    async def close_all(self):
        """Close all clients."""
        for client in self.clients:
            await client.close()
