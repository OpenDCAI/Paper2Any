"""
MinerU service server - FastAPI application.

Run with:
    uvicorn dataflow_agent.toolkits.ml_services.mineru.server:app --port 8001

Or programmatically:
    app = create_mineru_server(api_key="secret")
    uvicorn.run(app, host="0.0.0.0", port=8001)
"""

import os
import tempfile
import base64
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ..common.auth import APIKeyMiddleware
from ..common.schemas import (
    MinerURequest,
    MinerUResponse,
    MinerUBlock,
    HealthResponse,
)


# Lazy load MinerU to avoid import errors when not installed
_mineru_client = None


def _get_mineru_client():
    """Lazy load MinerU client."""
    global _mineru_client
    if _mineru_client is None:
        try:
            from mineru_vl_utils import MinerUClient
            # Use HTTP client mode if server URL is set
            server_url = os.environ.get("MINERU_SERVER_URL")
            if server_url:
                _mineru_client = MinerUClient(backend="http-client", server_url=server_url)
            else:
                _mineru_client = MinerUClient(backend="local")
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="MinerU not installed. Install with: pip install mineru-vl-utils"
            )
    return _mineru_client


def _parse_image_with_mineru(image_bytes: bytes) -> list:
    """Parse image bytes with MinerU and return blocks."""
    client = _get_mineru_client()

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        temp_path = f.name

    try:
        result = client.two_step_extract(temp_path)
        # Convert to our schema
        blocks = []
        for item in result.get("layout_dets", []):
            blocks.append(MinerUBlock(
                type=item.get("category_name", "unknown"),
                bbox=item.get("poly", [0, 0, 1, 1])[:4],  # Normalize if needed
                text=item.get("text"),
                content=item.get("content"),
            ))
        return blocks
    finally:
        os.unlink(temp_path)


def _parse_pdf_with_mineru(pdf_path: str, output_dir: str) -> dict:
    """Parse PDF with MinerU CLI."""
    import subprocess

    result = subprocess.run(
        ["mineru", "-p", pdf_path, "-o", output_dir, "--source", "modelscope"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"MinerU failed: {result.stderr}"
        )

    # Read output markdown
    md_files = list(Path(output_dir).glob("**/*.md"))
    if md_files:
        markdown = md_files[0].read_text()
    else:
        markdown = ""

    return {"markdown": markdown}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: optionally preload model
    if os.environ.get("MINERU_PRELOAD", "").lower() == "true":
        try:
            _get_mineru_client()
            app.state.model_loaded = True
        except Exception:
            app.state.model_loaded = False
    else:
        app.state.model_loaded = False

    yield

    # Shutdown: cleanup
    global _mineru_client
    _mineru_client = None


def create_mineru_server(api_key: Optional[str] = None) -> FastAPI:
    """
    Create MinerU FastAPI server.

    Args:
        api_key: API key for authentication (or set ML_SERVICE_API_KEY env var)

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="MinerU Service",
        description="PDF and image parsing with MinerU",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add API key middleware
    app.add_middleware(APIKeyMiddleware, api_key=api_key)

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check service health."""
        import torch
        return HealthResponse(
            status="healthy",
            service="mineru",
            model_loaded=getattr(app.state, "model_loaded", False),
            gpu_available=torch.cuda.is_available() if "torch" in dir() else False,
        )

    @app.post("/parse", response_model=MinerUResponse)
    async def parse_image(request: MinerURequest):
        """Parse a single image."""
        if not request.image:
            raise HTTPException(400, "image field required")

        try:
            image_bytes = request.image.to_bytes()
            blocks = _parse_image_with_mineru(image_bytes)

            return MinerUResponse(
                success=True,
                blocks=blocks,
                request_id=request.request_id,
            )
        except Exception as e:
            return MinerUResponse(
                success=False,
                error=str(e),
                request_id=request.request_id,
            )

    @app.post("/parse/batch", response_model=MinerUResponse)
    async def parse_images_batch(request: MinerURequest):
        """Parse multiple images."""
        if not request.images:
            raise HTTPException(400, "images field required")

        all_blocks = []
        for img in request.images:
            try:
                image_bytes = img.to_bytes()
                blocks = _parse_image_with_mineru(image_bytes)
                all_blocks.extend(blocks)
            except Exception as e:
                # Continue on error, log it
                pass

        return MinerUResponse(
            success=True,
            blocks=all_blocks,
            request_id=request.request_id,
        )

    @app.post("/parse/pdf", response_model=MinerUResponse)
    async def parse_pdf(request: MinerURequest):
        """Parse a PDF file."""
        if not request.pdf_path:
            raise HTTPException(400, "pdf_path field required")

        if not Path(request.pdf_path).exists():
            raise HTTPException(404, f"PDF not found: {request.pdf_path}")

        with tempfile.TemporaryDirectory() as output_dir:
            try:
                result = _parse_pdf_with_mineru(request.pdf_path, output_dir)
                return MinerUResponse(
                    success=True,
                    markdown=result.get("markdown"),
                    request_id=request.request_id,
                )
            except Exception as e:
                return MinerUResponse(
                    success=False,
                    error=str(e),
                    request_id=request.request_id,
                )

    return app


# Default app instance
app = create_mineru_server()
