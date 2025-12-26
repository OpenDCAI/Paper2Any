"""
Mock FastAPI app for frontend development and demos.

Includes auth, rate limiting, and mock workflow endpoints.
No ML dependencies required.

Usage:
    uvicorn fastapi_app.main_mock:app --port 8000
"""

import asyncio
import random
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi_app.auth import require_auth_and_quota, CurrentUser
from fastapi_app.services.rate_limiter import rate_limiter

# Create app
app = FastAPI(
    title="DataFlow-Agent API (Mock)",
    description="Mock API for frontend development - no ML dependencies",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "mock"}


# Import lightweight routers
import importlib.util
import sys
from pathlib import Path


def _import_router_directly(name: str):
    """Import a router module directly without going through the package."""
    router_path = Path(__file__).parent / "routers" / f"{name}.py"
    if not router_path.exists():
        raise ImportError(f"Router not found: {router_path}")

    spec = importlib.util.spec_from_file_location(
        f"fastapi_app.routers.{name}", router_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"fastapi_app.routers.{name}"] = module
    spec.loader.exec_module(module)
    return module


# Import real auth/user routers
user_module = _import_router_directly("user")
auth_module = _import_router_directly("auth")

app.include_router(user_module.router, prefix="/api", tags=["user"])
app.include_router(auth_module.router, prefix="/api", tags=["auth"])


# ============ Mock Workflow Endpoints ============


class MockWorkflowResponse(BaseModel):
    """Generic mock workflow response."""

    status: str = "completed"
    message: str
    output_url: Optional[str] = None
    processing_time: float


# Mock Paper2Figure
@app.post("/api/paper2figure/generate")
async def mock_paper2figure(
    file: UploadFile = File(None),
    text: str = Form(None),
    figure_type: str = Form("architecture"),
    user: CurrentUser = Depends(require_auth_and_quota),
):
    """Mock Paper2Figure workflow - returns fake result after delay."""
    # Simulate processing time
    await asyncio.sleep(random.uniform(2, 5))

    # Record usage
    await rate_limiter.record_usage(user.user_id, "paper2figure")

    return MockWorkflowResponse(
        status="completed",
        message=f"Mock: Generated {figure_type} diagram from paper",
        output_url="https://via.placeholder.com/800x600.png?text=Mock+Figure",
        processing_time=random.uniform(2, 5),
    )


# Mock Paper2PPT
@app.post("/api/paper2ppt/full_json")
async def mock_paper2ppt(
    file: UploadFile = File(None),
    text: str = Form(None),
    user: CurrentUser = Depends(require_auth_and_quota),
):
    """Mock Paper2PPT workflow."""
    await asyncio.sleep(random.uniform(3, 8))

    await rate_limiter.record_usage(user.user_id, "paper2ppt")

    return MockWorkflowResponse(
        status="completed",
        message="Mock: Generated PPT from paper",
        output_url="https://example.com/mock_presentation.pptx",
        processing_time=random.uniform(3, 8),
    )


# Mock PDF2PPT
@app.post("/api/pdf2ppt/generate")
async def mock_pdf2ppt(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_auth_and_quota),
):
    """Mock PDF2PPT workflow."""
    await asyncio.sleep(random.uniform(2, 6))

    await rate_limiter.record_usage(user.user_id, "pdf2ppt")

    return MockWorkflowResponse(
        status="completed",
        message="Mock: Converted PDF to editable PPT",
        output_url="https://example.com/mock_converted.pptx",
        processing_time=random.uniform(2, 6),
    )


# Mock Paper2Beamer
@app.post("/api/paper2beamer/generate")
async def mock_paper2beamer(
    file: UploadFile = File(None),
    text: str = Form(None),
    user: CurrentUser = Depends(require_auth_and_quota),
):
    """Mock Paper2Beamer workflow."""
    await asyncio.sleep(random.uniform(3, 7))

    await rate_limiter.record_usage(user.user_id, "paper2beamer")

    return MockWorkflowResponse(
        status="completed",
        message="Mock: Generated Beamer slides from paper",
        output_url="https://example.com/mock_beamer.pdf",
        processing_time=random.uniform(3, 7),
    )


def create_app():
    """Factory function for creating the mock app."""
    return app
