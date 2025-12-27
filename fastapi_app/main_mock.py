"""
Mock FastAPI app for frontend development and demos.

Simplified backend - only API key verification, no user auth.
All user-related logic (quota, usage) is handled by frontend.

Usage:
    uvicorn fastapi_app.main_mock:app --port 8000
"""

import asyncio
import random
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Depends, File, Form, UploadFile, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Hardcoded API key - must match frontend
API_KEY = "df-internal-2024-workflow-key"


async def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"),
) -> None:
    """Verify API key. Raises 401 if invalid."""
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


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


# Health check (no API key required)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "mock"}


# ============ Mock Workflow Endpoints ============
# All endpoints require API key verification


@app.post("/api/paper2figure/generate")
async def mock_paper2figure(
    file: UploadFile = File(None),
    text: str = Form(None),
    graph_type: str = Form("model_arch"),
    _: None = Depends(verify_api_key),
):
    """Mock Paper2Figure workflow - returns a mock PPTX file."""
    await asyncio.sleep(random.uniform(1, 3))

    filename = f"mock_{graph_type}.pptx"
    mock_pptx_content = b"PK\x03\x04" + b"\x00" * 100

    return Response(
        content=mock_pptx_content,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/paper2figure/generate_json")
async def mock_paper2figure_json(
    file: UploadFile = File(None),
    text: str = Form(None),
    graph_type: str = Form("tech_route"),
    _: None = Depends(verify_api_key),
):
    """Mock Paper2Figure JSON workflow."""
    await asyncio.sleep(random.uniform(1, 3))

    return {
        "success": True,
        "ppt_filename": "https://via.placeholder.com/800x600.png?text=Mock+PPT",
        "svg_filename": "https://via.placeholder.com/800x600.png?text=Mock+SVG",
        "svg_image_filename": "https://via.placeholder.com/800x600.png?text=Mock+SVG+Preview",
        "all_output_files": [
            "https://via.placeholder.com/800x600.png?text=Mock+Figure+1",
            "https://via.placeholder.com/800x600.png?text=Mock+Figure+2",
        ],
    }


@app.post("/api/paper2ppt/pagecontent_json")
async def mock_paper2ppt_pagecontent(
    file: UploadFile = File(None),
    text: str = Form(None),
    _: None = Depends(verify_api_key),
):
    """Mock Paper2PPT page content extraction."""
    await asyncio.sleep(random.uniform(2, 4))

    mock_pages = [
        {
            "page_num": i,
            "title": f"Mock Slide {i+1}",
            "content": f"This is mock content for slide {i+1}. Lorem ipsum dolor sit amet.",
            "bullet_points": [f"Point {j+1}" for j in range(3)],
        }
        for i in range(5)
    ]

    return {
        "success": True,
        "result_path": f"/tmp/mock_result_{random.randint(1000, 9999)}",
        "pagecontent": mock_pages,
    }


@app.post("/api/paper2ppt/ppt_json")
async def mock_paper2ppt_ppt(
    result_path: str = Form(None),
    pagecontent: str = Form(None),
    _: None = Depends(verify_api_key),
):
    """Mock Paper2PPT generation."""
    await asyncio.sleep(random.uniform(3, 6))

    mock_files = [
        f"https://via.placeholder.com/1920x1080.png?text=PPT+Page+{i}"
        for i in range(5)
    ]

    return {
        "success": True,
        "all_output_files": mock_files,
        "pptx_url": "https://example.com/mock_presentation.pptx",
    }


@app.post("/api/paper2ppt/full_json")
async def mock_paper2ppt_full(
    file: UploadFile = File(None),
    text: str = Form(None),
    _: None = Depends(verify_api_key),
):
    """Mock Paper2PPT full workflow."""
    await asyncio.sleep(random.uniform(3, 8))

    return {
        "success": True,
        "all_output_files": [
            f"https://via.placeholder.com/1920x1080.png?text=PPT+Page+{i}"
            for i in range(5)
        ],
        "pptx_url": "https://example.com/mock_presentation.pptx",
    }


@app.post("/api/pdf2ppt/generate")
async def mock_pdf2ppt(
    pdf_file: UploadFile = File(None),
    file: UploadFile = File(None),
    _: None = Depends(verify_api_key),
):
    """Mock PDF2PPT workflow - returns a mock PPTX file."""
    await asyncio.sleep(random.uniform(2, 6))

    filename = "mock_pdf2ppt.pptx"
    mock_pptx_content = b"PK\x03\x04" + b"\x00" * 100

    return Response(
        content=mock_pptx_content,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/paper2beamer/generate")
async def mock_paper2beamer(
    file: UploadFile = File(None),
    text: str = Form(None),
    _: None = Depends(verify_api_key),
):
    """Mock Paper2Beamer workflow."""
    await asyncio.sleep(random.uniform(3, 7))

    return {
        "success": True,
        "message": "Mock: Generated Beamer slides from paper",
        "pdf_url": "https://example.com/mock_beamer.pdf",
    }


def create_app():
    """Factory function for creating the mock app."""
    return app
