"""
Mock ML services for testing and development.

Provides mock implementations of all ML services with realistic responses.

Usage:
    # Run all mock services on a single port
    uvicorn dataflow_agent.toolkits.ml_services.mock_server:app --port 8000

    # Or run individual mock services
    python -m dataflow_agent.toolkits.ml_services.mock_server --service mineru --port 8001
"""

import base64
import io
import os
import random
import string
from typing import Optional

from fastapi import FastAPI, HTTPException

from .common.auth import APIKeyMiddleware
from .common.schemas import (
    MinerURequest,
    MinerUResponse,
    MinerUBlock,
    SAMRequest,
    SAMResponse,
    YOLORequest,
    YOLOResponse,
    SegmentationItem,
    OCRRequest,
    OCRResponse,
    OCRLine,
    RMBGRequest,
    RMBGResponse,
    HealthResponse,
)


def _create_mock_image_png(width: int = 100, height: int = 100) -> bytes:
    """Create a simple mock PNG image."""
    try:
        from PIL import Image
        import numpy as np

        # Create a simple gradient image with alpha
        arr = np.zeros((height, width, 4), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, width).astype(np.uint8)  # R gradient
        arr[:, :, 1] = np.linspace(0, 255, height).reshape(-1, 1).astype(np.uint8)  # G gradient
        arr[:, :, 2] = 128  # B constant
        arr[:, :, 3] = 255  # Full alpha

        img = Image.fromarray(arr, "RGBA")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        # Return minimal valid PNG if PIL not available
        return b'\x89PNG\r\n\x1a\n' + b'\x00' * 100


def _create_mock_mask(width: int = 100, height: int = 100) -> str:
    """Create a mock compressed mask."""
    import zlib
    import numpy as np

    # Create a circular mask
    y, x = np.ogrid[:height, :width]
    cx, cy = width // 2, height // 2
    mask = ((x - cx) ** 2 + (y - cy) ** 2 < (min(width, height) // 3) ** 2).astype(np.uint8)

    packed = np.packbits(mask)
    compressed = zlib.compress(packed.tobytes())
    return base64.b64encode(compressed).decode()


def create_mock_server(api_key: Optional[str] = None) -> FastAPI:
    """
    Create a mock ML services server.

    This server provides mock implementations for all ML services,
    returning realistic but fake data for testing and development.
    """

    app = FastAPI(
        title="Mock ML Services",
        description="Mock implementations of MinerU, SAM, OCR, RMBG for testing",
        version="1.0.0",
    )

    app.add_middleware(APIKeyMiddleware, api_key=api_key)

    # ============ Health Endpoints ============

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            service="mock-ml-services",
            model_loaded=True,
            gpu_available=False,
        )

    # ============ MinerU Mock Endpoints ============

    @app.post("/parse", response_model=MinerUResponse)
    async def mock_mineru_parse(request: MinerURequest):
        """Mock MinerU single image parsing."""
        # Generate mock blocks
        blocks = [
            MinerUBlock(
                type="title",
                bbox=[0.1, 0.05, 0.9, 0.1],
                text="Mock Document Title",
            ),
            MinerUBlock(
                type="text",
                bbox=[0.1, 0.15, 0.9, 0.4],
                text="This is mock paragraph text generated for testing. "
                     "It simulates the output of MinerU document parsing.",
            ),
            MinerUBlock(
                type="image",
                bbox=[0.2, 0.45, 0.8, 0.75],
                text=None,
            ),
            MinerUBlock(
                type="footer",
                bbox=[0.1, 0.9, 0.9, 0.95],
                text="Page 1",
            ),
        ]

        return MinerUResponse(
            success=True,
            blocks=blocks,
            request_id=request.request_id,
        )

    @app.post("/parse/batch", response_model=MinerUResponse)
    async def mock_mineru_parse_batch(request: MinerURequest):
        """Mock MinerU batch parsing."""
        num_images = len(request.images) if request.images else 1
        blocks = []

        for i in range(num_images):
            blocks.extend([
                MinerUBlock(
                    type="text",
                    bbox=[0.1, 0.1, 0.9, 0.3],
                    text=f"Mock text from image {i + 1}",
                ),
            ])

        return MinerUResponse(
            success=True,
            blocks=blocks,
            page_count=num_images,
            request_id=request.request_id,
        )

    @app.post("/parse/pdf", response_model=MinerUResponse)
    async def mock_mineru_parse_pdf(request: MinerURequest):
        """Mock MinerU PDF parsing."""
        markdown = """# Mock PDF Document

## Introduction

This is a mock markdown output from MinerU PDF parsing.

## Section 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

## Section 2

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

## Conclusion

Mock conclusion text.
"""
        return MinerUResponse(
            success=True,
            markdown=markdown,
            page_count=3,
            request_id=request.request_id,
        )

    # ============ SAM Mock Endpoints ============

    @app.post("/segment", response_model=SAMResponse)
    async def mock_sam_segment(request: SAMRequest):
        """Mock SAM segmentation."""
        num_items = random.randint(3, 8)
        items = []

        for i in range(num_items):
            # Random bbox within image
            x1 = random.uniform(0, 0.5)
            y1 = random.uniform(0, 0.5)
            x2 = x1 + random.uniform(0.1, 0.4)
            y2 = y1 + random.uniform(0.1, 0.4)

            items.append(SegmentationItem(
                mask_base64=_create_mock_mask(100, 100),
                bbox=[x1, y1, min(x2, 1.0), min(y2, 1.0)],
                score=random.uniform(0.7, 0.99),
                area=random.randint(1000, 50000),
            ))

        # Apply filters if specified
        if request.min_area:
            items = [it for it in items if (it.area or 0) >= request.min_area]
        if request.min_score:
            items = [it for it in items if (it.score or 0) >= request.min_score]
        if request.top_k:
            items = sorted(items, key=lambda x: x.area or 0, reverse=True)[:request.top_k]

        return SAMResponse(
            success=True,
            items=items,
            image_size=[800, 600],
            request_id=request.request_id,
        )

    @app.post("/segment/yolo", response_model=YOLOResponse)
    async def mock_yolo_segment(request: YOLORequest):
        """Mock YOLO segmentation."""
        labels = ["person", "car", "dog", "cat", "chair", "table", "laptop", "phone"]
        num_items = random.randint(2, 5)
        items = []

        for i in range(num_items):
            x1 = random.uniform(0, 0.6)
            y1 = random.uniform(0, 0.6)
            x2 = x1 + random.uniform(0.15, 0.35)
            y2 = y1 + random.uniform(0.15, 0.35)

            items.append(SegmentationItem(
                mask_base64=_create_mock_mask(100, 100),
                bbox=[x1, y1, min(x2, 1.0), min(y2, 1.0)],
                score=random.uniform(0.5, 0.95),
                area=random.randint(5000, 100000),
                label=random.choice(labels),
            ))

        return YOLOResponse(
            success=True,
            items=items,
            image_size=[800, 600],
            request_id=request.request_id,
        )

    # ============ OCR Mock Endpoints ============

    @app.post("/ocr", response_model=OCRResponse)
    async def mock_ocr(request: OCRRequest):
        """Mock PaddleOCR."""
        mock_texts = [
            "Welcome to DataFlow",
            "Machine Learning Pipeline",
            "Data Processing",
            "2024 Report",
            "Figure 1: System Architecture",
            "Table 1: Results",
        ]

        num_lines = random.randint(3, 8)
        lines = []
        y_pos = 50

        for i in range(num_lines):
            text = random.choice(mock_texts)
            height = random.randint(20, 40)
            width = len(text) * 12

            lines.append(OCRLine(
                text=text,
                bbox=[50, y_pos, 50 + width, y_pos + height],
                confidence=random.randint(85, 99),
            ))

            y_pos += height + random.randint(10, 30)

        return OCRResponse(
            success=True,
            lines=lines,
            image_size=[800, 600],
            body_line_height=28.0,
            background_color=[255, 255, 255],
            request_id=request.request_id,
        )

    # ============ RMBG Mock Endpoints ============

    @app.post("/remove", response_model=RMBGResponse)
    async def mock_rmbg(request: RMBGRequest):
        """Mock background removal."""
        # Create a mock RGBA PNG
        mock_png = _create_mock_image_png(400, 300)
        mock_b64 = base64.b64encode(mock_png).decode()

        return RMBGResponse(
            success=True,
            image_base64=mock_b64,
            original_size=[400, 300],
            request_id=request.request_id,
        )

    return app


# Default mock app instance
app = create_mock_server()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run mock ML services")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--api-key", type=str, default=None, help="API key for auth")
    args = parser.parse_args()

    if args.api_key:
        os.environ["ML_SERVICE_API_KEY"] = args.api_key

    uvicorn.run(app, host="0.0.0.0", port=args.port)
