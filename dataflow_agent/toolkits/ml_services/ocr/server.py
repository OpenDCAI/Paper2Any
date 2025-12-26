"""
PaddleOCR service server.

Run with:
    uvicorn dataflow_agent.toolkits.ml_services.ocr.server:app --port 8003
"""

import os
import io
import tempfile
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from ..common.auth import APIKeyMiddleware
from ..common.schemas import (
    OCRRequest,
    OCRResponse,
    OCRLine,
    HealthResponse,
)


# Lazy loaded OCR model
_ocr_model = None


def _get_ocr_model(lang: str = "ch"):
    """Lazy load PaddleOCR model."""
    global _ocr_model
    if _ocr_model is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                show_log=False,
            )
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="PaddleOCR not installed. Install with: pip install paddleocr"
            )
    return _ocr_model


def _run_ocr(image_path: str, drop_score: int = 30) -> List[OCRLine]:
    """Run OCR on image."""
    import cv2

    ocr = _get_ocr_model()
    img = cv2.imread(image_path)

    result = ocr.ocr(img, cls=True)

    lines = []
    if result and result[0]:
        for item in result[0]:
            bbox_points = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text, confidence = item[1]

            # Convert to confidence 0-100
            conf_100 = int(confidence * 100)
            if conf_100 < drop_score:
                continue

            # Convert polygon to bbox
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            lines.append(OCRLine(
                text=text,
                bbox=bbox,
                confidence=conf_100,
            ))

    return lines


def _analyze_layout(lines: List[OCRLine], img_shape: tuple) -> dict:
    """Analyze text layout."""
    if not lines:
        return {"body_line_height": None, "background_color": None}

    # Calculate median line height
    heights = [line.bbox[3] - line.bbox[1] for line in lines]
    body_line_height = float(np.median(heights)) if heights else None

    return {
        "body_line_height": body_line_height,
        "background_color": None,  # Could implement background detection
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    if os.environ.get("OCR_PRELOAD", "").lower() == "true":
        try:
            _get_ocr_model()
            app.state.model_loaded = True
        except Exception:
            app.state.model_loaded = False
    else:
        app.state.model_loaded = False

    yield

    global _ocr_model
    _ocr_model = None


def create_ocr_server(api_key: Optional[str] = None) -> FastAPI:
    """Create OCR FastAPI server."""

    app = FastAPI(
        title="PaddleOCR Service",
        description="OCR text recognition with PaddleOCR",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(APIKeyMiddleware, api_key=api_key)

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check service health."""
        try:
            import paddle
            gpu = paddle.device.is_compiled_with_cuda()
        except ImportError:
            gpu = False

        return HealthResponse(
            status="healthy",
            service="ocr",
            model_loaded=getattr(app.state, "model_loaded", False),
            gpu_available=gpu,
        )

    @app.post("/ocr", response_model=OCRResponse)
    async def ocr(request: OCRRequest):
        """OCR endpoint."""
        if not request.image:
            raise HTTPException(400, "image field required")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(request.image.to_bytes())
            temp_path = f.name

        try:
            import cv2
            img = cv2.imread(temp_path)
            h, w = img.shape[:2]
            image_size = [w, h]

            lines = _run_ocr(temp_path, drop_score=request.drop_score)

            layout = {}
            if request.with_layout:
                layout = _analyze_layout(lines, img.shape)

            return OCRResponse(
                success=True,
                lines=lines,
                image_size=image_size,
                body_line_height=layout.get("body_line_height"),
                background_color=layout.get("background_color"),
                request_id=request.request_id,
            )

        except Exception as e:
            return OCRResponse(
                success=False,
                error=str(e),
                lines=[],
                request_id=request.request_id,
            )
        finally:
            os.unlink(temp_path)

    return app


# Default app instance
app = create_ocr_server()
