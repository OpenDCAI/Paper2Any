"""
SAM/YOLO segmentation service server.

Run with:
    uvicorn dataflow_agent.toolkits.ml_services.sam.server:app --port 8002

Or programmatically:
    app = create_sam_server(api_key="secret")
"""

import os
import io
import base64
import tempfile
import zlib
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from ..common.auth import APIKeyMiddleware
from ..common.schemas import (
    SAMRequest,
    SAMResponse,
    YOLORequest,
    YOLOResponse,
    SegmentationItem,
    HealthResponse,
)


# Lazy loaded models
_sam_model = None
_yolo_model = None


def _encode_mask(mask: np.ndarray) -> str:
    """Encode boolean mask to compressed base64."""
    # Pack bits and compress
    packed = np.packbits(mask.astype(np.uint8))
    compressed = zlib.compress(packed.tobytes())
    return base64.b64encode(compressed).decode()


def _get_sam_model(checkpoint: str = "sam_b.pt"):
    """Lazy load SAM model."""
    global _sam_model
    if _sam_model is None:
        try:
            from ultralytics import SAM as UltralyticsSAM
            device = "cuda" if os.environ.get("SAM_DEVICE", "cuda") == "cuda" else "cpu"
            _sam_model = UltralyticsSAM(checkpoint)
            _sam_model.to(device)
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="SAM not installed. Install with: pip install ultralytics"
            )
    return _sam_model


def _get_yolo_model(weights: str = "yolov8n-seg.pt"):
    """Lazy load YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            device = "cuda" if os.environ.get("YOLO_DEVICE", "cuda") == "cuda" else "cpu"
            _yolo_model = YOLO(weights)
            _yolo_model.to(device)
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="YOLO not installed. Install with: pip install ultralytics"
            )
    return _yolo_model


def _run_sam_auto(image_path: str, min_area: int = None, min_score: float = None,
                   nms_threshold: float = None, top_k: int = None) -> List[SegmentationItem]:
    """Run SAM auto segmentation."""
    from PIL import Image

    model = _get_sam_model()
    img = Image.open(image_path)
    w, h = img.size

    results = model(image_path, device=model.device)

    items = []
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None

        for i, mask in enumerate(masks):
            # Calculate area
            area = int(mask.sum())
            if min_area and area < min_area:
                continue

            # Encode mask
            mask_b64 = _encode_mask(mask)

            # Normalize bbox
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = boxes[i]
                bbox = [x1/w, y1/h, x2/w, y2/h]
            else:
                bbox = [0, 0, 1, 1]

            items.append(SegmentationItem(
                mask_base64=mask_b64,
                bbox=bbox,
                area=area,
            ))

    # Apply top_k
    if top_k and len(items) > top_k:
        items = sorted(items, key=lambda x: x.area or 0, reverse=True)[:top_k]

    return items


def _run_yolo_seg(image_path: str, weights: str, conf_threshold: float) -> List[SegmentationItem]:
    """Run YOLO instance segmentation."""
    from PIL import Image

    model = _get_yolo_model(weights)
    img = Image.open(image_path)
    w, h = img.size

    results = model(image_path, conf=conf_threshold)

    items = []
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        names = results[0].names

        for i, mask in enumerate(masks):
            mask_b64 = _encode_mask(mask)

            x1, y1, x2, y2 = boxes[i]
            bbox = [x1/w, y1/h, x2/w, y2/h]

            items.append(SegmentationItem(
                mask_base64=mask_b64,
                bbox=bbox,
                score=float(scores[i]),
                area=int(mask.sum()),
                label=names[int(classes[i])],
            ))

    return items


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    if os.environ.get("SAM_PRELOAD", "").lower() == "true":
        try:
            _get_sam_model()
            app.state.sam_loaded = True
        except Exception:
            app.state.sam_loaded = False
    else:
        app.state.sam_loaded = False

    yield

    # Cleanup
    global _sam_model, _yolo_model
    if _sam_model is not None:
        del _sam_model
        _sam_model = None
    if _yolo_model is not None:
        del _yolo_model
        _yolo_model = None


def create_sam_server(api_key: Optional[str] = None) -> FastAPI:
    """Create SAM/YOLO FastAPI server."""

    app = FastAPI(
        title="SAM/YOLO Segmentation Service",
        description="Image segmentation with SAM and YOLO",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(APIKeyMiddleware, api_key=api_key)

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check service health."""
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False

        return HealthResponse(
            status="healthy",
            service="sam",
            model_loaded=getattr(app.state, "sam_loaded", False),
            gpu_available=gpu,
        )

    @app.post("/segment", response_model=SAMResponse)
    async def segment(request: SAMRequest):
        """SAM segmentation endpoint."""
        if not request.image:
            raise HTTPException(400, "image field required")

        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(request.image.to_bytes())
            temp_path = f.name

        try:
            from PIL import Image
            img = Image.open(temp_path)
            image_size = [img.width, img.height]

            if request.mode == "auto":
                items = _run_sam_auto(
                    temp_path,
                    min_area=request.min_area,
                    min_score=request.min_score,
                    nms_threshold=request.nms_threshold,
                    top_k=request.top_k,
                )
            else:
                # TODO: Implement box and point modes
                raise HTTPException(501, f"Mode {request.mode} not implemented yet")

            return SAMResponse(
                success=True,
                items=items,
                image_size=image_size,
                request_id=request.request_id,
            )

        except Exception as e:
            return SAMResponse(
                success=False,
                error=str(e),
                request_id=request.request_id,
            )
        finally:
            os.unlink(temp_path)

    @app.post("/segment/yolo", response_model=YOLOResponse)
    async def segment_yolo(request: YOLORequest):
        """YOLO segmentation endpoint."""
        if not request.image:
            raise HTTPException(400, "image field required")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(request.image.to_bytes())
            temp_path = f.name

        try:
            from PIL import Image
            img = Image.open(temp_path)
            image_size = [img.width, img.height]

            items = _run_yolo_seg(temp_path, request.weights, request.conf_threshold)

            return YOLOResponse(
                success=True,
                items=items,
                image_size=image_size,
                request_id=request.request_id,
            )

        except Exception as e:
            return YOLOResponse(
                success=False,
                error=str(e),
                request_id=request.request_id,
            )
        finally:
            os.unlink(temp_path)

    return app


# Default app instance
app = create_sam_server()
