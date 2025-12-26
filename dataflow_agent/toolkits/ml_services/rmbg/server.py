"""
RMBG-2.0 background removal server.

Run with:
    uvicorn dataflow_agent.toolkits.ml_services.rmbg.server:app --port 8004
"""

import os
import io
import base64
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ..common.auth import APIKeyMiddleware
from ..common.schemas import (
    RMBGRequest,
    RMBGResponse,
    HealthResponse,
)


# Lazy loaded model
_rmbg_model = None
_rmbg_transform = None


def _get_rmbg_model():
    """Lazy load RMBG model."""
    global _rmbg_model, _rmbg_transform

    if _rmbg_model is None:
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
            from torchvision import transforms

            model_path = os.environ.get(
                "RMBG_MODEL_PATH",
                "briaai/RMBG-2.0"
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"

            _rmbg_model = AutoModelForImageSegmentation.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            _rmbg_model = _rmbg_model.to(device)
            _rmbg_model.requires_grad_(False)

            _rmbg_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=f"RMBG dependencies not installed: {e}"
            )

    return _rmbg_model, _rmbg_transform


def _remove_background(image_bytes: bytes) -> bytes:
    """Remove background and return PNG with alpha."""
    import torch
    from PIL import Image
    import numpy as np

    model, transform = _get_rmbg_model()
    device = next(model.parameters()).device

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = img.size

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    if isinstance(output, tuple):
        mask = output[0]
    else:
        mask = output

    mask = torch.nn.functional.interpolate(
        mask, size=original_size[::-1], mode="bilinear", align_corners=False
    )
    mask = mask.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    img_array = np.array(img)
    rgba = np.dstack([img_array, mask])
    result = Image.fromarray(rgba, "RGBA")

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    if os.environ.get("RMBG_PRELOAD", "").lower() == "true":
        try:
            _get_rmbg_model()
            app.state.model_loaded = True
        except Exception:
            app.state.model_loaded = False
    else:
        app.state.model_loaded = False

    yield

    global _rmbg_model, _rmbg_transform
    if _rmbg_model is not None:
        del _rmbg_model
        _rmbg_model = None
    _rmbg_transform = None


def create_rmbg_server(api_key: Optional[str] = None) -> FastAPI:
    """Create RMBG FastAPI server."""

    app = FastAPI(
        title="RMBG Background Removal Service",
        description="Background removal with RMBG-2.0",
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
            service="rmbg",
            model_loaded=getattr(app.state, "model_loaded", False),
            gpu_available=gpu,
        )

    @app.post("/remove", response_model=RMBGResponse)
    async def remove_background(request: RMBGRequest):
        """Remove background endpoint."""
        if not request.image:
            raise HTTPException(400, "image field required")

        try:
            from PIL import Image

            image_bytes = request.image.to_bytes()

            img = Image.open(io.BytesIO(image_bytes))
            original_size = [img.width, img.height]

            result_bytes = _remove_background(image_bytes)
            result_b64 = base64.b64encode(result_bytes).decode()

            return RMBGResponse(
                success=True,
                image_base64=result_b64,
                original_size=original_size,
                request_id=request.request_id,
            )

        except Exception as e:
            return RMBGResponse(
                success=False,
                image_base64="",
                error=str(e),
                request_id=request.request_id,
            )

    return app


app = create_rmbg_server()
