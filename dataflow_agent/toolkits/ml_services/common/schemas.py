"""
Common Pydantic schemas for ML services.

These schemas define the unified API contract for all ML services.
"""

import base64
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    """
    Image input - supports file path or base64 encoded data.

    For remote calls, use base64. For local calls, use file path.
    """

    path: Optional[str] = Field(None, description="Local file path")
    base64_data: Optional[str] = Field(None, description="Base64 encoded image")
    filename: Optional[str] = Field(None, description="Original filename for reference")

    def to_bytes(self) -> bytes:
        """Convert to bytes (reads file or decodes base64)."""
        if self.base64_data:
            return base64.b64decode(self.base64_data)
        if self.path:
            with open(self.path, "rb") as f:
                return f.read()
        raise ValueError("Either path or base64_data must be provided")

    @classmethod
    def from_file(cls, path: str) -> "ImageInput":
        """Create from file path."""
        return cls(path=path)

    @classmethod
    def from_bytes(cls, data: bytes, filename: str = "image") -> "ImageInput":
        """Create from bytes (encodes to base64)."""
        return cls(base64_data=base64.b64encode(data).decode(), filename=filename)


class BaseMLRequest(BaseModel):
    """Base request schema for ML services."""

    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")


class BaseMLResponse(BaseModel):
    """Base response schema for ML services."""

    success: bool = True
    request_id: Optional[str] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema."""

    success: bool = False
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str
    version: str = "1.0.0"
    model_loaded: bool = False
    gpu_available: bool = False


# ============ MinerU Schemas ============


class MinerURequest(BaseMLRequest):
    """Request for MinerU PDF/image parsing."""

    image: Optional[ImageInput] = Field(None, description="Single image input")
    images: Optional[List[ImageInput]] = Field(None, description="Batch image inputs")
    pdf_path: Optional[str] = Field(None, description="PDF file path (server-local)")
    output_format: str = Field("blocks", description="Output format: blocks, markdown")


class MinerUBlock(BaseModel):
    """A single parsed block from MinerU."""

    type: str = Field(..., description="Block type: text, image, table, footer, etc.")
    bbox: List[float] = Field(..., description="Normalized bbox [x1, y1, x2, y2] in [0,1]")
    text: Optional[str] = Field(None, description="Text content if applicable")
    content: Optional[str] = Field(None, description="Alternative content field")


class MinerUResponse(BaseMLResponse):
    """Response from MinerU parsing."""

    blocks: Optional[List[MinerUBlock]] = None
    markdown: Optional[str] = None
    page_count: Optional[int] = None


# ============ SAM/YOLO Schemas ============


class SAMRequest(BaseMLRequest):
    """Request for SAM segmentation."""

    image: ImageInput
    mode: str = Field("auto", description="Segmentation mode: auto, box, point")
    # For box mode
    boxes: Optional[List[List[float]]] = Field(None, description="Bounding boxes [[x1,y1,x2,y2],...]")
    # For point mode
    points: Optional[List[List[float]]] = Field(None, description="Points [[x,y],...]")
    point_labels: Optional[List[int]] = Field(None, description="Point labels (1=foreground, 0=background)")
    # Post-processing
    min_area: Optional[int] = Field(None, description="Minimum mask area in pixels")
    min_score: Optional[float] = Field(None, description="Minimum confidence score")
    nms_threshold: Optional[float] = Field(None, description="NMS IoU threshold")
    top_k: Optional[int] = Field(None, description="Return top K results")


class SegmentationItem(BaseModel):
    """A single segmentation result."""

    mask_base64: str = Field(..., description="Base64 encoded mask (RLE or PNG)")
    bbox: List[float] = Field(..., description="Normalized bbox [x1, y1, x2, y2]")
    score: Optional[float] = Field(None, description="Confidence score")
    area: Optional[int] = Field(None, description="Mask area in pixels")
    label: Optional[str] = Field(None, description="Class label (YOLO only)")


class SAMResponse(BaseMLResponse):
    """Response from SAM segmentation."""

    items: List[SegmentationItem] = []
    image_size: Optional[List[int]] = Field(None, description="[width, height]")


class YOLORequest(BaseMLRequest):
    """Request for YOLO segmentation."""

    image: ImageInput
    weights: str = Field("yolov8n-seg.pt", description="YOLO weights file")
    conf_threshold: float = Field(0.25, description="Confidence threshold")


class YOLOResponse(BaseMLResponse):
    """Response from YOLO segmentation."""

    items: List[SegmentationItem] = []
    image_size: Optional[List[int]] = Field(None, description="[width, height]")


# ============ OCR Schemas ============


class OCRRequest(BaseMLRequest):
    """Request for PaddleOCR."""

    image: ImageInput
    lang: str = Field("ch", description="Language: ch, en, etc.")
    drop_score: int = Field(30, description="Confidence threshold 0-100")
    with_layout: bool = Field(True, description="Include layout analysis")


class OCRLine(BaseModel):
    """A single OCR text line."""

    text: str
    bbox: List[float] = Field(..., description="Bbox [x1, y1, x2, y2] in pixels")
    confidence: float = Field(..., description="Confidence 0-100")


class OCRResponse(BaseMLResponse):
    """Response from OCR."""

    lines: List[OCRLine] = []
    image_size: Optional[List[int]] = Field(None, description="[width, height]")
    body_line_height: Optional[float] = Field(None, description="Median line height")
    background_color: Optional[List[int]] = Field(None, description="RGB background color")


# ============ RMBG Schemas ============


class RMBGRequest(BaseMLRequest):
    """Request for background removal."""

    image: ImageInput


class RMBGResponse(BaseMLResponse):
    """Response from background removal."""

    image_base64: str = Field(..., description="Base64 encoded PNG with alpha")
    original_size: Optional[List[int]] = Field(None, description="[width, height]")
