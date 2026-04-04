"""Image2DrawIO toolkit utilities."""

from .utils import (
    classify_shape,
    extract_text_color,
    mask_to_bbox,
    normalize_mask,
    sample_fill_stroke,
    save_masked_rgba,
    bbox_iou_px,
)
from .metric_evaluator import evaluate as metric_evaluate
from .refinement_processor import refine as refinement_refine

__all__ = [
    "classify_shape",
    "extract_text_color",
    "mask_to_bbox",
    "normalize_mask",
    "sample_fill_stroke",
    "save_masked_rgba",
    "bbox_iou_px",
    "metric_evaluate",
    "refinement_refine",
]
