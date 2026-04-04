"""
refinement_processor.py — Fallback rescue for uncovered regions.

Takes bad regions from metric_evaluator, crops them from the original
image, saves as PNG, and returns new element dicts compatible with
the existing Paper2Any _render_xml_node format.

Strategy (conservative):
    - Crop the region from the original image
    - Save as PNG file
    - Return as kind="image" element with image_path
    - Skip regions that are >95% white or too small

Usage:
    from dataflow_agent.toolkits.image2drawio.refinement_processor import refine

    new_elements = refine(
        image_path="input.png",
        bad_regions=[...],           # from metric_evaluator
        existing_elements=[...],
        output_dir="outputs/xx",
    )
    # new_elements are dicts with kind="image", bbox_px, image_path, etc.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from dataflow_agent.logger import get_logger

log = get_logger(__name__)

# ======================== configuration ========================

DEFAULT_CONFIG: Dict[str, Any] = {
    "min_region_area": 100,         # skip regions smaller than this (px)
    "min_region_ratio": 0.0005,     # skip regions smaller than 0.05% of image
    "expand_margin": 5,             # expand crop by N pixels each side
    "skip_mostly_white": True,      # skip regions that are almost all white
    "white_threshold": 0.95,        # ratio of white pixels to skip
    "white_pixel_value": 245,       # grayscale > this = "white"
}


# ======================== public API ========================

def refine(
    image_path: str,
    bad_regions: List[Dict[str, Any]],
    existing_elements: List[Dict[str, Any]],
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Process bad regions and return new image elements.

    Args:
        image_path: path to the original image
        bad_regions: list of dicts from metric_evaluator (each has "bbox")
        existing_elements: current element list (for ID numbering)
        output_dir: directory to save cropped PNGs

    Returns:
        List of new element dicts (kind="image") ready for _render_xml_node.
        These should be appended to existing_elements by the caller.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if not bad_regions:
        log.info("[Refinement] No bad regions to process")
        return []

    cv2_image = cv2.imread(image_path)
    if cv2_image is None:
        log.error(f"[Refinement] Cannot read image: {image_path}")
        return []

    h, w = cv2_image.shape[:2]
    img_area = h * w

    crop_dir = Path(output_dir) / "refinement_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    min_area = cfg["min_region_area"]
    min_ratio = cfg["min_region_ratio"]
    margin = cfg["expand_margin"]

    new_elements: List[Dict[str, Any]] = []
    skipped = 0

    # Generate IDs that don't collide with existing elements
    max_existing_id = 0
    for el in existing_elements:
        eid = el.get("id", "")
        if isinstance(eid, str):
            # extract numeric part from "s42", "i13", etc.
            digits = "".join(c for c in eid if c.isdigit())
            if digits:
                max_existing_id = max(max_existing_id, int(digits))
        elif isinstance(eid, int):
            max_existing_id = max(max_existing_id, eid)

    next_id = max_existing_id + 1

    for i, region in enumerate(bad_regions):
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            skipped += 1
            continue

        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        # size filter
        if area < min_area or (img_area > 0 and area < img_area * min_ratio):
            log.debug(f"[Refinement] Region {i} too small ({area}px), skip")
            skipped += 1
            continue

        # white filter
        if cfg.get("skip_mostly_white", True) and _is_mostly_white(
            cv2_image, [x1, y1, x2, y2],
            cfg["white_pixel_value"], cfg["white_threshold"]
        ):
            log.debug(f"[Refinement] Region {i} mostly white, skip")
            skipped += 1
            continue

        # expand margin
        cx1 = max(0, x1 - margin)
        cy1 = max(0, y1 - margin)
        cx2 = min(w, x2 + margin)
        cy2 = min(h, y2 + margin)

        # crop and save
        crop = cv2_image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            skipped += 1
            continue

        crop_path = str(crop_dir / f"refine_{i}.png")
        cv2.imwrite(crop_path, crop)

        # build element dict compatible with _render_xml_node
        new_elements.append({
            "id": f"r{next_id}",
            "kind": "image",
            "bbox_px": [cx1, cy1, cx2, cy2],
            "image_path": crop_path,
            "area": (cx2 - cx1) * (cy2 - cy1),
            "group": "refinement",
            "prompt": "fallback_crop",
            "_source": "refinement",
            "_channel": region.get("channel", "unknown"),
        })
        next_id += 1

    log.info(
        f"[Refinement] Done: {len(new_elements)} new elements, "
        f"{skipped} skipped"
    )

    # save visualisation
    if new_elements:
        _save_visualization(cv2_image, new_elements, existing_elements, output_dir)

    return new_elements


# ======================== helpers ========================

def _is_mostly_white(
    cv2_image: np.ndarray,
    bbox: List[int],
    white_value: int = 245,
    threshold: float = 0.95,
) -> bool:
    """Check if a region is mostly white/empty."""
    x1, y1, x2, y2 = bbox
    h, w = cv2_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return True

    roi = cv2_image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    white_count = int(np.count_nonzero(gray > white_value))
    total = gray.size
    return (white_count / total) > threshold if total > 0 else True


def _save_visualization(
    cv2_image: np.ndarray,
    new_elements: List[Dict],
    existing_elements: List[Dict],
    output_dir: str,
):
    """Save a debug image showing original + new elements."""
    vis = cv2_image.copy()
    h, w = vis.shape[:2]

    # existing elements in blue
    for el in existing_elements:
        bbox = el.get("bbox_px")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 100, 0), 1)

    # new (refinement) elements in red
    for i, el in enumerate(new_elements):
        bbox = el.get("bbox_px")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"NEW-{i}"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    out_path = str(Path(output_dir) / "refinement_result.png")
    cv2.imwrite(out_path, vis)
    log.info(f"[Refinement] Visualisation saved: {out_path}")
