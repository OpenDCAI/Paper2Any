"""
metric_evaluator.py — Image2DrawIO quality evaluation module.

Computes a content-coverage score and detects uncovered "bad regions"
that need fallback rescue.  Works with Paper2Any's dict-based element
format (kind/bbox_px/image_path …).

Core idea:
    score = covered_content_pixels / total_content_pixels × 100

Three-channel bad-region detection:
    Fine   — small icons / sub-figures          (0.05 %–20 % of image)
    Coarse — panels / large images              (0.2 %–30 %)
    Complex— high-variance areas without base64 (heatmaps, photos …)

Usage:
    from dataflow_agent.toolkits.image2drawio.metric_evaluator import evaluate

    result = evaluate(
        image_path="input.png",
        elements=[...],          # from _build_elements_from_sam3
        text_blocks=[...],       # from _text_node
        output_dir="outputs/xx", # optional, saves debug images
    )
    print(result["score"], result["bad_regions"])
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from dataflow_agent.logger import get_logger

log = get_logger(__name__)

# ======================== helpers ========================

def _bbox_area(bbox: List[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_iou(a: List[int], b: List[int]) -> float:
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ======================== configuration ========================

DEFAULT_CONFIG: Dict[str, Any] = {
    # content mask
    "content_threshold": 240,       # 更严格：灰度<240即为内容（原245太宽松）
    "use_edge_detection": True,
    "edge_low": 25,                 # 更敏感的边缘检测（原30）
    "edge_high": 80,                # 降低高阈值以捕获更多边缘（原100）
    "denoise_kernel": 2,
    "min_content_area": 20,         # 保留更小的内容区域（原30）

    # fine channel — 检测小图标/人脸/小子图
    "fine_min_ratio": 0.0003,       # 更敏感：0.03%（原0.05%）
    "fine_max_ratio": 0.25,         # 扩大到25%（原20%）
    "fine_min_fill": 0.12,          # 更宽松的填充率（原0.15）
    "fine_max_aspect": 10.0,        # 允许更大宽高比（原8.0）

    # coarse channel — 检测版块/大图
    "coarse_min_ratio": 0.001,      # 更敏感：0.1%（原0.2%）
    "coarse_max_ratio": 0.35,       # 扩大到35%（原30%）
    "coarse_min_fill": 0.15,        # 更宽松的填充率（原0.20）
    "coarse_max_aspect": 10.0,      # 允许更大宽高比（原8.0）
    "coarse_kernel": 7,             # 稍大的核以合并相邻内容（原5）

    # NMS / dedup
    "nms_iou": 0.25,                # 更严格的NMS（原0.3）
    "existing_iou": 0.45,           # 更积极地保留候选（原0.5）
    "max_covered_ratio": 0.65,      # 降低已覆盖容忍度（原0.7）
    "min_missing_ratio": 0.03,      # 更敏感的漏检内容检测（原0.05）

    # merge
    "merge_distance_ratio": 0.08,   # 更保守的合并距离（原0.10）
    "small_region_threshold": 0.025, # 更小的小区域阈值（原0.03）
}

# element types that need base64 to count as "covered"
_IMAGE_CONTENT_TYPES = {
    "icon", "symbol", "pictogram", "logo",
    "picture", "photo", "chart", "diagram",
    "illustration", "character", "image",
}


# ======================== public API ========================

def evaluate(
    image_path: str,
    elements: List[Dict[str, Any]],
    text_blocks: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate how well existing elements cover the image content.

    Returns dict with keys:
        score          – 0..100  (100 = perfect coverage)
        bad_regions    – list of {bbox, area, area_ratio, channel, …}
        needs_refinement – bool
        metrics        – detailed numbers for debugging
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    cv2_image = cv2.imread(image_path)
    if cv2_image is None:
        log.error(f"[MetricEvaluator] Cannot read image: {image_path}")
        return {"score": 0, "bad_regions": [], "needs_refinement": False, "metrics": {}}

    h, w = cv2_image.shape[:2]
    img_area = h * w

    # 1. content mask (foreground pixels)
    content_mask = _create_content_mask(cv2_image, cfg)
    total_content = int(np.count_nonzero(content_mask))

    # 2. covered mask (elements that produced actual output)
    covered_mask, existing_bboxes = _create_covered_mask(
        elements, text_blocks or [], h, w
    )

    # 3. pixel coverage
    covered_content = int(np.count_nonzero(cv2.bitwise_and(content_mask, covered_mask)))
    pixel_coverage = (covered_content / total_content * 100) if total_content > 0 else 100.0

    # 4. uncovered content
    uncovered = cv2.bitwise_and(content_mask, cv2.bitwise_not(covered_mask))

    # 5. detect bad regions (three channels)
    bad_regions = _detect_bad_regions(
        cv2_image, content_mask, covered_mask, uncovered,
        existing_bboxes, elements, img_area, cfg,
    )

    # 6. score = 100 − bad-region area ratio (de-duplicated)
    bad_mask = np.zeros((h, w), dtype=np.uint8)
    for r in bad_regions:
        x1, y1, x2, y2 = r["bbox"]
        bad_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
    bad_ratio = np.count_nonzero(bad_mask) / img_area * 100 if img_area > 0 else 0
    score = max(0.0, 100.0 - bad_ratio)

    needs_refinement = len(bad_regions) > 0

    metrics = {
        "score": round(score, 2),
        "pixel_coverage": round(pixel_coverage, 2),
        "total_content_px": total_content,
        "covered_content_px": covered_content,
        "image_area": img_area,
        "element_count": len(elements),
        "text_block_count": len(text_blocks or []),
        "bad_region_count": len(bad_regions),
        "bad_region_ratio": round(bad_ratio, 2),
    }

    log.info(
        f"[MetricEvaluator] score={score:.1f}, "
        f"bad_regions={len(bad_regions)}, bad_ratio={bad_ratio:.1f}%"
    )

    # 7. optional visualisation / JSON dump
    if output_dir:
        _save_debug(cv2_image, covered_mask, uncovered, bad_regions,
                    metrics, needs_refinement, score, output_dir)

    return {
        "score": round(score, 2),
        "bad_regions": bad_regions,
        "needs_refinement": needs_refinement,
        "metrics": metrics,
    }


# ======================== content mask ========================

def _create_content_mask(img: np.ndarray, cfg: dict) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # threshold
    mask_gray = ((gray < cfg["content_threshold"]).astype(np.uint8)) * 255

    # edge detection (optional)
    if cfg.get("use_edge_detection", True):
        edges = cv2.Canny(gray, cfg["edge_low"], cfg["edge_high"])
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        mask_gray = cv2.bitwise_or(mask_gray, edges)

    # denoise
    ks = cfg.get("denoise_kernel", 2)
    if ks > 0:
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, np.ones((ks, ks), np.uint8))

    # remove tiny connected components
    min_cc = cfg.get("min_content_area", 30)
    if min_cc > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_gray, connectivity=8)
        clean = np.zeros_like(mask_gray)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_cc:
                clean[labels == i] = 255
        mask_gray = clean

    return mask_gray


# ======================== covered mask ========================

def _create_covered_mask(
    elements: List[Dict],
    text_blocks: List[Dict],
    h: int, w: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Build a mask of regions that already have real output.

    Rules:
      - shape with fill/stroke → covered (矢量已还原)
      - image with existing image_path → covered
      - text block with geometry → covered
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    bboxes: List[List[int]] = []

    for el in elements:
        bbox = el.get("bbox_px")
        if not bbox or len(bbox) != 4:
            continue
        kind = el.get("kind", "")

        # image kind: must have a valid file to count
        if kind == "image":
            ip = el.get("image_path", "")
            if not ip or not os.path.exists(ip):
                continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
            bboxes.append([x1, y1, x2, y2])

    for blk in text_blocks:
        geo = blk.get("geometry", {})
        x = int(float(geo.get("x", 0)))
        y = int(float(geo.get("y", 0)))
        bw = int(float(geo.get("width", 0)))
        bh = int(float(geo.get("height", 0)))
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
            bboxes.append([x1, y1, x2, y2])

    return mask, bboxes


# ======================== bad-region detection ========================

def _detect_bad_regions(
    cv2_image: np.ndarray,
    content_mask: np.ndarray,
    covered_mask: np.ndarray,
    uncovered: np.ndarray,
    existing_bboxes: List[List[int]],
    elements: List[Dict],
    img_area: int,
    cfg: dict,
) -> List[Dict[str, Any]]:
    h, w = cv2_image.shape[:2]
    candidates: List[Tuple[List[int], str]] = []

    # fine channel
    for box in _channel_cc(uncovered, img_area,
                           cfg["fine_min_ratio"], cfg["fine_max_ratio"],
                           cfg["fine_min_fill"], cfg["fine_max_aspect"]):
        candidates.append((box, "fine"))

    # coarse channel
    k = cfg["coarse_kernel"]
    closed = cv2.morphologyEx(uncovered, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    for box in _channel_cc(closed, img_area,
                           cfg["coarse_min_ratio"], cfg["coarse_max_ratio"],
                           cfg["coarse_min_fill"], cfg["coarse_max_aspect"]):
        candidates.append((box, "coarse"))

    # complex channel (high-variance regions without element coverage)
    for box in _detect_complex(cv2_image, elements, covered_mask, img_area):
        candidates.append((box, "complex"))

    log.info(
        f"[MetricEvaluator] candidates: "
        f"fine={sum(1 for _, c in candidates if c == 'fine')}, "
        f"coarse={sum(1 for _, c in candidates if c == 'coarse')}, "
        f"complex={sum(1 for _, c in candidates if c == 'complex')}"
    )

    # small-box-first NMS
    candidates = _nms_small_first(candidates, cfg["nms_iou"])

    # filter vs existing elements + coverage check
    regions = _filter_candidates(
        candidates, covered_mask, existing_bboxes, uncovered, img_area, cfg,
    )

    # merge nearby small regions
    merge_dist = min(h, w) * cfg.get("merge_distance_ratio", 0.10)
    regions = _merge_nearby(regions, merge_dist, img_area,
                            cfg.get("small_region_threshold", 0.03))

    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def _channel_cc(
    mask: np.ndarray, img_area: int,
    min_ratio: float, max_ratio: float,
    min_fill: float, max_aspect: float,
) -> List[List[int]]:
    """Connected-component channel: returns list of bboxes."""
    min_a = img_area * min_ratio
    max_a = img_area * max_ratio
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[List[int]] = []
    for i in range(1, n):
        x, y, rw, rh, cc_area = stats[i]
        if rw <= 0 or rh <= 0:
            continue
        ba = rw * rh
        if ba < min_a or ba > max_a:
            continue
        if max(rw, rh) / max(1, min(rw, rh)) > max_aspect:
            continue
        if cc_area / ba < min_fill:
            continue
        boxes.append([x, y, x + rw, y + rh])
    return boxes


def _detect_complex(
    cv2_image: np.ndarray,
    elements: List[Dict],
    covered_mask: np.ndarray,
    img_area: int,
) -> List[List[int]]:
    """Detect high-complexity regions not covered by any element."""
    h, w = cv2_image.shape[:2]
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ks = max(21, min(h, w) // 50)
    if ks % 2 == 0:
        ks += 1

    local_mean = cv2.blur(gray, (ks, ks))
    local_var = cv2.blur(gray ** 2, (ks, ks)) - local_mean ** 2
    local_var = np.maximum(local_var, 0)

    edges = cv2.Canny(gray.astype(np.uint8), 30, 100)
    edge_density = cv2.blur(edges.astype(np.float32), (ks, ks))

    var_norm = local_var / (local_var.max() + 1e-6)
    edge_norm = edge_density / (edge_density.max() + 1e-6)
    complexity = var_norm * 0.6 + edge_norm * 0.4

    thresh = np.percentile(complexity, 75)
    hi = (complexity > thresh).astype(np.uint8) * 255
    hi = cv2.morphologyEx(hi, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    uncovered_hi = cv2.bitwise_and(hi, cv2.bitwise_not(covered_mask))
    uncovered_hi = cv2.morphologyEx(uncovered_hi, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    uncovered_hi = cv2.morphologyEx(uncovered_hi, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))

    min_a = img_area * 0.002
    max_a = img_area * 0.30
    n, _, stats, _ = cv2.connectedComponentsWithStats(uncovered_hi, connectivity=8)
    boxes: List[List[int]] = []
    for i in range(1, n):
        x, y, rw, rh, _ = stats[i]
        ba = rw * rh
        if ba < min_a or ba > max_a:
            continue
        if max(rw, rh) / max(1, min(rw, rh)) > 8:
            continue
        boxes.append([x, y, x + rw, y + rh])
    return boxes


# ======================== NMS / filtering ========================

def _nms_small_first(
    candidates: List[Tuple[List[int], str]],
    iou_thresh: float,
) -> List[Tuple[List[int], str]]:
    """Keep smaller boxes, suppress larger overlapping ones."""
    if not candidates:
        return []
    items = [(b, c, _bbox_area(b)) for b, c in candidates]
    items.sort(key=lambda x: x[2])  # ascending area
    keep: List[Tuple[List[int], str]] = []
    suppressed = [False] * len(items)
    for i, (bi, ci, _) in enumerate(items):
        if suppressed[i]:
            continue
        keep.append((bi, ci))
        for j in range(i + 1, len(items)):
            if not suppressed[j] and _bbox_iou(bi, items[j][0]) > iou_thresh:
                suppressed[j] = True
    return keep


def _filter_candidates(
    candidates: List[Tuple[List[int], str]],
    covered_mask: np.ndarray,
    existing_bboxes: List[List[int]],
    uncovered: np.ndarray,
    img_area: int,
    cfg: dict,
) -> List[Dict[str, Any]]:
    iou_thresh = cfg["existing_iou"]
    max_covered = cfg["max_covered_ratio"]
    min_missing = cfg["min_missing_ratio"]

    regions: List[Dict[str, Any]] = []
    for box, channel in candidates:
        x1, y1, x2, y2 = box
        area = _bbox_area(box)
        is_complex = channel == "complex"

        # skip if high IoU with an existing element
        eff_iou = 0.8 if is_complex else iou_thresh
        if any(_bbox_iou(box, eb) > eff_iou for eb in existing_bboxes):
            continue

        # skip if mostly already covered (except complex)
        if not is_complex:
            roi = covered_mask[max(0, y1):min(covered_mask.shape[0], y2),
                               max(0, x1):min(covered_mask.shape[1], x2)]
            if roi.size > 0 and float(np.mean(roi > 0)) > max_covered:
                continue

        # missing content pixels
        roi_unc = uncovered[max(0, y1):min(uncovered.shape[0], y2),
                            max(0, x1):min(uncovered.shape[1], x2)]
        missing_px = int(np.count_nonzero(roi_unc)) if not is_complex else area

        if not is_complex and area > 0 and missing_px < area * min_missing:
            continue

        regions.append({
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "area_ratio": round(area / img_area, 4) if img_area > 0 else 0,
            "missing_pixels": missing_px,
            "channel": channel,
            "reason": "complex_image" if is_complex else "uncovered_content",
            "description": (
                f"({x1},{y1})-({x2},{y2}) "
                f"{'complex image' if is_complex else 'uncovered'} [{channel}]"
            ),
        })
    return regions


# ======================== merge ========================

def _merge_nearby(
    regions: List[Dict],
    merge_dist: float,
    img_area: int,
    small_thresh: float,
) -> List[Dict]:
    if len(regions) <= 1:
        return regions

    large = [r for r in regions if r["area_ratio"] >= small_thresh]
    small = [r for r in regions if r["area_ratio"] < small_thresh]
    if len(small) <= 1:
        return regions

    def _dist(a, b):
        dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        return max(dx, dy)

    n = len(small)
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _dist(small[i]["bbox"], small[j]["bbox"]) < merge_dist:
                pi, pj = _find(i), _find(j)
                if pi != pj:
                    parent[pi] = pj

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)

    merged: List[Dict] = []
    for indices in groups.values():
        if len(indices) == 1:
            merged.append(small[indices[0]])
        else:
            bxs = [small[i]["bbox"] for i in indices]
            mb = [min(b[0] for b in bxs), min(b[1] for b in bxs),
                  max(b[2] for b in bxs), max(b[3] for b in bxs)]
            ma = _bbox_area(mb)
            merged.append({
                "bbox": mb,
                "area": ma,
                "area_ratio": round(ma / img_area, 4) if img_area > 0 else 0,
                "missing_pixels": sum(small[i]["missing_pixels"] for i in indices),
                "channel": "merged",
                "reason": "merged_regions",
                "description": f"merged {len(indices)} small regions",
            })
    return large + merged


# ======================== debug output ========================

def _save_debug(
    cv2_image, covered_mask, uncovered, bad_regions,
    metrics, needs_refinement, score, output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    # visualisation
    vis = cv2_image.copy()
    overlay = cv2_image.copy()
    h, w = vis.shape[:2]
    for i, r in enumerate(bad_regions):
        x1, y1, x2, y2 = r["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"#{i+1} {r['channel']} ({r['area_ratio']*100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 255), -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    result_img = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
    cv2.imwrite(str(Path(output_dir) / "metric_eval.png"), result_img)

    # JSON
    def _native(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(str(Path(output_dir) / "metric_eval.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "score": round(float(score), 2),
                "needs_refinement": bool(needs_refinement),
                "metrics": {k: _native(v) for k, v in metrics.items()},
                "bad_regions": [{k: _native(v) for k, v in r.items()} for r in bad_regions],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info(f"[MetricEvaluator] debug saved to {output_dir}")
