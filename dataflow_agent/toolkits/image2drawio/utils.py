from __future__ import annotations

from typing import List, Tuple, Optional
import os
import math

import cv2
import numpy as np


def normalize_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Ensure mask is boolean and matches target (H, W)."""
    if mask is None:
        raise ValueError("mask is None")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    h, w = target_shape
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
    return mask


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def classify_shape(mask: np.ndarray) -> Tuple[str, float]:
    """
    Heuristic shape classification.

    Returns (shape_type, confidence) where shape_type in:
    rect | rounded_rect | ellipse | diamond | unknown
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "unknown", 0.0

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 50:
        return "unknown", 0.0

    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return "unknown", 0.0

    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    vertices = len(approx)

    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = float(w * h) if w > 0 and h > 0 else 1.0
    area_ratio = float(area) / bbox_area
    aspect = float(w) / float(h) if h > 0 else 1.0

    circularity = 4.0 * math.pi * float(area) / (float(peri) * float(peri) + 1e-6)

    # Ellipse / circle
    if circularity > 0.75:
        return "ellipse", min(1.0, circularity)

    # Quadrilaterals
    if vertices == 4:
        if area_ratio < 0.7:
            return "diamond", 0.8
        return "rect", 0.8

    # Rounded rectangle: many vertices, high area ratio
    if vertices >= 5 and area_ratio > 0.75 and 0.4 < aspect < 2.5:
        return "rounded_rect", 0.6

    return "unknown", 0.0


def _to_hex(color_bgr: Tuple[int, int, int]) -> str:
    b, g, r = [int(max(0, min(255, c))) for c in color_bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def sample_fill_stroke(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[str, str]:
    """
    Sample fill & stroke colors from original image using the mask.
    Returns (fill_hex, stroke_hex).
    """
    h, w = image_bgr.shape[:2]
    mask = normalize_mask(mask, (h, w))

    # Edge (stroke): dilate - erode
    k = max(1, int(min(h, w) * 0.002))
    k = min(k, 5)
    kernel = np.ones((k, k), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    ero = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    edge = (dil > 0) & (ero == 0)

    stroke_pixels = image_bgr[edge]
    if stroke_pixels.size == 0:
        stroke_pixels = image_bgr[mask]

    # Select darkest quartile by luminance
    if stroke_pixels.size > 0:
        rgb = stroke_pixels[:, ::-1].astype(np.float32)
        lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
        if lum.size > 10:
            thresh = np.percentile(lum, 25)
            sel = stroke_pixels[lum <= thresh]
        else:
            sel = stroke_pixels
        stroke = tuple(np.mean(sel, axis=0).tolist())
    else:
        stroke = (0, 0, 0)

    # Fill: erode mask to remove border
    erode_k = max(1, int(min(h, w) * 0.004))
    erode_k = min(erode_k, 7)
    kernel2 = np.ones((erode_k, erode_k), np.uint8)
    inner = cv2.erode(mask.astype(np.uint8), kernel2, iterations=1) > 0
    fill_pixels = image_bgr[inner]
    if fill_pixels.size == 0:
        fill_pixels = image_bgr[mask]

    if fill_pixels.size > 0:
        fill = tuple(np.median(fill_pixels, axis=0).tolist())
    else:
        fill = (255, 255, 255)

    return _to_hex(fill), _to_hex(stroke)


def extract_text_color(image_bgr: np.ndarray, bbox_px: List[int]) -> str:
    x1, y1, x2, y2 = bbox_px
    x1 = max(0, min(image_bgr.shape[1] - 1, int(x1)))
    x2 = max(0, min(image_bgr.shape[1], int(x2)))
    y1 = max(0, min(image_bgr.shape[0] - 1, int(y1)))
    y2 = max(0, min(image_bgr.shape[0], int(y2)))
    if x2 <= x1 or y2 <= y1:
        return "#000000"
    region = image_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return "#000000"
    rgb = region[:, :, ::-1].reshape(-1, 3).astype(np.float32)
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    if lum.size == 0:
        return "#000000"
    thresh = np.percentile(lum, 25)
    sel = rgb[lum <= thresh]
    if sel.size == 0:
        sel = rgb
    color = tuple(np.mean(sel, axis=0).tolist())
    r, g, b = [int(max(0, min(255, c))) for c in color]
    return f"#{r:02x}{g:02x}{b:02x}"


def save_masked_rgba(image_bgr: np.ndarray, mask: np.ndarray, out_path: str) -> str:
    """Save masked region as RGBA PNG with alpha channel."""
    h, w = image_bgr.shape[:2]
    mask = normalize_mask(mask, (h, w))
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    alpha = (mask.astype(np.uint8) * 255)
    rgba[:, :, 3] = alpha

    bbox = mask_to_bbox(mask)
    if bbox:
        x1, y1, x2, y2 = bbox
        x2 = min(w, x2 + 1)
        y2 = min(h, y2 + 1)
        crop = rgba[y1:y2, x1:x2]
    else:
        crop = rgba

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, crop)
    return out_path


def bbox_iou_px(a: List[int], b: List[int]) -> float:
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    if area_a == 0 or area_b == 0:
        return 0.0
    return inter / float(area_a + area_b - inter + 1e-6)
