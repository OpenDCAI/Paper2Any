"""
Test script: use models/sam3 checkpoint to segment all arrows in ori.png.

Run:
  pytest tests/test_sam3_arrows.py -v -s
  or: python tests/test_sam3_arrows.py --image tests/ori.png --checkpoint models/sam3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:  # optional for environments without opencv
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _cv2_import_error = exc

import pytest

from dataflow_agent.utils import get_project_root
from dataflow_agent.toolkits.multimodaltool.sam_tool import run_sam_auto, free_sam_model


def _resolve_path(path_str: str | None, candidates: List[str]) -> Path:
    root = get_project_root()
    if path_str:
        p = Path(path_str)
        if not p.is_absolute():
            p = (root / p).resolve()
        return p
    for rel in candidates:
        p = (root / rel).resolve()
        if p.exists():
            return p
    return (root / candidates[0]).resolve()


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        weights = list(path.glob("*.pt")) + list(path.glob("*.pth"))
        if not weights:
            raise FileNotFoundError(f"No .pt/.pth checkpoint under: {path}")
        # Prefer names containing "sam3" or "sam"
        weights.sort(key=lambda p: (("sam3" not in p.name.lower()), ("sam" not in p.name.lower()), p.name))
        return weights[0]
    return path


def _largest_contour(mask_u8: np.ndarray) -> Tuple[np.ndarray | None, float]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    return cnt, float(cv2.contourArea(cnt))


def _shape_features(mask: np.ndarray) -> Dict[str, float]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    cnt, area = _largest_contour(mask_u8)
    if cnt is None or area <= 0:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "compactness": 0.0,
            "aspect": 0.0,
            "fill": 0.0,
            "mean_thick": 0.0,
            "long_side": 0.0,
            "short_side": 0.0,
        }
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = float(max(w, 1) * max(h, 1))
    peri = float(cv2.arcLength(cnt, True))
    compactness = (4.0 * np.pi * area) / (peri * peri + 1e-6)
    aspect = float(max(w, h) / max(min(w, h), 1))
    fill = float(area / bbox_area)
    # Approximate mean thickness for thin structures
    mean_thick = float((2.0 * area) / (peri + 1e-6))
    return {
        "area": float(area),
        "perimeter": peri,
        "compactness": compactness,
        "aspect": aspect,
        "fill": fill,
        "mean_thick": mean_thick,
        "long_side": float(max(w, h)),
        "short_side": float(max(min(w, h), 1)),
    }


def _is_arrowhead(mask: np.ndarray, img_area: int) -> bool:
    mask_u8 = (mask.astype(np.uint8) * 255)
    cnt, area = _largest_contour(mask_u8)
    if cnt is None:
        return False
    if area < max(30.0, 0.00002 * img_area):
        return False
    if area > 0.08 * img_area:
        return False
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
    if len(approx) < 3 or len(approx) > 6:
        return False
    # Check for at least one sharp angle
    pts = approx.reshape(-1, 2)
    min_angle = 180.0
    for i in range(len(pts)):
        p0 = pts[i - 1].astype(np.float32)
        p1 = pts[i].astype(np.float32)
        p2 = pts[(i + 1) % len(pts)].astype(np.float32)
        v1 = p0 - p1
        v2 = p2 - p1
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosang)))
        min_angle = min(min_angle, angle)
    feats = _shape_features(mask)
    return (min_angle <= 70.0) and (feats["compactness"] >= 0.12)


def _line_hint_mask(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    line_mask = np.zeros_like(gray)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=25,
        maxLineGap=8,
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)
    return line_mask


def segment_arrows(
    image_path: Path,
    checkpoint: Path,
    device: str,
    out_dir: Path,
    save_debug: bool = True,
) -> Path:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(f"opencv-python is required: {_cv2_import_error}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint} (expected models/sam3)."
        )
    checkpoint = _resolve_checkpoint_path(checkpoint)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    img_area = int(img_bgr.shape[0] * img_bgr.shape[1])
    line_hint = _line_hint_mask(img_bgr)

    try:
        items = run_sam_auto(str(image_path), checkpoint=str(checkpoint), device=device)
    except Exception as exc:
        if device.startswith("cuda"):
            items = run_sam_auto(str(image_path), checkpoint=str(checkpoint), device="cpu")
        else:
            raise exc

    selected: List[Dict[str, object]] = []
    line_candidates: List[Dict[str, object]] = []
    arrowhead_candidates: List[Dict[str, object]] = []
    min_area = max(30.0, 0.00002 * img_area)

    for idx, it in enumerate(items):
        mask = it.get("mask")
        if mask is None:
            continue
        mask = mask.astype(bool)
        feats = _shape_features(mask)
        if feats["area"] < min_area:
            continue

        overlap = float((mask & (line_hint > 0)).sum()) / max(feats["area"], 1.0)
        slender = feats["long_side"] / max(feats["mean_thick"], 1e-6)
        line_like = (
            (overlap >= 0.05)
            or (feats["aspect"] >= 2.2 and feats["fill"] <= 0.7 and feats["compactness"] <= 0.25)
            or (feats["mean_thick"] > 0 and slender >= 5.5)
        )

        if line_like:
            line_candidates.append(
                {
                    "idx": idx,
                    "type": "line",
                    "area": feats["area"],
                    "overlap": overlap,
                }
            )

    for idx, it in enumerate(items):
        mask = it.get("mask")
        if mask is None:
            continue
        mask = mask.astype(bool)
        feats = _shape_features(mask)
        if feats["area"] < min_area:
            continue
        if not _is_arrowhead(mask, img_area):
            continue
        arrowhead_candidates.append(
            {
                "idx": idx,
                "type": "arrowhead",
                "area": feats["area"],
                "overlap": 0.0,
            }
        )

    # If we have arrowheads, keep only line segments that touch them.
    if arrowhead_candidates:
        ah_union = np.zeros(line_hint.shape, dtype=bool)
        for sel in arrowhead_candidates:
            idx = int(sel["idx"])
            mask = items[idx].get("mask")
            if mask is None:
                continue
            ah_union |= mask.astype(bool)
        ah_union_u8 = (ah_union.astype(np.uint8) * 255)
        ah_union_u8 = cv2.dilate(ah_union_u8, np.ones((7, 7), np.uint8), iterations=1)
        ah_union = ah_union_u8 > 0
        for sel in line_candidates:
            idx = int(sel["idx"])
            mask = items[idx].get("mask")
            if mask is None:
                continue
            if (mask.astype(bool) & ah_union).sum() == 0:
                continue
            selected.append(sel)
        if not any(s.get("type") == "line" for s in selected):
            selected = list(line_candidates)
    else:
        selected.extend(line_candidates)

    # Finally, include arrowheads that touch selected lines (to avoid stray triangles)
    line_union = np.zeros(line_hint.shape, dtype=bool)
    for sel in selected:
        if sel.get("type") != "line":
            continue
        idx = int(sel["idx"])
        mask = items[idx].get("mask")
        if mask is None:
            continue
        line_union |= mask.astype(bool)
    line_union_u8 = (line_union.astype(np.uint8) * 255)
    line_union_u8 = cv2.dilate(line_union_u8, np.ones((5, 5), np.uint8), iterations=1)
    line_union = line_union_u8 > 0

    added_all_arrowheads = False
    if not line_union.any() and arrowhead_candidates:
        selected.extend(arrowhead_candidates)
        added_all_arrowheads = True

    if not added_all_arrowheads:
        for sel in arrowhead_candidates:
            idx = int(sel["idx"])
            mask = items[idx].get("mask")
            if mask is None:
                continue
            if (mask.astype(bool) & line_union).sum() == 0:
                continue
            sel["overlap"] = float((mask.astype(bool) & line_union).sum())
            selected.append(sel)

    arrow_mask = np.zeros(line_hint.shape, dtype=bool)
    for sel in selected:
        idx = int(sel["idx"])
        mask = items[idx].get("mask")
        if mask is None:
            continue
        arrow_mask |= mask.astype(bool)

    mask_u8 = (arrow_mask.astype(np.uint8) * 255)
    mask_path = out_dir / "arrow_mask.png"
    cv2.imwrite(str(mask_path), mask_u8)

    overlay = img_bgr.copy()
    overlay[arrow_mask] = (0, 0, 255)
    overlay = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0.0)
    overlay_path = out_dir / "arrow_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    if save_debug:
        cv2.imwrite(str(out_dir / "line_hint.png"), line_hint)
        with (out_dir / "selected_instances.json").open("w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)

    free_sam_model(checkpoint=str(checkpoint))
    return mask_path


def test_sam3_segment_arrows():
    root = get_project_root()
    image_path = _resolve_path(None, ["tests/ori.png", "ori.png"])
    checkpoint = _resolve_path(None, ["models/sam3"])

    if not image_path.exists():
        pytest.skip(f"missing image: {image_path}")
    if not checkpoint.exists():
        pytest.skip(f"missing checkpoint: {checkpoint}")
    try:
        _ = _resolve_checkpoint_path(checkpoint)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    if cv2 is None:  # pragma: no cover
        pytest.skip(f"opencv-python is required: {_cv2_import_error}")

    out_dir = root / "outputs" / "sam3_arrows"
    mask_path = segment_arrows(
        image_path=image_path,
        checkpoint=checkpoint,
        device="cuda",
        out_dir=out_dir,
    )
    assert mask_path.exists()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None and int(mask.sum()) > 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 arrow segmentation test")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="outputs/sam3_arrows")
    parser.add_argument("--no-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image_path = _resolve_path(args.image, ["tests/ori.png", "ori.png"])
    checkpoint = _resolve_path(args.checkpoint, ["models/sam3"])
    out_dir = _resolve_path(args.out_dir, ["outputs/sam3_arrows"])
    segment_arrows(
        image_path=image_path,
        checkpoint=checkpoint,
        device=args.device,
        out_dir=out_dir,
        save_debug=not args.no_debug,
    )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
