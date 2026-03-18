"""
image2drawio workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convert a single diagram image into editable DrawIO XML.

Pipeline:
1) OCR (VLM Qwen-VL-OCR preferred, fallback to PaddleOCR)
2) Generate no-text mask + inpainting (optional)
3) SAM segmentation on clean background
4) Shape classification + color sampling
5) Text assignment + image/icon extraction
6) DrawIO XML generation
"""

from __future__ import annotations

import asyncio
import base64
import copy
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from dataflow_agent.workflow.registry import register
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from dataflow_agent.state import Paper2FigureState
from dataflow_agent.agentroles import create_vlm_agent

from dataflow_agent.toolkits.multimodaltool.req_img import generate_or_edit_and_save_image_async
from dataflow_agent.toolkits.multimodaltool.ocr_config import get_ocr_api_credentials
from dataflow_agent.toolkits.multimodaltool.sam_tool import (
    run_sam_auto,
    run_sam_auto_server,
    postprocess_sam_items,
    filter_sam_items_by_area_and_score,
    free_sam_model,
)
from dataflow_agent.toolkits.drawio_tools import wrap_xml
from dataflow_agent.toolkits.image2drawio import (
    classify_shape,
    extract_text_color,
    mask_to_bbox,
    normalize_mask,
    sample_fill_stroke,
    save_masked_rgba,
    bbox_iou_px,
)

log = get_logger(__name__)

TEXT_COLOR = "#111111"
TEXT_FONT_SIZE_DEFAULT = 14
TEXT_FONT_SIZE_MIN = 8
TEXT_FONT_SIZE_MAX = 48
TEXT_FONT_SCALE = 0.7
TEXT_FONT_MAX_RATIO_SHAPE = 0.45
TEXT_FONT_STYLE = 1  # draw.io fontStyle=1 => bold
MAX_DRAWIO_ELEMENTS = 600
MIN_IMAGE_AREA_RATIO = 0.00001
SHAPE_CONF_THRESHOLDS = {
    "rect": 0.75,
    "rounded_rect": 0.55,
    "ellipse": 0.75,
    "diamond": 0.7,
}


def _ensure_result_path(state: Paper2FigureState) -> str:
    raw = getattr(state, "result_path", None)
    if raw:
        return raw
    root = get_project_root()
    ts = int(time.time())
    base_dir = (root / "outputs" / "image2drawio" / str(ts)).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    state.result_path = str(base_dir)
    return state.result_path


def _escape_xml(text: str) -> str:
    if text is None:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def _clamp_int(val: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(val))))


def _font_size_from_height(height_px: float, max_px: Optional[float] = None) -> int:
    if height_px <= 0:
        return TEXT_FONT_SIZE_DEFAULT
    size = height_px * TEXT_FONT_SCALE
    if max_px is not None and max_px > 0:
        size = min(size, max_px)
    return _clamp_int(size, TEXT_FONT_SIZE_MIN, TEXT_FONT_SIZE_MAX)


def _hex_to_rgb(hex_color: str) -> Optional[tuple[int, int, int]]:
    if not hex_color or not isinstance(hex_color, str):
        return None
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except Exception:
        return None


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _average_hex(colors: List[str]) -> Optional[str]:
    vals = []
    for c in colors:
        rgb = _hex_to_rgb(c)
        if rgb is not None:
            vals.append(rgb)
    if not vals:
        return None
    arr = np.array(vals, dtype=np.float32)
    mean = np.mean(arr, axis=0)
    rgb = tuple(int(max(0, min(255, v))) for v in mean.tolist())
    return _rgb_to_hex(rgb)


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _pick_contrast_text_color(fill_hex: str) -> str:
    rgb = _hex_to_rgb(fill_hex)
    if rgb is None:
        return TEXT_COLOR
    return "#ffffff" if _luminance(rgb) < 140 else TEXT_COLOR


def _bbox_area(b: List[int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _sanitize_bbox(bbox_px: Optional[List[int]], target_shape: tuple[int, int, int]) -> Optional[List[int]]:
    if not bbox_px or len(bbox_px) != 4:
        return None
    h, w = target_shape[:2]
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_px]
    except Exception:
        return None
    # Detect normalized bbox
    if 0.0 <= x2 <= 1.5 and 0.0 <= y2 <= 1.5 and w > 2 and h > 2:
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def _bbox_intersection_ratio(a: List[int], b: List[int]) -> float:
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = _bbox_area(a)
    if area_a == 0:
        return 0.0
    return inter / float(area_a)


def _save_bbox_crop(image_bgr: np.ndarray, bbox_px: List[int], out_path: str) -> str:
    bbox = _sanitize_bbox(bbox_px, image_bgr.shape)
    if not bbox:
        return out_path
    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2, x1:x2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, crop)
    return out_path




def _build_mxcell(
    cell_id: str,
    value: str,
    style: str,
    bbox_px: List[int],
    parent: str = "1",
    vertex: bool = True,
) -> str:
    x1, y1, x2, y2 = bbox_px
    w = max(1, int(x2 - x1))
    h = max(1, int(y2 - y1))
    x = int(x1)
    y = int(y1)
    v_attr = "1" if vertex else "0"
    return (
        f"<mxCell id=\"{cell_id}\" value=\"{_escape_xml(value)}\" style=\"{style}\" "
        f"vertex=\"{v_attr}\" parent=\"{parent}\">"
        f"<mxGeometry x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" as=\"geometry\"/>"
        f"</mxCell>"
    )


def _shape_style(
    shape_type: str,
    fill_hex: str,
    stroke_hex: str,
    font_size: Optional[int] = None,
    font_color: Optional[str] = None,
) -> str:
    if shape_type == "ellipse":
        base = "shape=ellipse;"
    elif shape_type == "diamond":
        base = "shape=rhombus;"
    else:
        base = "rounded=1;" if shape_type == "rounded_rect" else "rounded=0;"
    fs = int(font_size) if font_size else TEXT_FONT_SIZE_DEFAULT
    fc = font_color or TEXT_COLOR
    return (
        f"{base}whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
        f"fillColor={fill_hex};strokeColor={stroke_hex};"
        f"fontColor={fc};fontStyle={TEXT_FONT_STYLE};fontSize={fs};"
    )


def _text_style(color_hex: str, font_size: Optional[int] = None) -> str:
    fs = int(font_size) if font_size else TEXT_FONT_SIZE_DEFAULT
    return (
        "text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;"
        f"strokeColor=none;fillColor=none;fontColor={color_hex or TEXT_COLOR};"
        f"fontStyle={TEXT_FONT_STYLE};fontSize={fs};"
    )


def _image_style(data_uri: str) -> str:
    safe_uri = data_uri.replace(";", "%3B")
    return f"shape=image;imageAspect=0;aspect=fixed;image={safe_uri};"


@register("image2drawio")
def create_image2drawio_graph() -> GenericGraphBuilder:
    builder = GenericGraphBuilder(state_model=Paper2FigureState, entry_point="_start_")

    def _init_node(state: Paper2FigureState) -> Paper2FigureState:
        _ensure_result_path(state)
        return state

    def _input_node(state: Paper2FigureState) -> Paper2FigureState:
        req = getattr(state, "request", None)
        if not req:
            return state
        img_path = getattr(req, "input_content", None) or getattr(req, "prev_image", None)
        if img_path and os.path.exists(img_path):
            state.fig_draft_path = img_path
        else:
            log.error(f"[image2drawio] input image not found: {img_path}")
        return state

    async def _ocr_node(state: Paper2FigureState) -> Paper2FigureState:
        """Use unified remote OCR service for text + bbox extraction."""
        img_path = state.fig_draft_path
        if not img_path or not os.path.exists(img_path):
            state.ocr_items = []
            return state

        ocr_items: List[Dict[str, Any]] = []
        try:
            ocr_api_url, ocr_api_key = get_ocr_api_credentials()
            temp_state = copy.copy(state)
            if getattr(temp_state, "request", None):
                temp_state.request = copy.copy(state.request)
                temp_state.request.chat_api_url = ocr_api_url
                temp_state.request.api_key = ocr_api_key
                temp_state.request.chat_api_key = ocr_api_key

            agent = create_vlm_agent(
                name="ImageTextBBoxAgent",
                model_name=getattr(state.request, "vlm_model", "qwen-vl-ocr-2025-11-20"),
                chat_api_url=ocr_api_url,
                vlm_mode="ocr",
                additional_params={"input_image": img_path},
            )
            new_state = await agent.execute(temp_state)
            bbox_res = getattr(new_state, "bbox_result", [])
        except Exception as e:
            log.warning(f"[image2drawio][VLM] OCR failed: {e}")
            bbox_res = []

        # Normalize to px
        try:
            pil_img = Image.open(img_path)
            w, h = pil_img.size
            VLM_SCALE = 1000.0
            for it in bbox_res or []:
                if "rotate_rect" in it and "bbox" not in it:
                    rr = it.get("rotate_rect")
                    if isinstance(rr, list) and len(rr) == 5:
                        cx, cy, rw, rh, angle = rr
                        rect = ((float(cx), float(cy)), (float(rw), float(rh)), float(angle))
                        box = cv2.boxPoints(rect)
                        x_min = np.min(box[:, 0])
                        x_max = np.max(box[:, 0])
                        y_min = np.min(box[:, 1])
                        y_max = np.max(box[:, 1])
                        it["bbox"] = [
                            max(0.0, min(1.0, y_min / VLM_SCALE)),
                            max(0.0, min(1.0, x_min / VLM_SCALE)),
                            max(0.0, min(1.0, y_max / VLM_SCALE)),
                            max(0.0, min(1.0, x_max / VLM_SCALE)),
                        ]
                if "bbox" in it:
                    y1_n, x1_n, y2_n, x2_n = it["bbox"]
                    x1 = int(x1_n * w)
                    y1 = int(y1_n * h)
                    x2 = int(x2_n * w)
                    y2 = int(y2_n * h)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    ocr_items.append({
                        "text": it.get("text", "").strip(),
                        "bbox_px": [x1, y1, x2, y2],
                    })
        except Exception as e:
            log.warning(f"[image2drawio][VLM] normalize failed: {e}")
            ocr_items = []

        # Build no_text image
        try:
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
            mask_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            for it in ocr_items:
                x1, y1, x2, y2 = it["bbox_px"]
                pad = 2
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), (255, 255, 255), -1)

            base_dir = Path(_ensure_result_path(state))
            debug_dir = base_dir / "ocr_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            no_text_path = debug_dir / "no_text.png"
            cv2.imwrite(str(no_text_path), mask_img)
            state.no_text_path = str(no_text_path)
        except Exception as e:
            log.warning(f"[image2drawio] no_text mask failed: {e}")
            state.no_text_path = ""

        state.ocr_items = ocr_items
        return state

    async def _inpainting_node(state: Paper2FigureState) -> Paper2FigureState:
        img_path = state.fig_draft_path
        no_text_path = getattr(state, "no_text_path", "")
        base_dir = Path(_ensure_result_path(state))
        clean_bg_path = base_dir / "clean_bg.png"

        api_key = getattr(state.request, "api_key", None) or getattr(state.request, "chat_api_key", None) or os.getenv("DF_API_KEY")
        api_url = getattr(state.request, "chat_api_url", None)
        model_name = getattr(state.request, "gen_fig_model", None)

        if api_key and api_url and model_name and no_text_path and os.path.exists(no_text_path):
            prompt = "Remove all text while keeping shapes, icons, and arrows. Do not change layout or colors."
            try:
                await generate_or_edit_and_save_image_async(
                    prompt=prompt,
                    save_path=str(clean_bg_path),
                    api_url=api_url,
                    api_key=api_key,
                    model=model_name,
                    use_edit=True,
                    image_path=no_text_path,
                    aspect_ratio=getattr(state, "aspect_ratio", "16:9"),
                    resolution="2K",
                )
            except Exception as e:
                log.warning(f"[image2drawio] inpainting failed: {e}")

        # fallback to no_text or original
        if not clean_bg_path.exists():
            if no_text_path and os.path.exists(no_text_path):
                try:
                    import shutil
                    shutil.copy(no_text_path, clean_bg_path)
                except Exception:
                    pass
            elif img_path and os.path.exists(img_path):
                try:
                    import shutil
                    shutil.copy(img_path, clean_bg_path)
                except Exception:
                    pass

        state.clean_bg_path = str(clean_bg_path) if clean_bg_path.exists() else ""
        # Normalize clean_bg to original image size if needed
        try:
            if state.clean_bg_path and img_path and os.path.exists(state.clean_bg_path) and os.path.exists(img_path):
                with Image.open(img_path) as orig_img:
                    orig_w, orig_h = orig_img.size
                with Image.open(state.clean_bg_path) as bg_img:
                    bg_w, bg_h = bg_img.size
                    if orig_w and orig_h and (orig_w != bg_w or orig_h != bg_h):
                        resized = bg_img.resize((orig_w, orig_h), Image.LANCZOS)
                        resized.save(state.clean_bg_path)
        except Exception as e:
            log.warning(f"[image2drawio] resize clean_bg failed: {e}")
        return state

    async def _sam_node(state: Paper2FigureState) -> Paper2FigureState:
        img_path = getattr(state, "clean_bg_path", None) or state.fig_draft_path
        if not img_path or not os.path.exists(img_path):
            state.layout_items = []
            return state

        base_dir = Path(_ensure_result_path(state))
        out_dir = base_dir / "sam_items"
        out_dir.mkdir(parents=True, exist_ok=True)

        sam_ckpt = os.environ.get(
            "SAM3_CHECKPOINT_PATH",
            str(get_project_root() / "models" / "sam3" / "sam3.pt"),
        )
        # optional server URLs (set SAM3_SERVER_URLS or SAM_SERVER_URLS env, comma-separated)
        sam_server_env = os.getenv("SAM3_SERVER_URLS", "").strip() or os.getenv("SAM_SERVER_URLS", "").strip()
        sam_server_urls = [u.strip() for u in sam_server_env.split(",") if u.strip()]
        layout_items: List[Dict[str, Any]] = []

        try:
            with Image.open(img_path) as tmp:
                w, h = tmp.size
        except Exception:
            w, h = 1024, 1024
        img_area = max(1, int(w * h))
        small_area = max(800, int(img_area * 0.001))
        min_area_small = max(20, int(img_area * 0.00001))
        min_area_large = max(120, int(img_area * 0.00005))

        if sam_server_urls:
            try:
                layout_items = run_sam_auto_server(
                    image_path=img_path,
                    server_urls=sam_server_urls,
                    checkpoint=sam_ckpt,
                )
            except Exception as e:
                log.warning(f"[image2drawio] SAM server failed: {e}, fallback to local")
                layout_items = []

        if not layout_items:
            try:
                layout_items = run_sam_auto(
                    image_path=img_path,
                    checkpoint=sam_ckpt,
                )
            except Exception as e_local:
                log.error(f"[image2drawio] SAM local failed: {e_local}")
                layout_items = []
            finally:
                try:
                    free_sam_model(checkpoint=sam_ckpt)
                except Exception:
                    pass

        if layout_items:
            small_items = []
            large_items = []
            for it in layout_items:
                area = int(it.get("area", 0) or 0)
                if area < small_area:
                    small_items.append(it)
                else:
                    large_items.append(it)

            large_items = postprocess_sam_items(
                large_items,
                min_area=min_area_large,
                min_score=0.0,
                iou_threshold=0.5,
                top_k=None,
                nms_by="mask",
            )
            small_items = filter_sam_items_by_area_and_score(
                small_items,
                min_area=min_area_small,
                min_score=0.0,
            )
            if small_items:
                small_items = postprocess_sam_items(
                    small_items,
                    min_area=min_area_small,
                    min_score=0.0,
                    iou_threshold=0.85,
                    top_k=None,
                    nms_by="mask",
                )
            layout_items = large_items + small_items

        # compute bbox_px
        for it in layout_items:
            bbox = it.get("bbox")
            if bbox and len(bbox) == 4:
                x1n, y1n, x2n, y2n = bbox
                x1 = int(round(x1n * w))
                y1 = int(round(y1n * h))
                x2 = int(round(x2n * w))
                y2 = int(round(y2n * h))
                bbox_px = _sanitize_bbox([x1, y1, x2, y2], (h, w, 1))
                if bbox_px:
                    it["bbox_px"] = bbox_px
            elif it.get("mask") is not None:
                try:
                    tmp_mask = normalize_mask(it.get("mask"), (h, w))
                    mask_bbox = mask_to_bbox(tmp_mask)
                    if mask_bbox:
                        bbox_px = _sanitize_bbox(mask_bbox, (h, w, 1))
                        if bbox_px:
                            it["bbox_px"] = bbox_px
                except Exception:
                    pass

        state.layout_items = layout_items
        return state

    async def _build_elements_node(state: Paper2FigureState) -> Paper2FigureState:
        img_path = getattr(state, "clean_bg_path", None) or state.fig_draft_path
        if not img_path or not os.path.exists(img_path):
            state.drawio_elements = []
            return state

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            state.drawio_elements = []
            return state
        orig_bgr = image_bgr
        if state.fig_draft_path and os.path.exists(state.fig_draft_path):
            tmp_orig = cv2.imread(state.fig_draft_path)
            if tmp_orig is not None:
                orig_bgr = tmp_orig

        base_dir = Path(_ensure_result_path(state))
        icon_dir = base_dir / "icons"
        icon_dir.mkdir(parents=True, exist_ok=True)

        shapes = []
        images = []
        shape_text_items: List[List[Dict[str, Any]]] = []

        image_area = max(1, int(image_bgr.shape[0] * image_bgr.shape[1]))

        # classify SAM items
        for idx, it in enumerate(getattr(state, "layout_items", []) or []):
            mask = it.get("mask")
            bbox_px = it.get("bbox_px")
            mask_bbox = None

            if mask is not None:
                try:
                    mask = normalize_mask(mask, image_bgr.shape[:2])
                    mask_bbox = mask_to_bbox(mask)
                except Exception:
                    mask = None
                    mask_bbox = None

            if bbox_px is None and mask_bbox is not None:
                bbox_px = mask_bbox
            bbox_px = _sanitize_bbox(bbox_px, image_bgr.shape)
            if bbox_px is None:
                continue

            area = int(it.get("area", 0) or 0)
            if area <= 0:
                area = _bbox_area(bbox_px)
            area_ratio = float(area) / float(image_area)
            bbox_area = _bbox_area(bbox_px)
            bbox_area_ratio = float(bbox_area) / float(image_area)
            if area_ratio > 0.98:
                continue
            if mask is None and bbox_area_ratio > 0.98:
                continue

            if mask is not None and mask_bbox is None:
                mask = None

            if mask is not None:
                shape_type, conf = classify_shape(mask)
                min_conf = SHAPE_CONF_THRESHOLDS.get(shape_type, 0.8)
                fill_ratio = float(area) / float(max(1, bbox_area))
                if shape_type == "unknown" and area_ratio > 0.005 and fill_ratio > 0.88:
                    shape_type = "rect"
                    conf = 0.6
                if shape_type != "unknown" and conf >= min_conf:
                    fill_hex, stroke_hex = sample_fill_stroke(image_bgr, mask)
                    shapes.append({
                        "id": f"s{idx}",
                        "kind": "shape",
                        "shape_type": shape_type,
                        "bbox_px": bbox_px,
                        "fill": fill_hex,
                        "stroke": stroke_hex,
                        "text": "",
                        "text_color": _pick_contrast_text_color(fill_hex),
                        "font_size": None,
                        "area": area,
                    })
                    shape_text_items.append([])
                else:
                    out_path = icon_dir / f"icon_{idx}.png"
                    save_masked_rgba(image_bgr, mask, str(out_path), dilate_px=1)
                    img_area_ratio = float(_bbox_area(mask_bbox or bbox_px)) / float(image_area)
                    if img_area_ratio < MIN_IMAGE_AREA_RATIO:
                        continue
                    images.append({
                        "id": f"i{idx}",
                        "kind": "image",
                        "bbox_px": mask_bbox or bbox_px,
                        "image_path": str(out_path),
                        "area": area,
                    })
            else:
                out_path = icon_dir / f"icon_{idx}.png"
                _save_bbox_crop(image_bgr, bbox_px, str(out_path))
                img_area_ratio = float(_bbox_area(bbox_px)) / float(image_area)
                if img_area_ratio < MIN_IMAGE_AREA_RATIO:
                    continue
                images.append({
                    "id": f"i{idx}",
                    "kind": "image",
                    "bbox_px": bbox_px,
                    "image_path": str(out_path),
                    "area": area,
                })

        # assign OCR text to shapes (scale OCR boxes to match clean_bg size if needed)
        ocr_items = getattr(state, "ocr_items", []) or []
        try:
            if state.fig_draft_path and os.path.exists(state.fig_draft_path):
                with Image.open(state.fig_draft_path) as orig_img:
                    orig_w, orig_h = orig_img.size
            else:
                orig_w, orig_h = None, None
        except Exception:
            orig_w, orig_h = None, None

        color_bgr = orig_bgr
        if orig_w and orig_h:
            tgt_h, tgt_w = image_bgr.shape[:2]
            scale_x = tgt_w / float(orig_w)
            scale_y = tgt_h / float(orig_h)
            if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
                scaled_items = []
                for it in ocr_items:
                    tb = it.get("bbox_px")
                    if not tb or len(tb) != 4:
                        continue
                    x1, y1, x2, y2 = tb
                    scaled_items.append({
                        **it,
                        "bbox_px": [
                            int(round(x1 * scale_x)),
                            int(round(y1 * scale_y)),
                            int(round(x2 * scale_x)),
                            int(round(y2 * scale_y)),
                        ],
                    })
                ocr_items = scaled_items
            if orig_bgr.shape[:2] != (tgt_h, tgt_w):
                try:
                    color_bgr = cv2.resize(orig_bgr, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    color_bgr = image_bgr
        else:
            color_bgr = image_bgr
        unassigned_text = []
        for t in ocr_items:
            tb = t.get("bbox_px")
            if not tb or len(tb) != 4:
                continue
            cx = (tb[0] + tb[2]) * 0.5
            cy = (tb[1] + tb[3]) * 0.5
            best_iou = 0.0
            best_overlap = 0.0
            best_idx = -1
            for i, s in enumerate(shapes):
                sb = s["bbox_px"]
                iou = bbox_iou_px(tb, sb)
                overlap = _bbox_intersection_ratio(tb, sb)
                if overlap > best_overlap or (overlap == best_overlap and iou > best_iou):
                    best_overlap = overlap
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                sb = shapes[best_idx]["bbox_px"]
                center_inside = sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]
                if best_overlap >= 0.3 or best_iou >= 0.05 or center_inside:
                    text_val = t.get("text", "").strip()
                    if text_val and best_idx < len(shape_text_items):
                        shape_text_items[best_idx].append({
                            "text": text_val,
                            "bbox_px": tb,
                            "color": extract_text_color(color_bgr, tb),
                        })
                    continue
            unassigned_text.append(t)

        texts = []
        text_id = 0
        for i, s in enumerate(shapes):
            items = shape_text_items[i] if i < len(shape_text_items) else []
            if len(items) == 1:
                item = items[0]
                tb = item["bbox_px"]
                sb = s.get("bbox_px", [0, 0, 0, 0])
                tb_area = _bbox_area(tb)
                sb_area = _bbox_area(sb)
                area_ratio = float(tb_area) / float(max(1, sb_area))
                if area_ratio >= 0.02:
                    s["text"] = item["text"]
                    s["text_color"] = item["color"] or s.get("text_color", TEXT_COLOR)
                    tb_h = int(tb[3] - tb[1])
                    s["font_size"] = _font_size_from_height(
                        tb_h,
                        max_px=(sb[3] - sb[1]) * TEXT_FONT_MAX_RATIO_SHAPE,
                    )
                else:
                    tb_h = int(tb[3] - tb[1])
                    texts.append({
                        "id": f"t{text_id}",
                        "kind": "text",
                        "bbox_px": tb,
                        "text": item["text"],
                        "color": item["color"],
                        "font_size": _font_size_from_height(tb_h),
                    })
                    text_id += 1
            elif len(items) > 1:
                for item in items:
                    tb = item["bbox_px"]
                    tb_h = int(tb[3] - tb[1])
                    texts.append({
                        "id": f"t{text_id}",
                        "kind": "text",
                        "bbox_px": tb,
                        "text": item["text"],
                        "color": item["color"],
                        "font_size": _font_size_from_height(tb_h),
                    })
                    text_id += 1

        for t in unassigned_text:
            tb = t.get("bbox_px")
            if not tb or len(tb) != 4:
                continue
            tb_h = int(tb[3] - tb[1])
            texts.append({
                "id": f"t{text_id}",
                "kind": "text",
                "bbox_px": tb,
                "text": t.get("text", ""),
                "color": extract_text_color(color_bgr, tb),
                "font_size": _font_size_from_height(tb_h),
            })
            text_id += 1

        # sort elements by z (shapes large -> small, then images, then texts)
        shapes.sort(key=lambda s: s.get("area", 0), reverse=True)
        images.sort(key=lambda s: s.get("area", 0), reverse=True)
        total = len(shapes) + len(images) + len(texts)
        if total > MAX_DRAWIO_ELEMENTS:
            keep = max(0, MAX_DRAWIO_ELEMENTS - len(shapes) - len(texts))
            if keep < len(images):
                images = images[:keep]
        state.drawio_elements = shapes + images + texts
        return state

    async def _render_xml_node(state: Paper2FigureState) -> Paper2FigureState:
        elements = getattr(state, "drawio_elements", []) or []
        clean_bg_path = getattr(state, "clean_bg_path", "") or ""
        has_bg = bool(clean_bg_path and os.path.exists(clean_bg_path))
        if not elements and not has_bg:
            state.drawio_xml = ""
            return state

        cells = []
        id_counter = 2
        page_width = 850
        page_height = 1100

        if has_bg:
            try:
                with Image.open(clean_bg_path) as bg_img:
                    bg_w, bg_h = bg_img.size
                    page_width = bg_w
                    page_height = bg_h
                data_uri = "data:image/png;base64," + _encode_image_base64(clean_bg_path)
                style = _image_style(data_uri)
                cells.append(_build_mxcell(str(id_counter), "", style, [0, 0, bg_w, bg_h]))
                id_counter += 1
            except Exception as e:
                log.warning(f"[image2drawio] embed background failed: {e}")

        for el in elements:
            if el.get("kind") == "shape":
                style = _shape_style(
                    el.get("shape_type", "rect"),
                    el.get("fill", "#ffffff"),
                    el.get("stroke", "#000000"),
                    font_size=el.get("font_size"),
                    font_color=el.get("text_color"),
                )
                value = el.get("text", "")
                cells.append(_build_mxcell(str(id_counter), value, style, el["bbox_px"]))
                id_counter += 1
            elif el.get("kind") == "image":
                img_path = el.get("image_path")
                if not img_path or not os.path.exists(img_path):
                    continue
                data_uri = "data:image/png;base64," + _encode_image_base64(img_path)
                style = _image_style(data_uri)
                cells.append(_build_mxcell(str(id_counter), "", style, el["bbox_px"]))
                id_counter += 1
            elif el.get("kind") == "text":
                style = _text_style(el.get("color", "#000000"), font_size=el.get("font_size"))
                value = el.get("text", "")
                cells.append(_build_mxcell(str(id_counter), value, style, el["bbox_px"]))
                id_counter += 1

        xml_cells = "\n".join(cells)
        full_xml = wrap_xml(xml_cells, page_width=page_width, page_height=page_height)

        base_dir = Path(_ensure_result_path(state))
        out_path = base_dir / "image2drawio.drawio"
        out_path.write_text(full_xml, encoding="utf-8")

        state.drawio_xml = full_xml
        state.drawio_output_path = str(out_path)
        return state

    nodes = {
        "_start_": _init_node,
        "input": _input_node,
        "ocr": _ocr_node,
        "inpainting": _inpainting_node,
        "sam": _sam_node,
        "build_elements": _build_elements_node,
        "render_xml": _render_xml_node,
        "_end_": lambda s: s,
    }

    edges = [
        ("input", "ocr"),
        ("ocr", "inpainting"),
        ("inpainting", "sam"),
        ("sam", "build_elements"),
        ("build_elements", "render_xml"),
        ("render_xml", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges)
    builder.add_edge("_start_", "input")
    return builder
