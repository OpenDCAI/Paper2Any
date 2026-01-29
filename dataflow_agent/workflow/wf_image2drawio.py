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
from dataflow_agent.toolkits.multimodaltool.sam_tool import segment_layout_boxes, segment_layout_boxes_server, free_sam_model
from dataflow_agent.toolkits.multimodaltool import ppt_tool
from dataflow_agent.toolkits.drawio_tools import wrap_xml
from dataflow_agent.toolkits.image2drawio import (
    classify_shape,
    mask_to_bbox,
    normalize_mask,
    sample_fill_stroke,
    save_masked_rgba,
    bbox_iou_px,
)

log = get_logger(__name__)

TEXT_COLOR = "#111111"
TEXT_FONT_SIZE = 14
TEXT_FONT_STYLE = 1  # draw.io fontStyle=1 => bold


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


def _shape_style(shape_type: str, fill_hex: str, stroke_hex: str) -> str:
    if shape_type == "ellipse":
        base = "shape=ellipse;"
    elif shape_type == "diamond":
        base = "shape=rhombus;"
    else:
        base = "rounded=1;" if shape_type == "rounded_rect" else "rounded=0;"
    return (
        f"{base}whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
        f"fillColor={fill_hex};strokeColor={stroke_hex};"
        f"fontColor={TEXT_COLOR};fontStyle={TEXT_FONT_STYLE};fontSize={TEXT_FONT_SIZE};"
    )


def _text_style(color_hex: str) -> str:
    return (
        "text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;"
        f"strokeColor=none;fillColor=none;fontColor={TEXT_COLOR};"
        f"fontStyle={TEXT_FONT_STYLE};fontSize={TEXT_FONT_SIZE};"
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
        """VLM OCR preferred; fallback to PaddleOCR."""
        img_path = state.fig_draft_path
        if not img_path or not os.path.exists(img_path):
            state.ocr_items = []
            return state

        ocr_items: List[Dict[str, Any]] = []
        api_key = getattr(state.request, "api_key", None) or getattr(state.request, "chat_api_key", None)
        use_vlm = bool(getattr(state.request, "chat_api_url", None)) and bool(api_key)
        if use_vlm:
            try:
                agent = create_vlm_agent(
                    name="ImageTextBBoxAgent",
                    model_name=getattr(state.request, "vlm_model", "qwen-vl-ocr-2025-11-20"),
                    chat_api_url=getattr(state.request, "chat_api_url", None),
                    vlm_mode="ocr",
                    additional_params={"input_image": img_path},
                )
                new_state = await agent.execute(state)
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

        # fallback to PaddleOCR if VLM unavailable or empty
        if not ocr_items:
            try:
                res = ppt_tool.paddle_ocr_page_with_layout(img_path)
                for bbox, text, _conf in res.get("lines", []):
                    if not bbox or not text:
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    ocr_items.append({
                        "text": text.strip(),
                        "bbox_px": [x1, y1, x2, y2],
                    })
            except Exception as e:
                log.error(f"[image2drawio][PaddleOCR] failed: {e}")

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

        sam_ckpt = f"{get_project_root()}/sam_b.pt"
        # optional server URLs (set SAM_SERVER_URLS env, comma-separated)
        sam_server_env = os.getenv("SAM_SERVER_URLS", "").strip()
        sam_server_urls = [u.strip() for u in sam_server_env.split(",") if u.strip()]
        layout_items: List[Dict[str, Any]] = []

        if sam_server_urls:
            try:
                layout_items = segment_layout_boxes_server(
                    image_path=img_path,
                    output_dir=str(out_dir),
                    server_urls=sam_server_urls,
                    checkpoint=sam_ckpt,
                    min_area=120,
                    min_score=0.0,
                    iou_threshold=0.2,
                    top_k=None,
                    nms_by="mask",
                )
            except Exception as e:
                log.warning(f"[image2drawio] SAM server failed: {e}, fallback to local")
                layout_items = []

        if not layout_items:
            try:
                layout_items = segment_layout_boxes(
                    image_path=img_path,
                    output_dir=str(out_dir),
                    checkpoint=sam_ckpt,
                    min_area=120,
                    min_score=0.0,
                    iou_threshold=0.2,
                    top_k=None,
                    nms_by="mask",
                )
            except Exception as e_local:
                log.error(f"[image2drawio] SAM local failed: {e_local}")
                layout_items = []
            finally:
                try:
                    free_sam_model(checkpoint=sam_ckpt)
                except Exception:
                    pass

        # compute bbox_px
        try:
            with Image.open(img_path) as tmp:
                w, h = tmp.size
        except Exception:
            w, h = 1024, 1024

        for it in layout_items:
            bbox = it.get("bbox")
            if bbox and len(bbox) == 4:
                x1n, y1n, x2n, y2n = bbox
                x1 = int(round(x1n * w))
                y1 = int(round(y1n * h))
                x2 = int(round(x2n * w))
                y2 = int(round(y2n * h))
                it["bbox_px"] = [x1, y1, x2, y2]

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

        base_dir = Path(_ensure_result_path(state))
        icon_dir = base_dir / "icons"
        icon_dir.mkdir(parents=True, exist_ok=True)

        shapes = []
        images = []

        # classify SAM items
        for idx, it in enumerate(getattr(state, "layout_items", []) or []):
            mask = it.get("mask")
            bbox_px = it.get("bbox_px")
            if mask is None or bbox_px is None:
                if mask is None:
                    continue
                try:
                    tmp_mask = normalize_mask(mask, image_bgr.shape[:2])
                    bbox_px = mask_to_bbox(tmp_mask)
                except Exception:
                    bbox_px = None
                if bbox_px is None:
                    continue

            mask = normalize_mask(mask, image_bgr.shape[:2])
            shape_type, conf = classify_shape(mask)

            if shape_type != "unknown" and conf >= 0.8:
                fill_hex, stroke_hex = sample_fill_stroke(image_bgr, mask)
                shapes.append({
                    "id": f"s{idx}",
                    "kind": "shape",
                    "shape_type": shape_type,
                    "bbox_px": bbox_px,
                    "fill": fill_hex,
                    "stroke": stroke_hex,
                    "text": "",
                    "area": it.get("area", 0),
                })
            else:
                out_path = icon_dir / f"icon_{idx}.png"
                save_masked_rgba(image_bgr, mask, str(out_path))
                images.append({
                    "id": f"i{idx}",
                    "kind": "image",
                    "bbox_px": bbox_px,
                    "image_path": str(out_path),
                    "area": it.get("area", 0),
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
        unassigned_text = []
        for t in ocr_items:
            tb = t.get("bbox_px")
            if not tb:
                continue
            cx = (tb[0] + tb[2]) * 0.5
            cy = (tb[1] + tb[3]) * 0.5
            best_iou = 0.0
            best_idx = -1
            for i, s in enumerate(shapes):
                sb = s["bbox_px"]
                if sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]:
                    iou = bbox_iou_px(tb, sb)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
            if best_idx >= 0 and best_iou > 0.05:
                text_val = t.get("text", "").strip()
                if text_val:
                    if shapes[best_idx]["text"]:
                        shapes[best_idx]["text"] += "\n" + text_val
                    else:
                        shapes[best_idx]["text"] = text_val
            else:
                unassigned_text.append(t)

        texts = []
        for i, t in enumerate(unassigned_text):
            tb = t.get("bbox_px")
            if not tb:
                continue
            texts.append({
                "id": f"t{i}",
                "kind": "text",
                "bbox_px": tb,
                "text": t.get("text", ""),
                "color": TEXT_COLOR,
            })

        # sort elements by z (shapes large -> small, then images, then texts)
        shapes.sort(key=lambda s: s.get("area", 0), reverse=True)
        images.sort(key=lambda s: s.get("area", 0), reverse=True)

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
                style = _shape_style(el.get("shape_type", "rect"), el.get("fill", "#ffffff"), el.get("stroke", "#000000"))
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
                style = _text_style(el.get("color", "#000000"))
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
