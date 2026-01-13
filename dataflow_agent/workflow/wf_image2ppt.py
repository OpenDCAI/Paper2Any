"""
image2ppt workflow (VLM debug only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
当前版本仅用于调试 VLM bbox：
1. Input: 单张图片 (FIGURE)
2. VLM: ImageTextBBoxAgent 提取文本 + bbox
3. Debug 输出：
   - 在原图上画 bbox + 文本，保存 *_bbox_debug.png
   - 把 bbox 区域刷成白色，保存 *_no_text.png
"""

from __future__ import annotations
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import time
import copy

import cv2
import numpy as np
from PIL import Image

from dataflow_agent.workflow.registry import register
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger

from dataflow_agent.state import Paper2FigureState
from dataflow_agent.utils import get_project_root
from dataflow_agent.agentroles import create_vlm_agent

log = get_logger(__name__)

# --- Helpers ---

def _ensure_result_path(state: Paper2FigureState) -> str:
    raw = getattr(state, "result_path", None)
    if raw:
        return raw
    root = get_project_root()
    ts = int(__import__("time").time())
    base_dir = (root / "outputs" / "image2ppt" / str(ts)).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    state.result_path = str(base_dir)
    return state.result_path

# --- Workflow Definition ---

@register("image2ppt")
def create_image2ppt_graph() -> GenericGraphBuilder:
    """
    简化版 image2ppt，仅做 VLM bbox 可视化 + 去字图生成，不生成 PPT。
    """
    builder = GenericGraphBuilder(state_model=Paper2FigureState, entry_point="_start_")

    # 1. 初始化输出目录
    def _init_result_path(state: Paper2FigureState) -> Paper2FigureState:
        _ensure_result_path(state)
        return state

    # 2. 输入处理：FIGURE -> slide_images
    async def input_processing_node(state: Paper2FigureState) -> Paper2FigureState:
        req = getattr(state, "request", None)
        if not req:
            log.warning("[image2ppt][input] state.request is None")
            return state

        if req.input_type == "FIGURE":
            img_path = req.input_content
            if img_path and os.path.exists(img_path):
                state.slide_images = [img_path]
            else:
                log.error(f"[image2ppt][input] FIGURE image not found: {img_path}")
        elif isinstance(req.input_content, list):
            state.slide_images = [p for p in req.input_content if os.path.exists(p)]
        else:
            log.warning("[image2ppt][input] unsupported input_type, expect FIGURE or list")

        if not getattr(state, "slide_images", []):
            log.warning("[image2ppt][input] No valid slide images found in input.")
        else:
            log.info(f"[image2ppt][input] slide_images = {state.slide_images}")
        return state

    # 3. 只跑 VLM，写 state.vlm_pages
    async def vlm_only_node(state: Paper2FigureState) -> Paper2FigureState:
        image_paths: List[str] = getattr(state, "slide_images", []) or []
        if not image_paths:
            log.warning("[image2ppt][vlm] no slide_images, skip")
            state.vlm_pages = []
            return state

        async def _process_single_image(page_idx: int, img_path: str) -> Dict[str, Any]:
            try:
                temp_state = copy.copy(state)
                temp_state.result_path = state.result_path

                agent = create_vlm_agent(
                    name="ImageTextBBoxAgent",
                    model_name="qwen-vl-ocr-2025-11-20",
                    chat_api_url=getattr(state.request, "chat_api_url", None),
                    vlm_mode="understanding",
                    additional_params={
                        "input_image": img_path
                    }
                )

                new_state = await agent.execute(temp_state)
                bbox_res = getattr(new_state, "bbox_result", [])
                log.info(f"[image2ppt][VLM] page#{page_idx+1} extracted {len(bbox_res)} text items")
                return {
                    "page_idx": page_idx,
                    "path": img_path,
                    "vlm_data": bbox_res,
                }
            except Exception as e:
                log.error(f"[image2ppt][VLM] page#{page_idx+1} failed: {e}")
                return {
                    "page_idx": page_idx,
                    "path": img_path,
                    "vlm_data": [],
                    "error": str(e),
                }

        tasks = [_process_single_image(i, p) for i, p in enumerate(image_paths)]
        results = await asyncio.gather(*tasks)

        # 关键：直接写主 state.vlm_pages
        state.vlm_pages = results
        log.info(f"[image2ppt][VLM] state.vlm_pages len = {len(state.vlm_pages)}")
        return state

    # 4. 画框 + 去字图
    async def debug_draw_and_mask_node(state: Paper2FigureState) -> Paper2FigureState:
        vlm_pages = getattr(state, "vlm_pages", []) or []
        if not vlm_pages:
            log.warning("[image2ppt][debug] No VLM pages, skip debug draw.")
            return state

        base_dir = Path(_ensure_result_path(state))
        debug_dir = base_dir / "vlm_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for page in vlm_pages:
            page_idx = page.get("page_idx", 0)
            img_path = page.get("path")
            items = page.get("vlm_data") or []

            log.info(f"[image2ppt][debug] page#{page_idx+1} img_path={img_path}, items={len(items)}")

            if not img_path or not os.path.exists(img_path):
                log.warning(f"[image2ppt][debug] image not found for page#{page_idx+1}: {img_path}")
                continue

            try:
                pil_img = Image.open(img_path).convert("RGB")
                w, h = pil_img.size
                img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                log.error(f"[image2ppt][debug] open image failed: {e}")
                continue

            debug_img = img_cv.copy()
            mask_img = img_cv.copy()

            # 用户要求去掉 1024 max 处理，直接使用原图尺寸
            model_process_w, model_process_h = w, h

            for it in items:
                # 兼容 qwen-vl-ocr 的 rotate_rect -> bbox
                if "rotate_rect" in it and "bbox" not in it:
                    try:
                        rr = it["rotate_rect"]
                        if isinstance(rr, list) and len(rr) == 5:
                            cx, cy, rw, rh, angle = rr
                            # cv2.boxPoints 需要 ((cx, cy), (w, h), angle)
                            rect = ((float(cx), float(cy)), (float(rw), float(rh)), float(angle))
                            box = cv2.boxPoints(rect)
                            
                            x_min = np.min(box[:, 0])
                            x_max = np.max(box[:, 0])
                            y_min = np.min(box[:, 1])
                            y_max = np.max(box[:, 1])
                            
                            # 坐标是归一化到 0-1000 的，所以除以 1000 得到 0-1 的 bbox
                            VLM_SCALE = 1000.0
                            it["bbox"] = [
                                max(0.0, min(1.0, y_min / VLM_SCALE)),
                                max(0.0, min(1.0, x_min / VLM_SCALE)),
                                max(0.0, min(1.0, y_max / VLM_SCALE)),
                                max(0.0, min(1.0, x_max / VLM_SCALE))
                            ]
                    except Exception as e:
                        log.warning(f"[image2ppt][debug] rotate_rect convert failed: {e}")

                bn = it.get("bbox")
                txt = it.get("text", "")
                if not bn or len(bn) != 4:
                    continue

                # VLM bbox: [ymin, xmin, ymax, xmax] in 0-1
                y1_n, x1_n, y2_n, x2_n = bn
                x1 = int(x1_n * w)
                y1 = int(y1_n * h)
                x2 = int(x2_n * w)
                y2 = int(y2_n * h)

                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = (txt or "")[:10].replace("\n", " ")
                cv2.putText(
                    debug_img,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                mask_img[y1:y2, x1:x2] = (255, 255, 255)

            debug_path = debug_dir / f"page_{page_idx+1:03d}_bbox_debug.png"
            no_text_path = debug_dir / f"page_{page_idx+1:03d}_no_text.png"

            cv2.imwrite(str(debug_path), debug_img)
            cv2.imwrite(str(no_text_path), mask_img)

            log.info(f"[image2ppt][debug] Saved bbox debug: {debug_path}")
            log.info(f"[image2ppt][debug] Saved no-text image: {no_text_path}")

        return state

    nodes = {
        "_start_": _init_result_path,
        "input_processing": input_processing_node,
        "vlm_only": vlm_only_node,
        "debug_draw_and_mask": debug_draw_and_mask_node,
        "_end_": lambda s: s,
    }

    edges = [
        ("input_processing", "vlm_only"),
        ("vlm_only", "debug_draw_and_mask"),
        ("debug_draw_and_mask", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges)
    builder.add_edge("_start_", "input_processing")
    return builder
