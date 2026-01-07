from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureState
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_react_agent
from dataflow_agent.toolkits.imtool.req_img import (
    generate_or_edit_and_save_image_async, 
    gemini_multi_image_edit_async
)
from dataflow_agent.toolkits.imtool.ppt_tool import convert_images_dir_to_pdf_and_ppt_api

log = get_logger(__name__)


def _ensure_result_path(state: Paper2FigureState) -> str:
    """
    统一 paper2ppt workflow 的根输出目录：
    - 若 state.result_path 已存在（通常由调用方传入），直接使用；
    - 否则：使用 get_project_root()/outputs/paper2ppt/<timestamp> 初始化，并写回 state.result_path。
    """
    raw = getattr(state, "result_path", None)
    if raw:
        return raw

    root = get_project_root()
    ts = int(time.time())
    base_dir = (root / "outputs" / "paper2ppt" / str(ts)).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    state.result_path = str(base_dir)
    return state.result_path


def _abs_path(p: str) -> str:
    if not p:
        return ""
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return p


def _is_table_asset(asset_ref: Optional[str]) -> bool:
    """
    约定：asset 是 Table 时，通过 asset_ref: "Table 2" 这种字符串标记。
    """
    if not asset_ref:
        return False
    s = str(asset_ref).strip().lower()
    return s.startswith("table")


def _serialize_prompt_dict(d: Dict[str, Any]) -> str:
    """
    把 dict 安全序列化为 prompt 文本（中文不转义）。
    """
    try:
        return json.dumps(d, ensure_ascii=False, indent=2)
    except Exception:
        return str(d)


def _normalize_single_asset_ref(asset_ref: str) -> str:
    """
    规范化 asset_ref，仅保留第一张图的路径/文件名。
    """
    if not asset_ref:
        return ""
    s = str(asset_ref).strip()
    if not s:
        return ""

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return ""

    if len(parts) > 1:
        log.warning(
            "[paper2ppt] asset_ref 包含多张图片，仅使用第一张。"
            f" raw={asset_ref!r}, first={parts[0]!r}"
        )

    return parts[0]


async def _make_prompt_for_structured_page(item: Dict[str, Any], style: str, state: Paper2FigureState) -> Tuple[str, Optional[str], bool]:
    """
    根据结构化 page item 生成:
    - prompt
    - image_path (如果是编辑模式)
    - use_edit

    规则：
    1) asset 为空：text2img
    2) asset 是图片路径：img2img/edit
    3) asset 是 Table：提取 table png 后走 edit
    """
    asset_ref = item.get("asset_ref") or item.get("asset") or item.get("assetRef") or ""
    asset_ref = str(asset_ref).strip() if asset_ref is not None else ""
    asset_ref = _normalize_single_asset_ref(asset_ref)

    prompt_dict = dict(item)
    for k in ["asset_ref", "asset", "assetRef", "asset_type", "type"]:
        if k in prompt_dict:
            prompt_dict.pop(k, None)

    base = _serialize_prompt_dict(prompt_dict)

    if not asset_ref:
        prompt = f"{base}\n\n根据上述内容。生成{style}风格的 PPT 图像, \n 使用语言：{state.request.language}"
        return prompt, None, False

    # table 占位提取
    if _is_table_asset(asset_ref):
        table_img_path = item.get("table_img_path") or item.get("table_png_path") or ""
        table_img_path = str(table_img_path).strip()

        if not table_img_path:
            state.asset_ref = asset_ref
            agent = create_react_agent(
                name="table_extractor",
                temperature=0.1,
                max_retries=6,
                parser_type="json",
            )
            state = await agent.execute(state=state)

            table_img_path = str(getattr(state, "table_img_path", "") or "").strip()
            log.critical(f'[table_img_path 表格图像路径]:   {table_img_path}')

        if not table_img_path:
            raise ValueError(f"[paper2ppt] 表格提取失败，未得到 table_img_path。asset_ref={asset_ref}")

        image_path = _resolve_asset_path(table_img_path, state)
        if not image_path or not os.path.exists(image_path):
            log.error(f"[paper2ppt] 表格图像文件不存在: {image_path!r} (asset_ref={asset_ref})")
            prompt = f"{base}\n\n根据上述内容生成{style}风格的 PPT 图像, \n 使用语言：{state.request.language}"
            return prompt, None, False

        prompt = f"{base}\n\n根据上述内容绘制ppt，把这个图作为PPT的一部分。生成{style}风格的PPT. \n 使用语言：{state.request.language} !!!"
        return prompt, image_path, True

    # 默认：当作图片路径，走编辑
    image_path = _resolve_asset_path(asset_ref, state)
    if not image_path or not os.path.exists(image_path):
        log.error(f"[paper2ppt] 图片文件不存在: {image_path!r} (asset_ref={asset_ref})")
        prompt = f"{base}\n\n根据上述内容生成{style}风格的 PPT 图像, \n 使用语言：{state.request.language}"
        return prompt, None, False

    prompt = f"{base}\n\n根据上述内容绘制ppt，把这个图作为PPT的一部分。生成{style}风格的PPT. \n 使用语言：{state.request.language} !!!"
    return prompt, image_path, True


def _resolve_asset_path(asset_ref: str, state: Paper2FigureState) -> str:
    """
    根据 state 解析 asset 引用为绝对路径。
    """
    if not asset_ref:
        return ""
    s = str(asset_ref).strip()
    if not s:
        return ""

    p = Path(s)
    if p.is_absolute() or s.startswith("~"):
        return _abs_path(s)

    base_dir = getattr(state, "mineru_root", None) or getattr(state, "result_path", None)
    
    if base_dir:
        try:
            return str((Path(base_dir) / p).resolve())
        except Exception:
            return _abs_path(s)

    return _abs_path(s)


def _extract_image_path_from_pagecontent_item(item: Any) -> Optional[str]:
    """
    支持 pagecontent 直接是图片路径的几种形态
    """
    if not item:
        return None
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for k in ["ppt_img_path", "img_path", "path", "image_path"]:
            v = item.get(k)
            if v:
                return str(v).strip()
    return None


@register("paper2ppt_parallel_consistent_style")
def create_paper2ppt_parallel_consistent_graph() -> GenericGraphBuilder:  # noqa: N802
    """
    Workflow factory: dfa run --wf paper2ppt_parallel_consistent_style

    功能：
    - 基于 paper2ppt_parallel 优化：
    - 先生成第 0 页 (img0)；
    - 后续页面并行生成，并强制使用 img0 作为风格参考：
      - 原本是 Text2Img 的页面 -> 改为 Img2Img (img0 as base)
      - 原本是 Img2Img (有素材) 的页面 -> 改为 Multi-Image Edit (img0 + asset)
    """
    builder = GenericGraphBuilder(state_model=Paper2FigureState, entry_point="_start_")

    def _start_(state: Paper2FigureState) -> Paper2FigureState:
        _ensure_result_path(state)
        state.pagecontent = state.pagecontent or []
        state.generated_pages = state.generated_pages or []
        if not getattr(state.request, "style", None) and getattr(state, "style", None):
            state.request.style = getattr(state, "style")
        return state

    def _route(state: Paper2FigureState) -> str:
        if getattr(state.request, "all_edited_down", False):
            return "export_ppt_assets"
        if not getattr(state, "gen_down", False):
            return "generate_pages"
        return "edit_single_page"

    async def generate_pages(state: Paper2FigureState) -> Paper2FigureState:
        """
        核心逻辑：
        1. 检查是否有用户传入的 ref_img (state.request.ref_img)。
        2. 若有 -> 全量并行生成，每一页都以 ref_img 为风格参考。
        3. 若无 -> 先生成第 0 页 (anchor)，再并行生成第 1~N 页 (以 anchor 为参考)。
        """
        
        async def _call_image_api_with_retry(coro_factory, retries: int = 3, delay: float = 1.0) -> bool:
            last_err: Optional[Exception] = None
            for attempt in range(1, retries + 1):
                try:
                    await coro_factory()
                    return True
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    log.error(f"[paper2ppt] image gen failed attempt {attempt}/{retries}: {e}")
                    if attempt < retries:
                        try:
                            await asyncio.sleep(delay)
                        except Exception:
                            pass
            log.error(f"[paper2ppt] image gen failed after {retries} attempts. last_err={last_err}")
            return False

        result_root = Path(_ensure_result_path(state))
        img_dir = result_root / "ppt_pages"
        img_dir.mkdir(parents=True, exist_ok=True)

        style = getattr(state.request, "style", None) or "kartoon"
        aspect_ratio = getattr(state, "aspect_ratio", None) or "16:9"
        
        page_items = state.pagecontent or []
        if not page_items:
            log.warning("Pagecontent is empty, nothing to generate.")
            return state

        # 检查是否有用户传入的参考图
        user_ref_img = getattr(state.request, "ref_img", None)
        if user_ref_img:
            user_ref_img = _abs_path(str(user_ref_img))
            if not os.path.exists(user_ref_img):
                log.warning(f"[paper2ppt] User provided ref_img not found: {user_ref_img}. Fallback to auto-anchor.")
                user_ref_img = None
            else:
                log.info(f"[paper2ppt] Using user-provided ref_img: {user_ref_img}")

        # 定义通用的单页处理函数，增加 ref_img_path 参数
        async def _process_single_page(
            idx: int, 
            item: Any, 
            ref_img_path: Optional[str] = None
        ) -> Dict[str, Any]:
            
            save_path = str((img_dir / f"page_{idx:03d}.png").resolve())
            
            # --- Case B: pagecontent 本身就是图片路径 (Direct Image) ---
            direct_img_path = _extract_image_path_from_pagecontent_item(item)
            is_direct_image = bool(direct_img_path) and (
                isinstance(item, str)
                or (isinstance(item, dict) and set(item.keys()).intersection({"ppt_img_path", "img_path", "path", "image_path"}))
            )

            if is_direct_image and (not isinstance(item, dict) or ("title" not in item and "layout_description" not in item)):
                # 直接图片编辑模式
                image_path = _abs_path(direct_img_path)
                
                # 如果有 ref_img，则走多图融合；否则单图编辑
                if ref_img_path and os.path.exists(ref_img_path):
                    prompt = (
                        f"Refine this slide image (second image) to match the style of the first image. "
                        f"Maintain the content layout of the second image but unify the color scheme, background, and design elements "
                        f"to be consistent with the {style} style of the first image."
                    )
                    log.info(f"[paper2ppt] page={idx} direct img multi-edit with ref={ref_img_path}")
                    ok = await _call_image_api_with_retry(
                        lambda: gemini_multi_image_edit_async(
                            prompt=prompt,
                            image_paths=[ref_img_path, image_path],
                            save_path=save_path,
                            api_url=state.request.chat_api_url,
                            api_key=state.request.chat_api_key or os.getenv("DF_API_KEY"),
                            model=state.request.gen_fig_model,
                            aspect_ratio=aspect_ratio,
                        )
                    )
                else:
                    prompt = (
                        f"Please beautify this PowerPoint slide image into {style} style. "
                        f"Keep text and structure, enhance colors and background."
                    )
                    log.info(f"[paper2ppt] page={idx} direct img single-edit")
                    ok = await _call_image_api_with_retry(
                        lambda: generate_or_edit_and_save_image_async(
                            prompt=prompt,
                            save_path=save_path,
                            aspect_ratio=aspect_ratio,
                            api_url=state.request.chat_api_url,
                            api_key=state.request.chat_api_key or os.getenv("DF_API_KEY"),
                            model=state.request.gen_fig_model,
                            image_path=image_path,
                            use_edit=True,
                        )
                    )

                if not ok:
                    return {
                        "generated_img_path": None,
                        "page_idx": idx,
                        "mode": "direct_edit_failed",
                        "error": "api failed"
                    }
                return {
                    "generated_img_path": save_path,
                    "page_idx": idx,
                    "mode": "direct_edit",
                }

            # --- Case A: 结构化页面 (Structured) ---
            if not isinstance(item, dict):
                return {"page_idx": idx, "generated_img_path": None, "mode": "invalid_item"}

            try:
                # 解析原始意图：是生图(prompt only) 还是 编辑(prompt + asset)
                base_prompt, asset_path, is_edit_originally = await _make_prompt_for_structured_page(item, style=style, state=state)
            except Exception as e:
                log.error(f"[paper2ppt] page={idx} prompt build failed: {e}")
                return {"page_idx": idx, "generated_img_path": None, "mode": "prompt_failed", "error": str(e)}

            # 策略分支
            use_ref = (ref_img_path and os.path.exists(ref_img_path))
            
            # 1. 如果有 Ref 图
            if use_ref:
                if is_edit_originally and asset_path:
                    # 原本就是 Edit (有素材) -> 改为多图 Edit (Ref + Asset)
                    final_prompt = (
                        f"{base_prompt}\n\n"
                        f"IMPORTANT: The first image provided is the STYLE REFERENCE. "
                        f"The second image is the CONTENT ASSET (e.g. chart/table). "
                        f"Please generate a {style} presentation slide that incorporates the CONTENT from the second image, "
                        f"but strictly follows the visual style, color palette, and background design of the first image."
                    )
                    mode = "multi_edit_ref_asset"
                    log.info(f"[paper2ppt] page={idx} Multi-Edit (Ref+Asset). Asset={asset_path}")
                    
                    ok = await _call_image_api_with_retry(
                        lambda: gemini_multi_image_edit_async(
                            prompt=final_prompt,
                            image_paths=[ref_img_path, asset_path],
                            save_path=save_path,
                            api_url=state.request.chat_api_url,
                            api_key=state.request.chat_api_key or os.getenv("DF_API_KEY"),
                            model=state.request.gen_fig_model,
                            aspect_ratio=aspect_ratio,
                        )
                    )
                else:
                    # 原本是 Text2Img -> 改为 Img2Img (Ref as base)
                    # 注意：这里我们用 Ref 图作为 image_path 传给 edit 接口，
                    # 提示词要求“基于此风格生成新内容”
                    final_prompt = (
                        f"{base_prompt}\n\n"
                        f"Reference the style of the provided image (layout, color, background), "
                        f"but generate NEW CONTENT based on the text description above. "
                        f"Keep the background style consistent."
                    )
                    mode = "edit_ref_style"
                    log.info(f"[paper2ppt] page={idx} Edit (Ref Style).")
                    
                    ok = await _call_image_api_with_retry(
                        lambda: generate_or_edit_and_save_image_async(
                            prompt=final_prompt,
                            save_path=save_path,
                            aspect_ratio=aspect_ratio,
                            api_url=state.request.chat_api_url,
                            api_key=state.request.chat_api_key or os.getenv("DF_API_KEY"),
                            model=state.request.gen_fig_model,
                            image_path=ref_img_path,  # Use ref as base
                            use_edit=True,
                        )
                    )
            
            # 2. 如果没有 Ref 图 (比如是第0页，或者第0页生成失败)
            else:
                final_prompt = base_prompt
                mode = "origin_edit" if is_edit_originally else "origin_gen"
                log.info(f"[paper2ppt] page={idx} Origin {mode}. Asset={asset_path}")
                
                ok = await _call_image_api_with_retry(
                    lambda: generate_or_edit_and_save_image_async(
                        prompt=final_prompt,
                        save_path=save_path,
                        aspect_ratio=aspect_ratio,
                        api_url=state.request.chat_api_url,
                        api_key=state.request.chat_api_key or os.getenv("DF_API_KEY"),
                        model=state.request.gen_fig_model,
                        image_path=asset_path,
                        use_edit=is_edit_originally,
                    )
                )

            if not ok:
                return {
                    "generated_img_path": None,
                    "page_idx": idx,
                    "mode": f"{mode}_failed",
                    "error": "api failed"
                }

            out_item = dict(item)
            out_item.update({
                "generated_img_path": save_path,
                "page_idx": idx,
                "mode": mode,
                "style": style,
            })
            return out_item

        # --- Execution Flow ---
        
        results_map = {}
        
        # 场景 A: 用户提供了参考图 (ref_img) -> 全量并行
        if user_ref_img:
            log.info(f"[paper2ppt_consistent] User ref_img provided. Processing all {len(page_items)} pages in parallel...")
            tasks = []
            for idx in range(len(page_items)):
                tasks.append(_process_single_page(idx, page_items[idx], ref_img_path=user_ref_img))
            
            t_start = time.time()
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            log.info(f"[paper2ppt_consistent] All pages done in {time.time()-t_start:.2f}s")
            
            for i, res in enumerate(all_results):
                if isinstance(res, Exception):
                    log.error(f"[paper2ppt_consistent] Page {i} exception: {res}")
                    results_map[i] = {"page_idx": i, "generated_img_path": None, "mode": "exception", "error": str(res)}
                else:
                    results_map[i] = res

        # 场景 B: 无参考图 -> 生成第0页作为 Anchor，再并行后续
        else:
            # 1. Process Page 0
            log.info("[paper2ppt_consistent] Generating Page 0 as Style Anchor...")
            t0_start = time.time()
            res0 = await _process_single_page(0, page_items[0], ref_img_path=None)
            log.info(f"[paper2ppt_consistent] Page 0 done in {time.time()-t0_start:.2f}s. Result: {res0.get('mode')}")
            
            results_map[0] = res0
            
            # 获取 Anchor Image
            anchor_img_path = res0.get("generated_img_path")
            if not anchor_img_path or not os.path.exists(anchor_img_path):
                log.warning("[paper2ppt_consistent] Page 0 generation failed! Subsequent pages will be generated independently.")
                anchor_img_path = None
            else:
                log.info(f"[paper2ppt_consistent] Anchor Image established: {anchor_img_path}")

            # 2. Process remaining pages in parallel
            if len(page_items) > 1:
                tasks = []
                for idx in range(1, len(page_items)):
                    tasks.append(_process_single_page(idx, page_items[idx], ref_img_path=anchor_img_path))
                
                log.info(f"[paper2ppt_consistent] Generating remaining {len(tasks)} pages concurrently...")
                t_rest_start = time.time()
                rest_results = await asyncio.gather(*tasks, return_exceptions=True)
                log.info(f"[paper2ppt_consistent] Remaining pages done in {time.time()-t_rest_start:.2f}s")
                
                for i, res in enumerate(rest_results):
                    real_idx = i + 1
                    if isinstance(res, Exception):
                        log.error(f"[paper2ppt_consistent] Page {real_idx} exception: {res}")
                        results_map[real_idx] = {
                            "page_idx": real_idx,
                            "generated_img_path": None,
                            "mode": "exception",
                            "error": str(res)
                        }
                    else:
                        results_map[real_idx] = res
        
        # 3. Assemble final list
        new_pagecontent = []
        state.generated_pages = []
        
        for idx in range(len(page_items)):
            res = results_map.get(idx, {})
            # Merge back into original item info
            orig_item = page_items[idx]
            if isinstance(orig_item, dict):
                final_item = dict(orig_item)
            else:
                final_item = {"raw_content": str(orig_item)}
            
            # Update with result
            # remove temp keys if needed, or just overwrite
            final_item.update(res)
            
            new_pagecontent.append(final_item)
            gen_path = final_item.get("generated_img_path")
            state.generated_pages.append(gen_path if gen_path else "")

        state.pagecontent = new_pagecontent
        state.gen_down = True
        return state

    async def edit_single_page(state: Paper2FigureState) -> Paper2FigureState:
        """
        单页重编辑逻辑：
        - 支持 ref_img (从 state.request.ref_img 获取)
        - 若有 ref_img，使用 gemini_multi_image_edit_async (ref + old_path)
        - 若无 ref_img，使用 generate_or_edit_and_save_image_async (old_path only)
        """
        idx = int(getattr(state, "edit_page_num", -1))
        prompt = (getattr(state, "edit_page_prompt", "") or "").strip()
        if idx < 0:
            raise ValueError("[paper2ppt] edit_page_num 必须是 0-based 且 >=0")
        
        old_path: Optional[str] = None
        if getattr(state, "generated_pages", None) and idx < len(state.generated_pages):
            old_path = state.generated_pages[idx]
        
        if not old_path and idx < len(state.pagecontent or []):
            it = state.pagecontent[idx]
            if isinstance(it, dict):
                old_path = it.get("generated_img_path") or it.get("ppt_img_path") or it.get("img_path")
        
        if not old_path:
            raise ValueError(f"[paper2ppt] 找不到要编辑的页图路径: idx={idx}")

        old_path = _abs_path(old_path)
        result_root = Path(_ensure_result_path(state))
        img_dir = result_root / "ppt_pages"
        img_dir.mkdir(parents=True, exist_ok=True)

        save_path = str((img_dir / f"page_{idx:03d}.png").resolve())
        aspect_ratio = getattr(state, "aspect_ratio", None) or "16:9"
        style = getattr(state.request, "style", None) or "kartoon"

        # 检查 ref_img
        user_ref_img = getattr(state.request, "ref_img", None)
        if user_ref_img:
            user_ref_img = _abs_path(str(user_ref_img))
            if not os.path.exists(user_ref_img):
                log.warning(f"[paper2ppt] Edit page: ref_img not found: {user_ref_img}")
                user_ref_img = None

        log.info(f"[paper2ppt] edit_single_page idx={idx} old={old_path} save={save_path} ref={user_ref_img}")

        if user_ref_img:
            # 有参考图 -> 多图融合
            if prompt:
                full_prompt = (
                    f"Refine this slide image (second image) based on instruction: '{prompt}'. "
                    f"CRITICAL: You MUST strictly match the style of the first image (Reference). "
                    f"Maintain the content layout of the second image but unify the color scheme, background, and design elements "
                    f"to be consistent with the {style} style of the first image."
                )
            else:
                full_prompt = (
                    f"Refine this slide image (second image) to match the style of the first image (Reference). "
                    f"Maintain the content layout of the second image but unify the color scheme, background, and design elements "
                    f"to be consistent with the {style} style of the first image."
                )
            
            await gemini_multi_image_edit_async(
                prompt=full_prompt,
                image_paths=[user_ref_img, old_path],
                save_path=save_path,
                api_url=state.request.chat_api_url,
                api_key=state.request.chat_api_key or os.getenv("DF_API_KEY"),
                model=state.request.gen_fig_model,
                aspect_ratio=aspect_ratio,
            )
        else:
            # 无参考图 -> 单图编辑
            if prompt:
                full_prompt = (
                    f"Beautify this PowerPoint slide based on this instruction: '{prompt}'. "
                    f"Transform the existing design into a high-end, professional {style} style presentation. "
                    f"Enhance the visual aesthetics, layout, and background while preserving the core message."
                )
            else:
                full_prompt = (
                    f"Beautify and re-design this PowerPoint slide. "
                    f"Transform the existing design into a high-end, professional {style} style presentation. "
                    f"Enhance the visual aesthetics, layout, and background while preserving the core message."
                )

            await generate_or_edit_and_save_image_async(
                prompt=full_prompt,
                save_path=save_path,
                aspect_ratio=aspect_ratio,
                api_url=state.request.chat_api_url,
                api_key=state.request.chat_api_key or os.getenv("DF_API_KEY"),
                model=state.request.gen_fig_model,
                image_path=old_path,
                use_edit=True,
            )

        if getattr(state, "generated_pages", None) and idx < len(state.generated_pages):
            state.generated_pages[idx] = save_path
        if idx < len(state.pagecontent or []):
            it = state.pagecontent[idx]
            if isinstance(it, dict):
                it["generated_img_path"] = save_path
                it["edit_prompt"] = prompt
                it["mode"] = "edit_again"

        state.edit_page_prompt = ""
        state.edit_page_num = -1
        return state

    async def export_ppt_assets(state: Paper2FigureState) -> Paper2FigureState:
        """
        导出节点 (同 paper2ppt_parallel)
        """
        if getattr(state, "gen_down", False):
            all_done = getattr(getattr(state, "request", None), "all_edited_down", False)
            if not all_done:
                return state

        result_root = Path(_ensure_result_path(state))
        img_dir = result_root / "ppt_pages"

        if not img_dir.exists():
            raise ValueError(f"[paper2ppt] export_ppt_assets: image dir not found: {img_dir}")

        pdf_path = result_root / "paper2ppt.pdf"
        pptx_path = result_root / "paper2ppt_editable.pptx"

        log.info(f"[paper2ppt] export_ppt_assets: pdf={pdf_path}, pptx={pptx_path}")

        out = await convert_images_dir_to_pdf_and_ppt_api(
            input_dir=str(img_dir),
            output_pdf_path=str(pdf_path),
            output_pptx_path=None,
            api_url=state.request.chat_api_url,
            api_key=state.request.chat_api_key or os.getenv("DF_API_KEY"),
            model=state.request.gen_fig_model,
            use_api_inpaint=False,
        )

        setattr(state, "ppt_pdf_path", out.get("pdf") or str(pdf_path))
        setattr(state, "ppt_pptx_path", None)
        return state

    nodes = {
        "_start_": _start_,
        "generate_pages": generate_pages,
        "edit_single_page": edit_single_page,
        "export_ppt_assets": export_ppt_assets,
        "_end_": lambda state: state,
    }

    edges = [
        ("generate_pages", "export_ppt_assets"),
        ("edit_single_page", "export_ppt_assets"),
        ("export_ppt_assets", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edge("_start_", _route)
    return builder
