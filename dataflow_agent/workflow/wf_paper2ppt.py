from __future__ import annotations

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

from dataflow_agent.toolkits.imtool.req_img import generate_or_edit_and_save_image_async
from dataflow_agent.toolkits.imtool.ppt_tool import convert_images_dir_to_pdf_and_ppt

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
    你给的约定：asset 是 Table 时，通过 asset_ref: "Table 2" 这种字符串标记。
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
        # 兜底：不要因为序列化失败而中断
        return str(d)


async def _make_prompt_for_structured_page(item: Dict[str, Any], style: str, state: Paper2FigureState) -> Tuple[str, Optional[str], bool]:
    """
    根据结构化 page item 生成:
    - prompt
    - image_path (如果是编辑模式)
    - use_edit

    规则：
    1) asset 为空：text2img，用 “json(去asset)” + “根据上述内容生成{style}风格的PPT”
    2) asset 是图片路径：img2img/edit，用 “json(去asset)” + “把这个图作为PPT的一部分...”
    3) asset 是 Table（asset_ref="Table 2"）：先提取 table png（这里先占位），再走 edit
    """
    asset_ref = item.get("asset_ref") or item.get("asset") or item.get("assetRef") or ""
    asset_ref = str(asset_ref).strip() if asset_ref is not None else ""

    # prompt 基础：去掉 asset 字段，避免模型把路径当文本生成
    prompt_dict = dict(item)
    for k in ["asset_ref", "asset", "assetRef", "asset_type", "type"]:
        if k in prompt_dict:
            prompt_dict.pop(k, None)

    base = _serialize_prompt_dict(prompt_dict)

    if not asset_ref:
        prompt = f"{base}\n\n根据上述内容。生成{style}风格的PPT图像。"
        return prompt, None, False

    # table 走占位提取
    if _is_table_asset(asset_ref):
        # 优先使用 item 自己带的表格图（如果调用方已经生成过）
        table_img_path = item.get("table_img_path") or item.get("table_png_path") or ""
        table_img_path = str(table_img_path).strip()

        # 若没有，则调用 table_extractor agent：生成 html->png，并写入 state.table_img_path
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

        if not table_img_path:
            raise ValueError(f"[paper2ppt] 表格提取失败，未得到 table_img_path。asset_ref={asset_ref}")

        image_path = _resolve_asset_path(table_img_path, state)
        prompt = f"{base}\n\n根据上述内容绘制ppt，把这个图作为PPT的一部分。生成{style}风格的PPT。"
        return prompt, image_path, True

    # 默认：当作图片路径，走编辑
    image_path = _resolve_asset_path(asset_ref, state)
    prompt = f"{base}\n\n根据上述内容绘制ppt，把这个图作为PPT的一部分。生成{style}风格的PPT。"
    return prompt, image_path, True


def _resolve_asset_path(asset_ref: str, state: Paper2FigureState) -> str:
    """
    根据 state 解析 asset 引用为绝对路径。

    规则：
    - 为空直接返回 ""；
    - 绝对路径或以 ~ 开头：直接通过 _abs_path 规范化；
    - 相对路径：
        * 优先挂在 state.mineru_root（MinerU 输出根目录）下；
        * 否则挂在 state.result_path 下；
        * 再否则退化为当前工作目录下的相对路径解析（_abs_path）。
    """
    if not asset_ref:
        return ""
    s = str(asset_ref).strip()
    if not s:
        return ""

    p = Path(s)

    # 已经是绝对路径，或者显式使用家目录
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
    支持 pagecontent 直接是图片路径的几种形态：
    - "/abs/xxx.png"
    - {"ppt_img_path": "/abs/xxx.png"}
    - {"img_path": "/abs/xxx.png"}
    - {"path": "/abs/xxx.png"}
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


@register("paper2ppt")
def create_paper2ppt_graph() -> GenericGraphBuilder:  # noqa: N802
    """
    Workflow factory: dfa run --wf paper2ppt

    功能：
    - 若 state.gen_down == False：批量生成/编辑每页 PPT 图，保存到统一目录
    - 若 state.gen_down == True：按 0-based edit_page_num 对已有页面图做二次编辑（edit_page_prompt）
    """
    builder = GenericGraphBuilder(state_model=Paper2FigureState, entry_point="_start_")

    def _start_(state: Paper2FigureState) -> Paper2FigureState:
        _ensure_result_path(state)
        state.pagecontent = state.pagecontent or []
        state.generated_pages = state.generated_pages or []
        # 兼容：有些调用方把 style 放 state.style，而不是 request.style
        if not getattr(state.request, "style", None) and getattr(state, "style", None):
            state.request.style = getattr(state, "style")
        return state

    def _route(state: Paper2FigureState) -> str:
        # gen_down == False: 第一次批量生成
        if not getattr(state, "gen_down", False):
            return "generate_pages"
        # gen_down == True: 进入按页编辑
        return "edit_single_page"

    async def generate_pages(state: Paper2FigureState) -> Paper2FigureState:
        """
        批量生成/编辑页面图：
        - pagecontent 是结构化 list[dict]：按 asset 规则决定 text2img / img2img
        - pagecontent 直接是图片路径列表：逐页用“修改成xxx风格”编辑

        出错时的处理策略：
        - 调用底层图像生成函数最多重试 3 次；
        - 若 3 次都失败，则跳过当前页，继续后面的页；
        - 不中断整个 workflow，最终仍然导出已有页面到 PDF/PPTX。
        """
        import asyncio

        async def _call_image_api_with_retry(coro_factory, retries: int = 3, delay: float = 1.0) -> bool:
            """
            对图像生成/编辑进行最多 retries 次重试。
            - 成功：返回 True
            - 多次失败：返回 False（由上层决定如何处理当前页）
            """
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
                            # sleep 失败不影响后续重试
                            pass
            log.error(f"[paper2ppt] image gen failed after {retries} attempts, skip this page. last_err={last_err}")
            return False

        result_root = Path(_ensure_result_path(state))
        img_dir = result_root / "ppt_pages"
        img_dir.mkdir(parents=True, exist_ok=True)

        style = getattr(state.request, "style", None) or "kartoon"
        aspect_ratio = getattr(state, "aspect_ratio", None) or "16:9"

        # 清空旧数据（避免重复执行堆积）
        state.generated_pages = []
        new_pagecontent: List[Dict[str, Any]] = []

        for idx, item in enumerate(state.pagecontent or []):
            # Case B: pagecontent 本身就是图片路径
            direct_img_path = _extract_image_path_from_pagecontent_item(item)
            is_direct_image_list = bool(direct_img_path) and (
                isinstance(item, str)
                or (isinstance(item, dict) and set(item.keys()).intersection({"ppt_img_path", "img_path", "path", "image_path"}))
            )

            save_path = str((img_dir / f"page_{idx:03d}.png").resolve())

            if is_direct_image_list and (not isinstance(item, dict) or ("title" not in item and "layout_description" not in item)):
                # 规则 2：只做风格化编辑
                image_path = _abs_path(direct_img_path)
                prompt = f"修改成{style}风格"
                log.info(f"[paper2ppt] page={idx} direct image edit: image={image_path}, save={save_path}")

                ok = await _call_image_api_with_retry(
                    lambda: generate_or_edit_and_save_image_async(
                        prompt=prompt,
                        save_path=save_path,
                        aspect_ratio=aspect_ratio,
                        api_url=state.request.chat_api_url,
                        api_key=os.getenv("DF_API_KEY") or state.request.chat_api_key,
                        model=state.request.gen_fig_model,
                        image_path=image_path,
                        use_edit=True,
                    )
                )
                if not ok:
                    # 记录失败信息，但不中断；不把该页加入 generated_pages/new_pagecontent
                    new_pagecontent.append(
                        {
                            "source_img_path": image_path,
                            "generated_img_path": None,
                            "page_idx": idx,
                            "mode": "edit_direct_image_failed",
                            "style": style,
                        }
                    )
                    continue

                state.generated_pages.append(save_path)
                new_pagecontent.append(
                    {
                        "source_img_path": image_path,
                        "generated_img_path": save_path,
                        "page_idx": idx,
                        "mode": "edit_direct_image",
                        "style": style,
                    }
                )
                continue

            # Case A: 结构化页面
            if not isinstance(item, dict):
                log.warning(f"[paper2ppt] page={idx} 非 dict 且非 image path，跳过。item={item}")
                continue

            try:
                prompt, image_path, use_edit = await _make_prompt_for_structured_page(item, style=style, state=state)
            except Exception as e:  # noqa: BLE001
                log.error(f"[paper2ppt] page={idx} prompt 构造失败: {e}")
                # prompt 都构造不出来，直接记录失败并跳过本页
                failed_item = dict(item)
                failed_item["generated_img_path"] = None
                failed_item["page_idx"] = idx
                failed_item["mode"] = "prompt_build_failed"
                failed_item["style"] = style
                failed_item["error"] = str(e)
                new_pagecontent.append(failed_item)
                continue

            log.info(
                f"[paper2ppt] page={idx} structured: use_edit={use_edit}, "
                f"image_path={image_path}, save={save_path}, \n 本次生成的 prompt 为:\n{prompt}"
            )

            ok = await _call_image_api_with_retry(
                lambda: generate_or_edit_and_save_image_async(
                    prompt=prompt,
                    save_path=save_path,
                    aspect_ratio=aspect_ratio,
                    api_url=state.request.chat_api_url,
                    api_key=os.getenv("DF_API_KEY") or state.request.chat_api_key,
                    model=state.request.gen_fig_model,
                    image_path=image_path,
                    use_edit=use_edit,
                )
            )
            if not ok:
                # 记录失败信息，但不中断；不写入 generated_pages
                failed_item = dict(item)
                failed_item["generated_img_path"] = None
                failed_item["page_idx"] = idx
                failed_item["mode"] = "generate_failed" if not use_edit else "edit_failed"
                failed_item["style"] = style
                new_pagecontent.append(failed_item)
                continue

            state.generated_pages.append(save_path)
            # 透传原始结构化信息，并写入生成结果
            out_item = dict(item)
            out_item["generated_img_path"] = save_path
            out_item["page_idx"] = idx
            out_item["mode"] = "edit" if use_edit else "generate"
            out_item["style"] = style
            new_pagecontent.append(out_item)

        state.pagecontent = new_pagecontent
        state.gen_down = True
        return state

    async def edit_single_page(state: Paper2FigureState) -> Paper2FigureState:
        """
        gen_down == True 时的路径：
        通过 edit_page_num(0-based) + edit_page_prompt 对已经生成好的某一页做二次编辑。
        """
        idx = int(getattr(state, "edit_page_num", -1))
        prompt = (getattr(state, "edit_page_prompt", "") or "").strip()
        if idx < 0:
            raise ValueError("[paper2ppt] edit_page_num 必须是 0-based 且 >=0")
        if not prompt:
            raise ValueError("[paper2ppt] edit_page_prompt 不能为空")

        # 取出原图路径：优先 generated_pages，其次 pagecontent[i].generated_img_path
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

        ts = int(time.time())
        save_path = str((img_dir / f"page_{idx:03d}_edit_{ts}.png").resolve())
        aspect_ratio = getattr(state, "aspect_ratio", None) or "16:9"

        log.info(f"[paper2ppt] edit_single_page idx={idx} old={old_path} save={save_path}")

        await generate_or_edit_and_save_image_async(
            prompt=prompt,
            save_path=save_path,
            aspect_ratio=aspect_ratio,
            api_url=state.request.chat_api_url,
            api_key=os.getenv("DF_API_KEY") or state.request.chat_api_key,
            model=state.request.gen_fig_model,
            image_path=old_path,
            use_edit=True,
        )

        # 回写路径
        if getattr(state, "generated_pages", None) and idx < len(state.generated_pages):
            state.generated_pages[idx] = save_path
        if idx < len(state.pagecontent or []):
            it = state.pagecontent[idx]
            if isinstance(it, dict):
                it["generated_img_path"] = save_path
                it["edit_prompt"] = prompt
                it["mode"] = "edit_again"

        # 清理编辑请求（可选）
        state.edit_page_prompt = ""
        state.edit_page_num = -1
        return state

    async def export_ppt_assets(state: Paper2FigureState) -> Paper2FigureState:
        """
        最终导出节点：
        - 使用 ppt_tool.convert_images_dir_to_pdf_and_ppt
          将 result_path/ppt_pages 中的页面图导出为 PDF 和可编辑 PPTX。
        """
        result_root = Path(_ensure_result_path(state))
        img_dir = result_root / "ppt_pages"

        if not img_dir.exists():
            raise ValueError(f"[paper2ppt] export_ppt_assets: image dir not found: {img_dir}")

        pdf_path = result_root / "paper2ppt.pdf"
        pptx_path = result_root / "paper2ppt_editable.pptx"

        log.info(
            f"[paper2ppt] export_ppt_assets: images_dir={img_dir}, "
            f"pdf={pdf_path}, pptx={pptx_path}"
        )

        out = convert_images_dir_to_pdf_and_ppt(
            input_dir=str(img_dir),
            output_pdf_path=str(pdf_path),
            output_pptx_path=str(pptx_path),
        )

        # 可选：把导出结果路径挂到 state 上，方便后续使用
        setattr(state, "ppt_pdf_path", out.get("pdf") or str(pdf_path))
        setattr(state, "ppt_pptx_path", out.get("pptx") or str(pptx_path))

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
