from __future__ import annotations

"""
paper2ppt 工作流封装。

拆分为三个 API：
- run_paper2page_content_wf_api: 只跑 paper2page_content，侧重解析/生成 pagecontent
- run_paper2page_content_refine_wf_api: 只跑 paper2page_content，用于基于反馈修订 outline
- run_paper2ppt_wf_api: 只跑 paper2ppt，基于已有 pagecontent 生成 PPT 资源
- run_paper2ppt_full_pipeline: full pipeline，串联 paper2page_content + paper2ppt
"""

import json
import time
from pathlib import Path
from typing import Any, List

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureState
from dataflow_agent.toolkits.multimodaltool.mineru_tool import _shrink_markdown
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow import run_workflow

from fastapi_app.schemas import Paper2PPTRequest, Paper2PPTResponse

log = get_logger(__name__)


def _to_serializable(obj: Any):
    """递归将对象转成可 JSON 序列化结构"""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(i) for i in obj]
    if hasattr(obj, "__dict__"):
        return _to_serializable(obj.__dict__)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _ensure_result_path_for_full(email: str | None) -> Path:
    """
    为 full pipeline 统一一个根输出目录：
    outputs/{email or 'default'}/paper2ppt/<timestamp>/
    """
    project_root = get_project_root()
    ts = int(time.time())
    code = email or "default"
    base_dir = (project_root / "outputs" / code / "paper2ppt" / str(ts)).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _init_state_from_request(
    req: Paper2PPTRequest,
    result_path: Path | None = None,
    override_pagecontent: list[dict] | None = None,
) -> Paper2FigureState:
    """
    从 Paper2PPTRequest 初始化 Paper2FigureState，兼容三种场景：
    - full pipeline: 需要根据 input_type / input_content 设置 paper_file / text_content 等；
    - pagecontent-only: 只关心 PDF/TEXT/PPT 解析，不一定马上生成 PPT 资源；
    - ppt-only: 直接从外部提供的 pagecontent / result_path 生成 PPT。
    """
    state = Paper2FigureState(
        messages=[],
        agent_results={},
        request=req,
    )

    # 根据场景设置输入
    input_type = (req.input_type or "").upper()
    input_content = req.input_content or ""

    # PDF / TEXT / FIGURE 的解析与 wf_paper2page_content 的约定保持一致
    if input_type == "PDF":
        state.paper_file = input_content
    elif input_type in ("PPT", "PPTX"):
        # 对于 PPT/PPTX，我们也统一挂在 paper_file 上，wf_paper2page_content 中会走 ppt_to_images 路径
        state.paper_file = input_content
    elif input_type == "TEXT":
        # 纯文本场景：直接作为 text_content
        state.text_content = input_content
    elif input_type == "TOPIC":
        state.text_content = input_content
    else:
        log.warning(f"[paper2ppt] Unknown input_type on init_state: {input_type}")

    # 兼容样式等控制参数
    state.aspect_ratio = req.aspect_ratio
    state.style = req.style

    # 覆盖 pagecontent（主要用于只跑 paper2ppt 的场景）
    if override_pagecontent is not None:
        try:
            state.pagecontent = list(override_pagecontent)
        except TypeError:
            log.warning("[paper2ppt] override_pagecontent 不是 list[dict]，将忽略。")

    # 统一 result_path（如果调用方显式指定，则优先使用）
    if result_path is not None:
        state.result_path = str(result_path)

    return state


def _try_load_existing_mineru_markdown(result_root: Path) -> tuple[str, str]:
    """
    从既有的 result_root 中尝试加载 MinerU 解析的 markdown。
    期望路径形态：<pdf_stem>/auto/<pdf_stem>.md

    Returns:
        (mineru_output, mineru_root_dir)
    """
    try:
        candidates = list(result_root.glob("*/auto/*.md"))
    except Exception:
        candidates = []

    if not candidates:
        return "", ""

    md_path = candidates[0]
    try:
        md = md_path.read_text(encoding="utf-8")
    except Exception:
        return "", ""

    mineru_output = _shrink_markdown(md, max_h1=8, max_chars=30_000)
    return mineru_output, str(md_path.parent.resolve())


async def run_paper2page_content_wf_api(req: Paper2PPTRequest, result_path: Path | None = None) -> Paper2PPTResponse:
    """
    只执行 paper2page_content 工作流，主要用于从 PDF / PPTX / TEXT
    中解析出结构化的 pagecontent。

    - 输入：Paper2PPTRequest（需提供 input_type / input_content 等）
    - 输出：Paper2PPTResponse，其中：
        - success: 是否成功
        - pagecontent: 解析后的页面内容（结构化列表）
        - result_path: 本次 workflow 使用的统一输出目录
    """
    # 统一 result_path：若调用方希望自定义，可在 req 中扩展字段；目前统一使用 email 路径
    if result_path is None:
        result_root = _ensure_result_path_for_full(req.email)
    else:
        result_root = result_path

    state = _init_state_from_request(req, result_path=result_root)

    log.info(f"[paper2page_content_wf_api] start, result_path={state.result_path}, input_type={req.input_type}")
    if req.use_long_paper:
        final_state: Paper2FigureState = await run_workflow("paper2page_content_for_long_paper", state)
    else:    
        final_state: Paper2FigureState = await run_workflow("paper2page_content", state)
    # 提取结果
    pagecontent = final_state["pagecontent"] or []
    log.critical(f"[paper2page_content_wf_api] pagecontent={pagecontent}")
    result_path = final_state["result_path"] or str(result_root)

    # 构造响应：目前 Paper2PPTResponse 只有 success，占位扩展字段通过动态属性注入
    resp_data: dict[str, Any] = {
        "success": True,
        "pagecontent": pagecontent,
        "result_path": result_path,
    }

    return Paper2PPTResponse(**resp_data)


async def run_paper2page_content_refine_wf_api(
    req: Paper2PPTRequest,
    pagecontent: list[dict],
    outline_feedback: str,
    result_path: Path | None = None,
) -> Paper2PPTResponse:
    """
    只执行 paper2page_content 工作流，用于基于反馈修订已有 outline。
    """
    if result_path is None:
        result_root = _ensure_result_path_for_full(req.email)
    else:
        result_root = result_path

    state = _init_state_from_request(req, result_path=result_root)
    state.pagecontent = list(pagecontent or [])
    state.outline_feedback = outline_feedback or ""
    if not getattr(state, "minueru_output", ""):
        mineru_output, mineru_root = _try_load_existing_mineru_markdown(result_root)
        if mineru_output:
            state.minueru_output = mineru_output
        if mineru_root:
            state.mineru_root = mineru_root

    log.info(f"[paper2page_content_refine_wf_api] start, result_path={state.result_path}")
    final_state: Paper2FigureState = await run_workflow("paper2page_content", state)

    pagecontent = final_state["pagecontent"] or []
    result_path = final_state["result_path"] or str(result_root)

    resp_data: dict[str, Any] = {
        "success": True,
        "pagecontent": pagecontent,
        "result_path": result_path,
    }
    return Paper2PPTResponse(**resp_data)


async def run_paper2ppt_wf_api(
    req: Paper2PPTRequest,
    pagecontent: list[dict] | None = None,
    result_path: str | None = None,
    get_down: bool | None = None,
    edit_page_num: int | None = None,
    edit_page_prompt: str | None = None,
    auto_fill_generated_pages: bool = True,
) -> Paper2PPTResponse:
    """
    只执行 paper2ppt 工作流。通常用于：
    - 外部已经有 pagecontent（可能来自前端编辑好的 JSON），现在只想生成 PPT 资源；
    - 或者已经跑过一次 paper2page_content，希望在同一 result_path 下重复生成。

    参数：
    - req: Paper2PPTRequest
    - pagecontent: 若提供，则覆盖 state.pagecontent
    - result_path: 若提供，则强制使用该输出目录；否则 wf_paper2ppt 自行决定
    - get_down: 对应 workflow 的 state.gen_down
        * False/None：走 generate_pages（批量生成）
        * True：走 edit_single_page（按页二次编辑）
    - edit_page_num/edit_page_prompt: 仅在 get_down=True 时生效
    - auto_fill_generated_pages: 编辑模式下，是否从 result_path/ppt_pages 扫描 page_*.png 回填 state.generated_pages
    """
    base_dir: Path | None = None
    if result_path:
        base_dir = Path(result_path).expanduser().resolve()
        base_dir.mkdir(parents=True, exist_ok=True)

    state = _init_state_from_request(
        req,
        result_path=base_dir,
        override_pagecontent=pagecontent,
    )

    # 映射 get_down -> workflow state.gen_down
    if get_down is not None:
        state.gen_down = bool(get_down)

    # 编辑模式参数注入
    if bool(getattr(state, "gen_down", False)):
        if edit_page_num is not None:
            state.edit_page_num = int(edit_page_num)
        if edit_page_prompt is not None:
            state.edit_page_prompt = str(edit_page_prompt)

        if auto_fill_generated_pages and base_dir is not None:
            try:
                img_dir = base_dir / "ppt_pages"
                if img_dir.exists():
                    imgs = sorted(img_dir.glob("page_*.png"))
                    state.generated_pages = [str(p.resolve()) for p in imgs]
            except Exception as e:  # pragma: no cover
                log.warning(f"[paper2ppt_wf_api] auto_fill_generated_pages failed: {e}")
         
    #  mineru_root 写死
    state.mineru_root = f"{base_dir}/input/auto"

    # 尝试回填 mineru_output (markdown)，供 table_extractor 等使用
    try:
        md_dir = Path(state.mineru_root)
        if md_dir.exists():
            md_files = list(md_dir.glob("*.md"))
            if md_files:
                # 默认取第一个 md
                md_path = md_files[0]
                raw_md = md_path.read_text(encoding="utf-8")
                state.mineru_output = _shrink_markdown(raw_md, max_h1=8, max_chars=30_000)
                log.info(f"[paper2ppt_wf_api] Loaded mineru_output from {md_path}, len={len(state.mineru_output)}")
            else:
                log.warning(f"[paper2ppt_wf_api] No .md file found in {md_dir}")
        else:
            log.warning(f"[paper2ppt_wf_api] mineru_root dir not found: {md_dir}")
    except Exception as e:
        log.warning(f"[paper2ppt_wf_api] Failed to load mineru_output: {e}")

    log.info(
        f"[paper2ppt_wf_api] start, result_path={getattr(state, 'result_path', None)}, "
        f"pagecontent_len={len(getattr(state, 'pagecontent', []) or [])}"
    )

    # final_state: Paper2FigureState = await run_workflow("paper2ppt_parallel", state)
    log.critical(f'[wa_paper2ppt] req.ref_img 路径 {req.ref_img}')
    final_state: Paper2FigureState = await run_workflow("paper2ppt_parallel_consistent_style", state)

    # 提取关键输出
    ppt_pdf_path = getattr(final_state, "ppt_pdf_path", "")
    ppt_pptx_path = getattr(final_state, "ppt_pptx_path", "")
    final_pagecontent = getattr(final_state, "pagecontent", []) or []
    final_result_path = getattr(final_state, "result_path", result_path or "")

    resp_data: dict[str, Any] = {
        "success": True,
        "ppt_pdf_path": str(ppt_pdf_path) if ppt_pdf_path else "",
        "ppt_pptx_path": str(ppt_pptx_path) if ppt_pptx_path else "",
        "pagecontent": final_pagecontent,
        "result_path": final_result_path,
    }

    return Paper2PPTResponse(**resp_data)


async def run_paper2ppt_full_pipeline(req: Paper2PPTRequest) -> Paper2PPTResponse:
    """
    full pipeline：
    - 先跑 paper2page_content：根据 PDF/PPT/TEXT 解析 pagecontent
    - 再跑 paper2ppt：基于 pagecontent 生成 PPT 资源（PDF + PPTX）

    入参：
    - Paper2PPTRequest（需至少提供 input_type / input_content）

    出参：
    - Paper2PPTResponse：
        - success
        - ppt_pdf_path
        - ppt_pptx_path
        - pagecontent
        - result_path
    """
    # 统一输出根目录，两个 workflow 共用
    result_root = _ensure_result_path_for_full(req.email)

    # ---------- 第一步：paper2page_content ----------
    state_pc = _init_state_from_request(req, result_path=result_root)
    log.info(
        f"[paper2ppt_full_pipeline] step1 paper2page_content, "
        f"result_path={state_pc.result_path}, input_type={req.input_type}, use_long_paper={req.use_long_paper}"
    )
    if req.use_long_paper:
        state_pc = await run_workflow("paper2page_content_for_long_paper", state_pc)
    else:
        state_pc = await run_workflow("paper2page_content", state_pc)

    pagecontent = getattr(state_pc, "pagecontent", []) or []
    # 确保 result_path 一致
    final_result_path = getattr(state_pc, "result_path", str(result_root))

    # ---------- 第二步：paper2ppt ----------
    # 复用 state_pc 继续执行 paper2ppt，避免丢失中间状态
    log.info(
        f"[paper2ppt_full_pipeline] step2 paper2ppt, "
        f"result_path={final_result_path}, pagecontent_len={len(pagecontent)}"
    )
    state_pc.pagecontent = pagecontent
    state_pc.result_path = final_result_path

    state_pp: Paper2FigureState = await run_workflow("paper2ppt", state_pc)

    ppt_pdf_path = getattr(state_pp, "ppt_pdf_path", "")
    ppt_pptx_path = getattr(state_pp, "ppt_pptx_path", "")
    final_pagecontent = getattr(state_pp, "pagecontent", []) or []

    resp_data: dict[str, Any] = {
        "success": True,
        "ppt_pdf_path": str(ppt_pdf_path) if ppt_pdf_path else "",
        "ppt_pptx_path": str(ppt_pptx_path) if ppt_pptx_path else "",
        "pagecontent": final_pagecontent,
        "result_path": final_result_path,
    }

    return Paper2PPTResponse(**resp_data)
