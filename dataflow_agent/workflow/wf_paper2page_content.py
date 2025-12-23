from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Dict, Any
import re

from dataflow_agent.state import Paper2FigureState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_react_agent
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

from dataflow_agent.toolkits.imtool.mineru_tool import run_mineru_pdf_extract, _shrink_markdown

log = get_logger(__name__)

def _ensure_result_path(state: Paper2FigureState) -> str:
    """
    参考 wf_paper2figure_with_sam.py 的做法：
    统一本次 paper2page_content workflow 的根输出目录：
    - 如果 state.result_path 已存在（通常由调用方传入），直接使用；
    - 否则：使用 get_project_root() / "outputs" / "paper2page_content" / <timestamp>，
      并写回 state.result_path，后续节点共享同一目录。
    """
    raw = getattr(state, "result_path", None)
    if raw:
        return raw

    root = get_project_root()
    ts = int(time.time())
    base_dir = (root / "outputs" / "paper2page_content" / str(ts)).resolve()
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


@register("paper2page_content")
def create_paper2page_content_graph() -> GenericGraphBuilder:  # noqa: N802
    """
    Workflow factory: dfa run --wf paper2page_content
    """
    builder = GenericGraphBuilder(state_model=Paper2FigureState, entry_point="_start_")

    # ----------------------------------------------------------------------
    # TOOLS (pre_tool definitions)
    # ----------------------------------------------------------------------
    @builder.pre_tool("minueru_output", "outline_agent")
    def _get_mineru_markdown(state: Paper2FigureState):
        return state.minueru_output or ""

    @builder.pre_tool("text_content", "outline_agent")
    def _get_text_content(state: Paper2FigureState):
        return state.text_content or ""

    # ==============================================================
    # NODES
    # ==============================================================
    def _start_(state: Paper2FigureState) -> Paper2FigureState:
        _ensure_result_path(state)
        # 清理/初始化 paper2ppt 专用字段（避免复用 state 时脏数据串场）
        state.minueru_output = state.minueru_output or ""
        state.text_content = state.text_content or ""
        state.pagecontent = state.pagecontent or []
        return state

    async def parse_pdf_pages(state: Paper2FigureState) -> Paper2FigureState:
        """
        PDF: MinerU 解析 -> 读取 markdown 全文 -> 写入 state.minueru_output

        目录约定（与 MinerU 实际行为对齐）：
        - 传入的输出根目录为 result_root = state.result_path
        - MinerU 会在其下创建:
            <pdf_stem>/auto/<pdf_stem>.md
            <pdf_stem>/auto/images/*.jpg
        - 我们将 state.mineru_root 指向实际承载 md 和 images 的 auto 目录，
          这样后续 asset_ref="images/xxx.jpg" 能解析到正确路径。
        """
        paper_pdf_path = Path(_abs_path(state.paper_file))
        if not paper_pdf_path.exists():
            log.error(f"[paper2page_content] PDF 文件不存在: {paper_pdf_path}")
            state.minueru_output = ""
            return state

        # 统一本次 workflow 的根输出目录
        result_root = Path(_ensure_result_path(state))
        result_root.mkdir(parents=True, exist_ok=True)

        pdf_stem = paper_pdf_path.stem
        paper_dir = result_root / pdf_stem           # e.g. outputs/2506.02454v1
        auto_dir = paper_dir / "auto"                # e.g. outputs/2506.02454v1/auto

        # 若不存在 MinerU 结果，则触发 MinerU：输出到 result_root 下
        # MinerU 内部会创建 <pdf_stem>/auto 结构
        if not auto_dir.exists():
            try:
                run_mineru_pdf_extract(str(paper_pdf_path), str(result_root), "modelscope")
            except Exception as e:
                log.error(f"[paper2page_content] run_mineru_pdf_extract 失败: {e}")
                state.minueru_output = ""
                return state

        # 重新计算一次 auto_dir，防止 MinerU 在内部调整目录结构
        auto_dir = (result_root / pdf_stem / "auto").resolve()
        markdown_path = auto_dir / f"{pdf_stem}.md"
        if not markdown_path.exists():
            log.error(f"[paper2page_content] Markdown 文件不存在: {markdown_path}")
            state.minueru_output = ""
            return state

        try:
            md = markdown_path.read_text(encoding="utf-8")
        except Exception as e:
            log.error(f"[paper2page_content] 读取 markdown 失败: {markdown_path}, err={e}")
            md = ""
        # Avoid passing overly long markdown to downstream agents.
        state.minueru_output = _shrink_markdown(md, max_h1=8, max_chars=30_000)
        # 记录 MinerU 输出根目录 = 实际承载 md 与 images 的 auto 目录
        state.mineru_root = str(auto_dir)
        log.info(f"[paper2page_content] minueru_output : {state.minueru_output[:100]} ")
        return state

    async def prepare_text_input(state: Paper2FigureState) -> Paper2FigureState:
        """
        TEXT: 直接进入 outline agent 前，把文本放到 state.text_content
        """
        # 兼容：优先 paper2ppt 专用 text_content；如果外部通过 request.target 传入文本，也做兜底
        if not state.text_content:
            state.text_content = getattr(state.request, "target", "") or ""
        return state

    async def ppt_to_images(state: Paper2FigureState) -> Paper2FigureState:
        """
        PPT/PPTX: 转成每页图片，写入 state.pagecontent:
          [{"ppt_img_path": "/abs/slide_001.png"}, ...]
        注意：这里的 pagecontent 仅作为 outline agent 的输入材料，最终 pagecontent 会被 agent 改写。
        """
        ppt_path = Path(_abs_path(state.paper_file))
        if not ppt_path.exists():
            log.error(f"[paper2page_content] PPT 文件不存在: {ppt_path}")
            state.pagecontent = []
            return state

        output_dir = Path(_ensure_result_path(state)) / "ppt_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 策略：优先 soffice 转 pdf，再 pdf2image
        pdf_path = output_dir / f"{ppt_path.stem}.pdf"
        if not pdf_path.exists():
            cmd = (
                f'soffice --headless --convert-to pdf --outdir "{output_dir}" "{ppt_path}"'
            )
            # 这里不能用 execute_command 工具（在 workflow runtime 内执行），因此用 os.system 兜底；

            ret = os.system(cmd)
            if ret != 0:
                log.error(
                    f"[paper2page_content] soffice 转 pdf 失败(ret={ret}). "
                    f"请确认部署机器安装了 libreoffice/soffice。cmd={cmd}"
                )
                state.pagecontent = []
                return state

        if not pdf_path.exists():
            log.error(f"[paper2page_content] soffice 转出的 pdf 不存在: {pdf_path}")
            state.pagecontent = []
            return state

        try:
            from pdf2image import convert_from_path
        except Exception as e:
            log.error(f"[paper2page_content] 缺少 pdf2image 依赖，无法将 pdf 转图片: {e}")
            state.pagecontent = []
            return state

        try:
            slide_imgs = convert_from_path(str(pdf_path))
        except Exception as e:
            log.error(f"[paper2page_content] pdf2image 转换失败: {e}")
            state.pagecontent = []
            return state

        page_items: List[Dict[str, Any]] = []
        for i, img in enumerate(slide_imgs):
            img_path = output_dir / f"slide_{i:03d}.png"
            try:
                img.save(img_path, "PNG")
            except Exception as e:
                log.error(f"[paper2page_content] 保存 slide png 失败: {img_path}, err={e}")
                continue
            page_items.append({"ppt_img_path": str(img_path.resolve())})

        state.pagecontent = page_items
        return state

    async def outline_agent(state: Paper2FigureState) -> Paper2FigureState:
        """
        Outline agent 骨架：你后续实现 agent 逻辑，产出 state.pagecontent(list[dict])。
        这里仅负责创建并执行 agent，然后返回 state。
        """
        agent = create_react_agent(
            name="outline_agent",
            temperature=0.1,
            max_retries=5,
            parser_type="json",
        )
        state = await agent.execute(state=state)
        return state

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    def _route_input(state: Paper2FigureState) -> str:
        t = getattr(state.request, "input_type", None) or getattr(state, "input_type", None) or ""
        t = str(t).upper().strip()
        if t == "PDF":
            log.critical("走 PDF 路径")
            return "parse_pdf_pages"
        if t == "TEXT":
            log.critical("走 TEXT 路径")
            return "prepare_text_input"
        if t in ["PPT", "PPTX"]:
            log.critical("走 PPT 路径")
            return "ppt_to_images"
        log.error(f"[paper2page_content] Invalid input_type: {t}")
        return "_end_"

    nodes = {
        "_start_": _start_,
        "parse_pdf_pages": parse_pdf_pages,
        "prepare_text_input": prepare_text_input,
        "ppt_to_images": ppt_to_images,
        "outline_agent": outline_agent,
        "_end_": lambda state: state,
    }

    edges = [
        ("parse_pdf_pages", "outline_agent"),
        ("prepare_text_input", "outline_agent"),
        ("ppt_to_images", "_end_"),
        ("outline_agent", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edge("_start_", _route_input)
    return builder
