"""
icongen workflow (Image Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Only generates the figure (and layout template), stops before SAM/PPT generation.
Used for preview/confirmation step.

1. Idea Extraction (PDF/Text)
2. Prompt Generation
3. Image Generation (Content + Layout Template)
"""

from __future__ import annotations
import asyncio
import json
import os
import time
from pathlib import Path

from dataflow_agent.state import Paper2FigureState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_graph_agent, create_react_agent
from dataflow_agent.toolkits.imtool.req_img import generate_or_edit_and_save_image_async
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

import fitz
import PyPDF2

log = get_logger(__name__)

def _ensure_result_path(state: Paper2FigureState) -> str:
    raw = getattr(state, "result_path", None)
    if raw:
        return raw
    root = get_project_root()
    ts = int(time.time())
    base_dir = (root / "outputs" / "paper2figure" / str(ts)).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    state.result_path = str(base_dir)
    return state.result_path

@register("paper2fig_image_only")
def create_p2fig_image_only_graph() -> GenericGraphBuilder:
    builder = GenericGraphBuilder(state_model=Paper2FigureState,
                                  entry_point="_start_")

    # ----------------------------------------------------------------------
    # TOOLS (pre_tool definitions)
    # ----------------------------------------------------------------------
    @builder.pre_tool("paper_content", "paper_idea_extractor")
    def _get_abstract_intro(state: Paper2FigureState):
        try:
            with open(state.paper_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                paper_title = reader.metadata.get('/Title', 'Unknown Title')
        except Exception:
            paper_title = "Unknown Title"

        file_path = state.paper_file
        pdf_document = fitz.open(file_path)

        text = ""
        for page_num in range(min(10, len(pdf_document))):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")

        content = text.strip()
        final_text = (
            f"The title of the paper is {paper_title}\n\n"
            f"Here's first ten page content: {content}"
        )
        log.info(f"{final_text}")
        return final_text
    
    @builder.pre_tool("paper_idea", "figure_desc_generator")
    def _get_paper_idea(state: Paper2FigureState):
        # 根据请求语言添加指令
        lang = getattr(state.request, "language", "zh")
        log.critical(f'[image_only]: lang {lang}')
        lang_instruction = ""
        if lang == "zh":
            lang_instruction = "\n\nIMPORTANT: The text content inside the generated figure MUST be in Chinese (Simplified Chinese). Please ensure all labels, titles, and descriptions in the figure description are in Chinese."
        else:
            lang_instruction = "\n\nIMPORTANT: The text content inside the generated figure MUST be in English. Please ensure all labels, titles, and descriptions in the figure description are in English."
            
        # 如果是图片编辑模式，state.paper_idea 可能被设为了 "Image Edit Mode"，此时不应追加指令到 meaningless string
        # 但如果是 TEXT 模式，则需要。如果是 PDF 模式，paper_idea 是提取出来的摘要。
        
        return state.paper_idea + lang_instruction

    # ==============================================================
    # NODES
    # ==============================================================
    async def paper_idea_extractor_node(state: Paper2FigureState) -> Paper2FigureState:
        paper_idea_extractor = create_graph_agent("paper_idea_extractor")
        state = await paper_idea_extractor.execute(state, use_agent=True)
        return state
    
    async def figure_desc_generator_node(state: Paper2FigureState) -> Paper2FigureState:
        figure_desc_generator = create_react_agent("figure_desc_generator",
                                                    max_retries=5,
                                                    model_name=getattr(state.request, "fig_desc_model", "gpt-5.1"))
        state = await figure_desc_generator.execute(state, use_agent=True)
        return state

    async def figure_generator_node(state: Paper2FigureState) -> Paper2FigureState:
        """
        调用图像模型生成/编辑单张图（内容 + 布局模版），增加轻量 retry + 降低超时时间。

        参考 pdf2ppt_optimized 中 _call_image_api_with_retry 的做法，这里做一个本地版：
        - 单次 HTTP 调用 timeout 控制在 60s（通过参数下传）
        - 最多重试 3 次，每次失败打印详细日志
        """
        # 安全获取 figure_desc_generator 的结果，可能因 input_type=FIGURE 跳过了该节点
        fd_gen_result = state.agent_results.get("figure_desc_generator")
        prompt = ""
        if fd_gen_result:
            prompt = fd_gen_result.get("results", {}).get("fig_desc", "")
        
        safe_prompt = json.dumps(prompt, ensure_ascii=False) if prompt else ""

        edit_prompt = state.request.get("edit_prompt")
        image_path = state.request.get("prev_image")

        final_prompt = edit_prompt if image_path else safe_prompt

        log.info(
            f"[p2f_image_only] final_prompt(len={len(final_prompt)}), "
            f"edit_prompt_len={len(edit_prompt or '')}, image_path={image_path}, "
            f"safe_prompt_len={len(safe_prompt or '')}"
        )

        result_root = Path(_ensure_result_path(state))
        result_root.mkdir(parents=True, exist_ok=True)

        # 1) Generate Content Image
        fig_name = f"fig_{int(time.time())}.png"
        save_path = str(result_root / fig_name)

        api_url = state.request.chat_api_url
        api_key = state.request.chat_api_key or os.getenv("DF_API_KEY")
        model = state.request.gen_fig_model
        aspect_ratio = state.aspect_ratio

        async def _call_image_api_with_retry(coro_factory, retries: int = 3, delay: float = 2.0) -> bool:
            """
            本地轻量级 retry：仅负责 httpx / 网络级错误或服务超时。
            失败时不会抛到外层，而是返回 False，由 workflow 决定如何处理。
            """
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    log.info(f"[p2f_image_only] image api attempt {attempt}/{retries} ...")
                    await coro_factory()
                    log.info("[p2f_image_only] image api succeed")
                    return True
                except Exception as e:
                    last_err = e
                    log.error(f"[p2f_image_only] image api failed attempt {attempt}/{retries}: {e}")
                    if attempt < retries:
                        await asyncio.sleep(delay)
            log.error(f"[p2f_image_only] image api failed after {retries} attempts: {last_err}")
            return False

        async def _gen_image():
            await generate_or_edit_and_save_image_async(
                prompt=final_prompt,
                save_path=save_path,
                aspect_ratio=aspect_ratio,
                api_url=api_url,
                api_key=api_key,
                model=model,
                image_path=image_path,
                use_edit=True if image_path else False,
                timeout=60,
                resolution='2K'
            )

        ok = await _call_image_api_with_retry(_gen_image)
        if not ok:
            # 将失败信息写入 state，避免直接 500
            state.agent_results["gen_img_error"] = {
                "msg": "image generation failed after retries",
                "save_path": save_path,
            }
            log.error("[p2f_image_only] image generation failed, see previous logs for details")
            return state

        state.agent_results["gen_img"] = {"path": save_path}
        state.fig_draft_path = save_path

        return state

    # ==============================================================
    # Registry
    # ==============================================================
    def set_entry_node(state: Paper2FigureState) -> str:
        if state.request.input_type == "PDF":
            return "paper_idea_extractor"
        elif state.request.input_type == "TEXT":
            return "figure_desc_generator"
        elif state.request.input_type == "FIGURE":
            return "figure_generator"
        else:
            log.error(f"Invalid input type: {state.request.input_type}. Only PDF, TEXT and FIGURE are supported.")
            return "_end_"

    def _init_result_path(state: Paper2FigureState) -> Paper2FigureState:
        _ensure_result_path(state)
        return state

    nodes = {
        '_start_': _init_result_path,
        "paper_idea_extractor": paper_idea_extractor_node,
        "figure_desc_generator": figure_desc_generator_node,
        "figure_generator": figure_generator_node,
        '_end_': lambda state: state,
    }

    edges = [
        ("paper_idea_extractor", "figure_desc_generator"),
        ("figure_desc_generator", "figure_generator"),
        ("figure_generator", "_end_"), # End after image generation
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edge("_start_", set_entry_node)
    return builder
