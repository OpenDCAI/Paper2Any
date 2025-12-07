"""
paper2technical workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-07 23:36:51

1. 在 **TOOLS** 区域定义需要暴露给 Prompt 的前置工具
2. 在 **NODES**  区域实现异步节点函数 (await-able)
3. 在 **EDGES**  区域声明有向边
4. 最后返回 builder.compile() 或 GenericGraphBuilder
"""

from __future__ import annotations
import json
import time
from pathlib import Path

from dataflow_agent.state import Paper2FigureState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_graph_agent, create_react_agent, create_simple_agent
from dataflow_agent.toolkits.tool_manager import get_tool_manager
from dataflow_agent.toolkits.imtool.bg_tool import local_tool_for_svg_render
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

@register("paper2technical")
def create_paper2technical_graph() -> GenericGraphBuilder:  # noqa: N802
    """
    Workflow factory: dfa run --wf paper2technical
    """
    # 使用 Paper2FigureState，复用其中的 paper_file / paper_idea / fig_desc 等字段，
    # 这里不做图像生成和抠图，只负责“技术路线图”的 SVG + PPT 逻辑。
    builder = GenericGraphBuilder(
        state_model=Paper2FigureState,
        entry_point="_start_",        # 入口统一为 _start_，再由路由函数分发
    )

    # ----------------------------------------------------------------------
    # TOOLS (pre_tool definitions)
    # ----------------------------------------------------------------------
    # 1) 提供给 paper_idea_extractor 的 PDF 内容（标题 + 前几页正文）
    @builder.pre_tool("paper_content", "paper_idea_extractor")
    def _get_paper_content(state: Paper2FigureState):
        """
        前置工具: 读取论文 PDF 的标题和前若干页内容，供 paper_idea_extractor 节点使用。

        - 作用: 为大模型提供足够的上下文，让其抽取论文中的技术路线/实验流程关键信息。
        - 输出: 一个字符串，包含论文标题 + 前若干页文本。
        """
        import fitz  # PyMuPDF
        import PyPDF2

        pdf_path = state.paper_file
        if not pdf_path:
            log.warning("paper_file 为空，无法读取 PDF 内容")
            return ""

        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                paper_title = reader.metadata.get("/Title", "Unknown Title")
        except Exception:
            paper_title = "Unknown Title"

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            log.error(f"打开 PDF 失败: {e}")
            return f"The title of the paper is {paper_title}"

        text_parts: list[str] = []
        # 读取前 10 页内容，通常技术路线、整体框架会在前几页出现
        for page_idx in range(min(10, len(doc))):
            page = doc.load_page(page_idx)
            text_parts.append(page.get_text("text") or "")

        content = "\n".join(text_parts).strip()
        final_text = (
            f"The title of the paper is {paper_title}\n\n"
            f"Here are the first 10 pages of the paper:\n{content}"
        )
        log.info("paper_content 提取完成")
        return final_text

    # 2) 提供给技术路线图描述生成器的“论文核心想法/摘要”
    @builder.pre_tool("paper_idea", "technical_route_desc_generator")
    def _get_paper_idea(state: Paper2FigureState):
        """
        前置工具: 为 technical_route_desc_generator 节点暴露论文的核心想法摘要。

        - 在 PDF 模式下，该摘要由 paper_idea_extractor 节点写入 state.paper_idea。
        - 在 TEXT 模式下，可以直接由调用方事先把概要写入 state.paper_idea。
        """
        return state.paper_idea or ""

    # ----------------------------------------------------------------------

    # ==============================================================
    # NODES
    # ==============================================================
    async def paper_idea_extractor_node(state: Paper2FigureState) -> Paper2FigureState:
        """
        节点 1: 从 PDF 中抽取论文的核心思想 / 技术路线相关信息

        - 只在 input_type == "PDF" 时作为入口节点被调用。
        - 基于 pre_tool("paper_content") 提供的标题 + 前若干页内容，
          调用专门的 agent（例如 paper_idea_extractor）生成摘要。
        - 该摘要用于后续技术路线图描述生成。

        输入:
            state.paper_file : 论文 PDF 路径
        输出:
            state.paper_idea : 论文核心思想 / 技术路线要点摘要
            state.agent_results["paper_idea_extractor"] : agent 原始输出
        """
        agent = create_simple_agent("paper_idea_extractor", tool_manager=get_tool_manager())
        state = await agent.execute(state=state)
        return state

    async def technical_route_desc_generator_node(state: Paper2FigureState) -> Paper2FigureState:
        """
        节点 2: 技术路线图描述生成器

        - 根据论文摘要（PDF 模式）或用户直接提供的文本描述（TEXT 模式），
          生成“技术路线/实验流程”的结构化自然语言描述或 JSON。
        - 典型内容包括: 各阶段实验步骤、模块之间的依赖关系、输入输出数据流等。

        输入:
            - PDF 模式: state.paper_idea 由 paper_idea_extractor 填充
            - TEXT 模式: 可以事先把文本写入 state.paper_idea 或其他字段
        输出:
            - 建议: 在 agent 内把结果存到 state.fig_desc 或 state.agent_results["technical_route_desc_generator"]
        """
        agent = create_react_agent(name = "technical_route_desc_generator", max_retries=4)
        state = await agent.execute(state=state)

        # --------------------------------------------------------------
        # 将 LLM 生成的 SVG 源码渲染为实际图像文件，并写入 state.result_path
        # --------------------------------------------------------------
        svg_code = getattr(state, "figure_tec_svg_content", None)
        if svg_code:
            # 默认落盘目录（当 state.result_path 无效或不存在时回退）
            default_base_dir = Path("dataflow_agent/tmps/paper2technical").resolve()

            raw_result_path = getattr(state, "result_path", None)
            if raw_result_path:
                p = Path(raw_result_path)
                # 如果原路径存在且是目录，则直接使用该目录
                if p.exists() and p.is_dir():
                    base_dir = p
                else:
                    # 否则尝试将其视为文件路径或目录路径，取父目录/本身
                    base_dir = p.parent if p.suffix else p
                    # 若该目录不存在，则回退到默认目录
                    if not base_dir.exists():
                        base_dir = default_base_dir
            else:
                base_dir = default_base_dir

            base_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            output_path = str((base_dir / f"technical_route_{timestamp}.png").resolve())

            try:
                png_path = local_tool_for_svg_render({
                    "svg_code": svg_code,
                    "output_path": output_path,
                })
                # 将最终图像路径写回 state.svg_img_path
                log.critical(f'[state.svg_img_path]: {state.svg_img_path}')
                state.svg_img_path = png_path
            except Exception as e:
                # 渲染失败时仅记录日志，避免打断整体 workflow
                log.error(f"technical_route_desc_generator_node: SVG 渲染失败: {e}")

        return state

    # async def svg_code_generator_node(state: Paper2FigureState) -> Paper2FigureState:
    #     """
    #     节点 3: SVG 技术路线图代码生成器

    #     - 基于上一节点生成的技术路线描述（自然语言/结构化 JSON），
    #       生成一份完整的 SVG 源代码，用于表示技术路线图。
    #     - 该节点只负责“代码层面”的生成，不涉及 PNG/JPG 等位图。

    #     输入:
    #         - state.fig_desc 或 state.agent_results["technical_route_desc_generator"] 中的内容
    #     输出:
    #         - 建议: 在 agent 内把 SVG 字符串写入 state.agent_results["technical_svg"]["svg_code"]
    #         - 此节点本身不强制写文件，文件写入可以在后续节点统一处理。
    #     """
    #     agent = create_graph_agent("technical_svg_generator", tool_manager=get_tool_manager())
    #     state = await agent.execute(state=state, use_agent=True)
    #     return state

    async def svg_fragment_miner_node(state: Paper2FigureState) -> Paper2FigureState:
        """
        节点 4: SVG 结构切分 / 小图块生成 (占位)

        目标:
        - 将整张技术路线 SVG 图，切分成若干“小 SVG”或逻辑块（例如每个阶段一个块），
          以便后续在 PPT 中灵活排版。
        - 实际实现可以对接 MinerU 或自研的 SVG 解析逻辑。

        当前实现(占位):
        - 仅从 agent_results 中读取 svg_code，简单记录到 state.temp_data 中，
          不做真正的图像/结构切分，留作 TODO。
        """

        return state

    async def technical_ppt_generator_node(state: Paper2FigureState) -> Paper2FigureState:
        """
        节点 5: 基于技术路线 SVG / 片段生成 PPT

        - 根据前面步骤生成的 SVG 代码或 svg_fragments，
          生成一份或多份 PPT 幻灯片，用于展示技术路线图。
        - 与 paper2figure 的 PPT 生成不同:
          - 这里不依赖位图图片和抠图，不需要图像背景去除模型；
          - 完全围绕“技术路线图”的结构信息进行排版。

        当前实现(占位):
        - 仅在 state.result_path 下生成一个空的 pptx 文件，记录路径到 state.ppt_path。
        - 具体的 SVG 渲染 & 排版逻辑留作后续补充。
        """
        from pptx import Presentation

        # 输出目录
        output_dir = Path(state.result_path or "./outputs/paper2technical")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        ppt_path = output_dir / f"technical_route_{timestamp}.pptx"

        prs = Presentation()
        # 占位: 新建一个空白幻灯片，后续可在此基础上根据 SVG 结构添加 shapes/textbox 等
        blank_slide_layout = prs.slide_layouts[6]
        prs.slides.add_slide(blank_slide_layout)

        prs.save(str(ppt_path))
        state.ppt_path = str(ppt_path)
        log.info(f"technical_ppt_generator_node: 占位 PPT 已生成: {ppt_path}")

        return state

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================

    def set_entry_node(state: Paper2FigureState) -> str:
        """
        路由函数: 根据输入类型选择技术路线工作流的入口节点。

        - input_type == "PDF"  : 从 PDF 中抽取论文想法，先走 paper_idea_extractor
        - input_type == "TEXT" : 直接使用调用方提供的文本描述，跳过 PDF 抽取，
                                 从 technical_route_desc_generator 开始
        其他值:
        - 认为是不合法输入，直接结束工作流。
        """
        input_type = getattr(state.request, "input_type", "PDF")
        if input_type == "PDF":
            log.critical("paper2technical: 进入 PDF 流程 (paper_idea_extractor)")
            return "paper_idea_extractor"
        elif input_type == "TEXT":
            log.critical("paper2technical: 进入 TEXT 流程 (technical_route_desc_generator)")
            return "technical_route_desc_generator"
        else:
            log.error(f"paper2technical: Invalid input type: {input_type}")
            return "_end_"

    nodes = {
        "_start_": lambda state: state,
        "paper_idea_extractor": paper_idea_extractor_node,
        "technical_route_desc_generator": technical_route_desc_generator_node,
        # "svg_code_generator": svg_code_generator_node,  # 预留节点，当前未实现
        "svg_fragment_miner": svg_fragment_miner_node,
        "technical_ppt_generator": technical_ppt_generator_node,
        "_end_": lambda state: state,  # 终止节点
    }

    # ------------------------------------------------------------------
    # EDGES  (从节点 A 指向节点 B)
    # ------------------------------------------------------------------
    edges = [
        # PDF 流程: 先抽想法，再生成技术路线描述
        ("paper_idea_extractor", "technical_route_desc_generator"),
        # PDF/TEXT 后续流程共用: 描述 -> 结构切分 -> PPT
        ("technical_route_desc_generator", "svg_fragment_miner"),
        ("svg_fragment_miner", "technical_ppt_generator"),
        ("technical_ppt_generator", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edge("_start_", set_entry_node)
    return builder
