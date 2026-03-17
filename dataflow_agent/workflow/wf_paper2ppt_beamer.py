"""
paper2ppt_beamer workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pagecontent（来自 paper2page_content）→ 每页单独生成 Beamer → 每页编译成 PDF → 合并为一份 PDF → _end_。
调用方需先跑 paper2page_content，再传入带 pagecontent / result_path / mineru_root 的 state。
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from dataclasses import replace

from dataflow_agent.state import Paper2PptBeamerRequest, Paper2PptBeamerState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.toolkits.p2vtool.p2v_tool import (
    compile_tex,
    merge_pdfs,
    is_overfull_warning,
    is_table_asset,
)
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


@register("paper2ppt_beamer_pagecontent")
def create_paper2ppt_beamer_graph() -> GenericGraphBuilder:
    """
    Workflow factory: dfa run --wf paper2ppt_beamer_pagecontent
    pagecontent → 每页 pagecontent_to_beamer + compile → merge_slides → _end_
    """
    builder = GenericGraphBuilder(
        state_model=Paper2PptBeamerState,
        entry_point="_start_",
    )

    def _request_language(state: Paper2PptBeamerState) -> str:
        req = state.request
        if isinstance(req, dict):
            return req.get("language", "en")
        return getattr(req, "language", "en")

    @builder.pre_tool("pagecontent", "p2b_pagecontent_to_beamer")
    def get_pagecontent(state: Paper2PptBeamerState):
        pc = getattr(state, "pagecontent", None)
        return pc or []

    @builder.pre_tool("output_language", "p2b_pagecontent_to_beamer")
    def get_output_language(state: Paper2PptBeamerState):
        language_map = {"en": "English", "zh": "Chinese"}
        return language_map.get(_request_language(state), "English")

    @builder.pre_tool("pdf_images_working_dir", "p2b_pagecontent_to_beamer")
    def get_pdf_images_working_dir(state: Paper2PptBeamerState):
        mineru_root = getattr(state, "mineru_root", "") or ""
        if mineru_root:
            return str(Path(mineru_root).expanduser().resolve())
        return ""

    # ----------------------------------------------------------------------
    # NODES
    # ----------------------------------------------------------------------

    async def p2b_pagecontent_to_beamer(
        state: Paper2PptBeamerState,
    ) -> Paper2PptBeamerState:
        from dataflow_agent.agentroles import create_simple_agent

        pages = getattr(state, "pagecontent", None) or []
        result_path = Path(getattr(state, "result_path", "") or ".").expanduser().resolve()
        output_dir = result_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 未得到有效 Beamer 代码（如 LLM 500/usage_limit）或编译失败时重试
        max_error_retries = 3
        retry_delay_seconds = 3  # 无有效代码时延迟再试，缓解限流/usage_limit
        max_warning_fixes = 2

        p2b_agent = create_simple_agent(
            name="p2b_pagecontent_to_beamer",
            model_name="gpt-5-codex",
            temperature=0.1,
            parser_type="json",
        )
        debug_agent = create_simple_agent(
            name="p2v_beamer_code_debug",
            model_name="gpt-5-codex",
            temperature=0.1,
            parser_type="json",
        )

        per_page_beamer_paths: list[str] = []
        per_page_pdf_paths: list[str] = []
        full_pagecontent = list(pages)

        # 并行度，避免 API 限流
        max_concurrent_pages = 4
        semaphore = asyncio.Semaphore(max_concurrent_pages)

        async def process_one_page(
            i: int,
            one_page: dict,
        ) -> tuple[int, str, str | None]:
            """处理单页，返回 (页索引, tex 路径, pdf 路径或 None)。"""
            async with semaphore:
                log.info("生成第 %s/%s 页 Beamer 并编译", i + 1, len(pages))
                one_page = dict(one_page)
                # asset_ref 为 Table_1 等时直接忽略，不跑 table_extractor
                asset_ref = one_page.get("asset_ref") or one_page.get("asset") or ""
                asset_ref = str(asset_ref).strip() if asset_ref else ""
                if asset_ref and is_table_asset(asset_ref):
                    one_page["asset_ref"] = None

                page_state = replace(
                    state,
                    pagecontent=[one_page],
                    beamer_code_path="",
                    is_beamer_wrong=False,
                    is_beamer_warning=False,
                    code_debug_result="",
                )
                page_tex = output_dir / f"page_{i}.tex"
                is_wrong = True
                is_warning = False
                code_debug_result = ""

                # ---------- Error 重试（含未得到有效代码，如 LLM 500/usage_limit）----------
                for error_attempt in range(max_error_retries):
                    page_state = await p2b_agent.execute(state=page_state)
                    if not getattr(page_state, "beamer_code_path", ""):
                        log.warning("第 %s 页未得到 beamer 代码（可能 LLM 限流/500），第 %s/%s 次重试", i + 1, error_attempt + 1, max_error_retries)
                        if error_attempt < max_error_retries - 1:
                            await asyncio.sleep(retry_delay_seconds)
                        continue
                    shutil.copy(page_state.beamer_code_path, page_tex)
                    try:
                        is_wrong, is_warning, code_debug_result = compile_tex(str(page_tex))
                    except Exception as e:
                        is_wrong, is_warning, code_debug_result = True, True, str(e)
                        log.warning("第 %s 页编译异常: %s", i + 1, e)
                    if not is_wrong:
                        break
                    log.warning("第 %s 页编译 error，第 %s/%s 次重新生成", i + 1, error_attempt + 1, max_error_retries)

                if is_wrong:
                    log.warning("第 %s 页经 %s 次重试仍有 error，跳过", i + 1, max_error_retries)
                    return (i, str(page_tex), None)

                # ---------- Warning 修复（Overfull）----------
                if is_warning and is_overfull_warning(code_debug_result):
                    for fix_attempt in range(max_warning_fixes):
                        page_state.beamer_code_path = str(page_tex)
                        page_state.is_beamer_wrong = is_wrong
                        page_state.is_beamer_warning = is_warning
                        page_state.code_debug_result = code_debug_result
                        page_state.pre_tool_results = {
                            "beamer_code": page_tex.read_text(encoding="utf-8"),
                            "code_debug_result": code_debug_result,
                        }
                        page_state = await debug_agent.execute(page_state)
                        is_wrong = getattr(page_state, "is_beamer_wrong", True)
                        is_warning = getattr(page_state, "is_beamer_warning", False)
                        code_debug_result = getattr(page_state, "code_debug_result", "")
                        if is_wrong:
                            break
                        if not is_warning or not is_overfull_warning(code_debug_result):
                            break
                        log.info("第 %s 页 Overfull warning，第 %s/%s 次修复", i + 1, fix_attempt + 1, max_warning_fixes)

                pdf_path = page_tex.with_suffix(".pdf")
                pdf_str = str(pdf_path) if pdf_path.exists() else None
                return (i, str(page_tex), pdf_str)

        # 并行处理所有页，保持页序
        results = await asyncio.gather(
            *[process_one_page(i, one_page) for i, one_page in enumerate(pages)],
            return_exceptions=True,
        )
        # 按页索引排序，保证与原始页序一致（gather 完成顺序可能乱序）
        tex_by_index: dict[int, str] = {}
        pdf_by_index: dict[int, str] = {}
        for r in results:
            if isinstance(r, Exception):
                log.exception("某页处理异常: %s", r)
                continue
            i, tex_path, pdf_path = r
            tex_by_index[i] = tex_path
            if pdf_path:
                pdf_by_index[i] = pdf_path
        for idx in range(len(pages)):
            if idx in tex_by_index:
                per_page_beamer_paths.append(tex_by_index[idx])
            if idx in pdf_by_index:
                per_page_pdf_paths.append(pdf_by_index[idx])

        state.pagecontent = full_pagecontent
        state.per_page_beamer_paths = per_page_beamer_paths
        state.per_page_pdf_paths = per_page_pdf_paths
        log.info("每页生成与编译完成: %s 个 tex, %s 个 pdf", len(per_page_beamer_paths), len(per_page_pdf_paths))
        return state

    def merge_slides_node(state: Paper2PptBeamerState) -> Paper2PptBeamerState:
        log.info("开始执行 merge_slides_node")
        pdf_paths = getattr(state, "per_page_pdf_paths", None) or []
        if not pdf_paths:
            log.warning("无每页 PDF，无法合并")
            return state
        result_path = Path(getattr(state, "result_path", "") or ".").expanduser().resolve()
        merged_path = result_path / "output" / "merged.pdf"
        state.ppt_path = merge_pdfs(pdf_paths, merged_path)
        log.info("合并完成: %s", state.ppt_path)
        return state

    def _start_(state: Paper2PptBeamerState) -> Paper2PptBeamerState:
        return state

    def _end_(state: Paper2PptBeamerState) -> Paper2PptBeamerState:
        log.info(f"The ppt_path is {state.ppt_path}")
        return state

    nodes = {
        "_start_": _start_,
        "p2b_pagecontent_to_beamer": p2b_pagecontent_to_beamer,
        "merge_slides": merge_slides_node,
        "_end_": _end_,
    }

    edges = [
        ("_start_", "p2b_pagecontent_to_beamer"),  
        ("p2b_pagecontent_to_beamer", "merge_slides"),
        ("merge_slides", "_end_"),
    ]   

    builder.add_nodes(nodes).add_edges(edges)
    return builder

if __name__ == "__main__":
    import asyncio

    result_path = Path("outputs/default/paper2ppt/1772284521/input")
    pagecontent = [{'title': 'DataFlow: LLM驱动的统一数据准备与工作流自动化框架', 'layout_description': '整页居中布局，仅包含标题、副标题和汇报人信息。标题大号加粗居中，副标题为论文完整英文标题置于标题下方，作者及汇报人信息放在页面下方居中，不放任何图表。', 'key_points': ['DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI', '作者：Hao Liang 等，机构：Peking University 等', '汇报人：XXX'], 'asset_ref': None}, {'title': '研究背景与问题：LLM时代的数据准备挑战', 'layout_description': '上方简要小结背景，两栏布局：左侧为要点式文本，右侧为对比表格示意区（可用表格或示意图说明现有系统特点对比），下方一行突出本工作目标。', 'key_points': ['LLM 发展依赖大规模、高质量、语义丰富的数据准备流程，涉及合成、精炼、过滤和领域特定转换。', '当前实践以临时脚本和松散工作流为主，缺乏统一抽象、原子算子与可优化、可重现的数据流表示。', '传统大数据引擎（Spark、Dask、Hadoop）对模型闭环、GPU高效批处理和文本语义操作支持不足，工程负担巨大。', '现有数据准备系统如 NeMo Curator、Data-Juicer 主要聚焦提取与过滤，对多步生成与语义精炼的模型闭环工作流支持有限。', '研究问题：如何构建一个以 LLM 为一等公民、可编程、可复用、可扩展的统一数据准备框架？'], 'asset_ref': None}, {'title': 'DataFlow 概览：目标、定位与整体架构', 'layout_description': '上部用一两行文字概述 DataFlow 作为统一系统的定位，中间居中放系统架构示意图（核心执行引擎+管线+CLI+Agent+生态），下方采用两列要点：左列列出六大设计目标，右列说明系统范围与工作流。', 'key_points': ['系统定位：面向多领域 LLM 数据准备的统一、自动化系统，以 LLM 驱动合成与精炼为核心，覆盖文本、数学推理、代码、Text-to-SQL、Agentic RAG 和大规模知识抽取。', '设计目标：易用性（PyTorch 风格、IDE 友好）、可扩展性（模块化算子与管线）、统一范式（跨领域抽象）、性能效率（不牺牲 SOTA 表现）、智能自动化（Agent 解释自然语言意图）、开源与社区生态。', '核心组件：全局存储抽象、统一 LLM Serving、算子库、Prompt 模板、管线 Zoo，以及基于 Python 包的扩展生态 DataFlow-Ecosystem。', '用户控制层：命令行工具链（CLI）用于脚本化执行，DataFlow-Agent 将自然语言规格翻译为可执行管线并迭代调试。', '输出形态：高质量、任务对齐的数据集，可直接用于下游 LLM 训练与评测。'], 'asset_ref': 'images/ba397b4c85a1c1bd0022e9dd145db42f9ab3f956df48273d92694b3cad820a48.jpg'}, {'title': '框架设计：存储抽象、接口层次与算子生态', 'layout_description': '左右分栏布局：左侧重点用流程步骤和要点解释全局存储抽象与算子交互模式，右侧上方放算子执行模式示意图，下方用简短 bullet 解释层次化接口（Serving/Operator/Prompt/Pipeline）。', 'key_points': ['全局存储抽象：以表格化键值结构统一表示指令、回答、CoT、评分与元数据，DataFlowStorage 提供 backend 无关的 read()/write() 接口，算子只面向逻辑视图。', '算子执行模式：遵循统一的 read–transform–write 流程，可以在不修改内部逻辑的前提下重排、复用与批处理；默认实现基于 Pandas，支持 JSON/JSONL/CSV/Parquet 等格式。', '统一 LLM Serving API：generate_from_input(user_inputs, system_prompt, json_schema) 将本地引擎（vLLM、SGLang）与在线服务（ChatGPT、Gemini）统一抽象，屏蔽批处理、重试与限流细节。', '层次化接口：算子定义可复用数据变换单元，Prompt 模板声明输入渲染和输出结构约束，管线将算子按显式依赖组合成多阶段工作流，可编译验证与优化。', '算子与生态：近 200 个可复用算子，分为生成、评估、过滤、精炼四大类，搭配 90+ Prompt 模板，并通过 Python 包实现 DataFlow-Extensions，形成可插拔、社区驱动的 DataFlow-Ecosystem。'], 'asset_ref': 'images/31c09ede8e57c6b583ac2663f145fd113a811470772998506d502e3bb5ebf3ea.jpg'}, {'title': 'DataFlow-Agent 与实验结果：自动化管线构建与性能提升', 'layout_description': '上半部分两列：左列介绍 DataFlow-Agent 的角色设计与智能管线推荐，右列概括六大用例管线（文本、数学、代码、Text-to-SQL、Agentic RAG、知识抽取）。下半部分用要点强调核心实验结果与性能增益。', 'key_points': ['DataFlow-Agent 作为编排层：基于 AgentRoles 理解自然语言规格，执行算子合成、管线规划与迭代验证，可自动构造和调试新的数据准备工作流。', '智能管线推荐：面向目标任务与数据源，自动选择合适的算子组合与模板，降低工程门槛，加速原型迭代。', '六大代表性用例：文本数据准备、数学推理数据、代码处理、Text-to-SQL 数据生成、Agentic RAG 数据构造、从网页/PDF 的大规模知识抽取。', '实验结果（部分）：Text-to-SQL 管线在仅使用 <0.1M 样本的情况下，相比 250 万样本 SynSQL 提升约 +3% 执行准确率；代码管线在多个基准上平均提升超过 7%。', '统一数据集效果：将文本、数学、代码数据融合为 DataFlowInstruct-10K，仅 10K 样本即可让 Qwen2-base/Qwen2.5-base 超过在 100 万 Infinity-Instruct 上训练的同规模模型，并接近对应 Instruct 模型性能。', '整体结论：DataFlow 管线在六个场景中普遍带来 1–3 分甚至更高的性能增益，验证了统一抽象与 LLM 驱动数据合成在质量与数据效率上的优势。'], 'asset_ref': 'images/80627ebb10b377adbb7f5c301c785fa17fd0ba4b8a49b0942f308faba59aa249.jpg'}, {'title': '总结与致谢', 'layout_description': '上方 concise 总结本文贡献，中间用要点强调框架价值与未来方向，下方居中放置“致谢”字样及感谢合作者和数据/代码开源社区，不放图表。', 'key_points': ['工作总结：提出 DataFlow——一个以 LLM 为中心、具备可编程算子与 PyTorch 风格管线抽象的统一数据准备框架，系统性提升了 LLM 数据构造的可复用性、可重现性与可扩展性。', '技术贡献：构建近 200 个算子与六大高性能模板管线，提供统一 LLM Serving、全局存储、层次化接口与扩展生态，并通过 DataFlow-Agent 实现自然语言到可执行管线的自动化映射。', '实证结论：在文本、数学、代码、Text-to-SQL、Agentic RAG、知识抽取等多场景中，DataFlow 生成的数据显著提升下游 LLM 性能和数据效率，部分场景超过精心人工或专用合成数据集。', '未来方向：进一步扩展多模态与多语言算子与管线，强化分布式执行与调优能力，推动 DataFlow 成为数据中心 AI 时代社区共享的统一数据准备协议。', '致谢：感谢合作者、开源社区（模型、数据、工具）及相关项目团队对本工作的支持与启发。'], 'asset_ref': None}]

    graph_builder = create_paper2ppt_beamer_graph().build()
    state = Paper2PptBeamerState(
        request=Paper2PptBeamerRequest(language="zh"),
        pagecontent=pagecontent,
        result_path=str(result_path),
        mineru_root=str(result_path),
    )
    state = asyncio.run(graph_builder.ainvoke(state))
