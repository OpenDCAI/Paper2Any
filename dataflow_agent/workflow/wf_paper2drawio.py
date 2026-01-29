"""
paper2drawio workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AI 驱动的 draw.io 图表生成工作流

支持三种模式:
1. 从 PDF 论文生成图表
2. 从文本描述生成图表
3. 编辑现有图表

工作流程：
1. _start_ → 路由到对应模式
2. paper_content_extractor → 提取 PDF 内容 (PDF 模式)
3. diagram_planner → 规划图表结构
4. drawio_xml_generator → 生成 draw.io XML
5. _end_ → 输出结果
"""

from __future__ import annotations
import os
import time
from pathlib import Path
import json
from typing import Dict, Any

from dataflow_agent.state import Paper2DrawioState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_simple_agent, create_react_agent, create_vlm_agent
from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.drawio_tools import wrap_xml, validate_xml, export_drawio_png

log = get_logger(__name__)


def _ensure_result_path(state: Paper2DrawioState) -> str:
    """确保输出目录存在"""
    raw = getattr(state, "result_path", None)
    if raw:
        return raw

    ts = int(time.time())
    base_dir = Path(f"outputs/paper2drawio/{ts}").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    state.result_path = str(base_dir)
    return state.result_path


@register("paper2drawio")
def create_paper2drawio_graph() -> GenericGraphBuilder:
    """
    Paper2Drawio Workflow: AI 驱动的图表生成

    命令: dfa run --wf paper2drawio
    """
    builder = GenericGraphBuilder(
        state_model=Paper2DrawioState,
        entry_point="_start_"
    )

    # ==================== PRE-TOOLS ====================

    @builder.pre_tool("paper_content", "diagram_planner")
    def _get_paper_content(state: Paper2DrawioState) -> str:
        """提取 PDF 内容"""
        pdf_path = state.paper_file
        if not pdf_path:
            return state.text_content or ""

        try:
            import fitz
            doc = fitz.open(pdf_path)
            text_parts = []
            for page_idx in range(min(15, len(doc))):
                page = doc.load_page(page_idx)
                text_parts.append(page.get_text("text") or "")
            return "\n".join(text_parts).strip()
        except Exception as e:
            log.error(f"PDF 解析失败: {e}")
            return state.text_content or ""

    @builder.pre_tool("text_content", "diagram_planner")
    def _get_text_content(state: Paper2DrawioState) -> str:
        return state.text_content or ""

    @builder.pre_tool("diagram_type", "diagram_planner")
    def _get_diagram_type(state: Paper2DrawioState) -> str:
        return state.request.diagram_type

    @builder.pre_tool("language", "diagram_planner")
    def _get_diagram_language(state: Paper2DrawioState) -> str:
        return state.request.language or "en"

    @builder.pre_tool("diagram_plan", "drawio_xml_generator")
    def _get_diagram_plan(state: Paper2DrawioState) -> str:
        plan = state.diagram_plan or ""
        if isinstance(plan, (dict, list)):
            try:
                return json.dumps(plan, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                return str(plan)
        return plan

    @builder.pre_tool("diagram_style", "drawio_xml_generator")
    def _get_diagram_style(state: Paper2DrawioState) -> str:
        return state.request.diagram_style or "default"

    @builder.pre_tool("language", "drawio_xml_generator")
    def _get_generator_language(state: Paper2DrawioState) -> str:
        return state.request.language or "en"

    @builder.pre_tool("validation_feedback", "drawio_xml_generator")
    def _get_validation_feedback(state: Paper2DrawioState) -> str:
        return state.validation_feedback or ""

    @builder.pre_tool("current_xml", "diagram_editor")
    def _get_current_xml(state: Paper2DrawioState) -> str:
        return state.drawio_xml or ""

    @builder.pre_tool("edit_instruction", "diagram_editor")
    def _get_edit_instruction(state: Paper2DrawioState) -> str:
        return state.request.edit_instruction or ""

    @builder.pre_tool("diagram_xml", "diagram_vlm_validator")
    def _get_diagram_xml_for_validation(state: Paper2DrawioState) -> str:
        return state.drawio_xml or ""

    @builder.pre_tool("diagram_type", "diagram_vlm_validator")
    def _get_diagram_type_for_validation(state: Paper2DrawioState) -> str:
        return state.request.diagram_type or "auto"

    # ==================== NODES ====================

    def _init_node(state: Paper2DrawioState) -> Paper2DrawioState:
        """初始化节点"""
        _ensure_result_path(state)
        return state

    async def diagram_planner_node(state: Paper2DrawioState) -> Paper2DrawioState:
        """规划图表结构"""
        agent = create_simple_agent(
            name="diagram_planner",
            model_name=state.request.model,
            temperature=0.3,
        )
        state = await agent.execute(state=state)
        return state

    async def drawio_xml_generator_node(state: Paper2DrawioState) -> Paper2DrawioState:
        """生成 draw.io XML（可选 VLM 验证 + 反馈再生）"""
        base_dir = Path(_ensure_result_path(state))

        agent = create_react_agent(
            name="drawio_xml_generator",
            max_retries=state.request.max_retries or 3,
            model_name=state.request.model,
            temperature=0.0,
            max_tokens=16384,
            parser_type="text",
        )

        enable_vlm = bool(state.request.enable_vlm_validation)
        if not enable_vlm:
            env_flag = os.getenv("PAPER2DRAWIO_ENABLE_VLM_VALIDATION", "false").lower()
            enable_vlm = env_flag in ("1", "true", "yes", "on")
        max_vlm_rounds = state.request.vlm_validation_max_retries or 3

        def _format_validation_feedback(result: Dict[str, Any]) -> str:
            if not isinstance(result, dict):
                return ""
            issues = result.get("issues") or []
            suggestions = result.get("suggestions") or []
            lines = ["DIAGRAM VISUAL VALIDATION FAILED", ""]
            if issues:
                lines.append("Issues:")
                for issue in issues:
                    if isinstance(issue, dict):
                        itype = issue.get("type", "issue")
                        desc = issue.get("description", "")
                        lines.append(f"- [{itype}] {desc}")
                    else:
                        lines.append(f"- {issue}")
                lines.append("")
            if suggestions:
                lines.append("Suggestions:")
                for s in suggestions:
                    lines.append(f"- {s}")
                lines.append("")
            return "\n".join(lines).strip()

        async def _run_vlm_validation(
            state: Paper2DrawioState,
            png_path: str,
        ) -> Dict[str, Any]:
            schema = {
                "valid": "boolean",
                "issues": [
                    {
                        "type": "overlap|edge_routing|text|layout|rendering|arrow_direction",
                        "severity": "critical|warning",
                        "description": "string",
                    }
                ],
                "suggestions": ["string"],
            }
            vlm_agent = create_vlm_agent(
                name="diagram_vlm_validator",
                vlm_mode="understanding",
                model_name=state.request.vlm_model or state.request.model,
                chat_api_url=state.request.chat_api_url,
                parser_type="json",
                parser_config={
                    "schema": schema,
                    "required_fields": ["valid"],
                },
                additional_params={"input_image": png_path},
            )

            # Execute VLM agent using current state (diagram_xml pre-tool will read state)
            state = await vlm_agent.execute(state)
            result = state.agent_results.get("diagram_vlm_validator", {}).get("results", {})
            return result if isinstance(result, dict) else {"raw": result}

        last_feedback = state.validation_feedback or ""

        for attempt in range(max(1, max_vlm_rounds if enable_vlm else 1)):
            if last_feedback:
                state.validation_feedback = last_feedback

            state = await agent.execute(state=state)

            # 保存 XML 文件
            xml_code = state.drawio_xml
            if xml_code:
                is_valid, errors = validate_xml(xml_code)
                if not is_valid:
                    log.warning(f"XML 验证警告: {errors}")

                full_xml = wrap_xml(xml_code)
                timestamp = int(time.time())
                xml_path = base_dir / f"diagram_{timestamp}_try{attempt + 1}.drawio"
                xml_path.write_text(full_xml, encoding="utf-8")
                state.output_xml_path = str(xml_path)
                log.info(f"XML 已保存: {xml_path}")

            if not enable_vlm:
                break

            if not xml_code:
                log.warning("[paper2drawio] empty XML, skip VLM validation")
                break

            png_path = base_dir / f"diagram_{int(time.time())}_try{attempt + 1}.png"
            ok, msg = export_drawio_png(xml_code, str(png_path))
            if not ok:
                log.warning(f"[paper2drawio] drawio PNG export skipped: {msg}")
                break

            state.validation_png_path = str(png_path)

            vlm_result = await _run_vlm_validation(state, str(png_path))
            valid = bool(vlm_result.get("valid") is True)
            if valid:
                state.validation_feedback = ""
                log.info(f"[paper2drawio] VLM validation passed on attempt {attempt + 1}")
                break

            last_feedback = _format_validation_feedback(vlm_result)
            state.validation_feedback = last_feedback
            log.warning(f"[paper2drawio] VLM validation failed on attempt {attempt + 1}")

        return state

    async def diagram_editor_node(state: Paper2DrawioState) -> Paper2DrawioState:
        """编辑现有图表"""
        agent = create_react_agent(
            name="diagram_editor",
            max_retries=state.request.max_retries or 3,
            model_name=state.request.model,
            temperature=0.0,
            parser_type="text",
        )
        state = await agent.execute(state=state)
        return state

    # ==================== ROUTING ====================

    def _route_entry(state: Paper2DrawioState) -> str:
        """入口路由"""
        edit_mode = bool(state.request.edit_instruction and state.drawio_xml)
        if edit_mode:
            return "diagram_editor"
        return "diagram_planner"

    # ==================== BUILD GRAPH ====================

    nodes = {
        "_start_": _init_node,
        "diagram_planner": diagram_planner_node,
        "drawio_xml_generator": drawio_xml_generator_node,
        "diagram_editor": diagram_editor_node,
        "_end_": lambda state: state,
    }

    edges = [
        ("diagram_planner", "drawio_xml_generator"),
        ("drawio_xml_generator", "_end_"),
        ("diagram_editor", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges)
    builder.add_conditional_edge("_start_", _route_entry)

    return builder
