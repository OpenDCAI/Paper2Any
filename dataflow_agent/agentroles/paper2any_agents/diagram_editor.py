"""
DiagramEditor agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Edit existing draw.io XML based on user instructions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from dataflow_agent.state import Paper2DrawioState
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.agentroles.cores.base_agent import BaseAgent, ValidatorFunc
from dataflow_agent.toolkits.drawio_tools import validate_xml, sanitize_cells_xml

log = get_logger(__name__)


@register("diagram_editor")
class DiagramEditor(BaseAgent):
    """Edit existing draw.io XML."""

    def __init__(self, **kwargs):
        kwargs["parser_type"] = "text"
        super().__init__(**kwargs)

    @property
    def role_name(self) -> str:
        return "diagram_editor"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_diagram_editor"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_diagram_editor"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "current_xml": pre_tool_results.get("current_xml", ""),
            "edit_instruction": pre_tool_results.get("edit_instruction", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "current_xml": "",
            "edit_instruction": "",
        }

    def get_react_validators(self) -> List[ValidatorFunc]:
        return [
            self._validator_has_mxcell,
            self._validator_no_markdown,
            self._validator_no_xml_comments,
            self._validator_xml_valid,
        ]

    @staticmethod
    def _extract_xml_text(content: str, parsed_result: Dict[str, Any]) -> str:
        if isinstance(parsed_result, dict):
            return (
                parsed_result.get("text", "")
                or parsed_result.get("xml", "")
                or parsed_result.get("drawio_xml", "")
                or ""
            )
        return content or ""

    @classmethod
    def _validator_has_mxcell(
        cls,
        content: str,
        parsed_result: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        xml_text = cls._extract_xml_text(content, parsed_result).strip()
        if not xml_text or "<mxCell" not in xml_text:
            return False, "输出必须包含 mxCell 元素，且仅包含 mxCell 片段。"
        return True, None

    @classmethod
    def _validator_no_markdown(
        cls,
        content: str,
        parsed_result: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        xml_text = cls._extract_xml_text(content, parsed_result)
        if "```" in xml_text or "```" in content:
            return False, "不要输出 markdown 代码块标记（```），仅输出 mxCell XML。"
        return True, None

    @classmethod
    def _validator_no_xml_comments(
        cls,
        content: str,
        parsed_result: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        xml_text = cls._extract_xml_text(content, parsed_result)
        if "<!--" in xml_text:
            return False, "不要输出 XML 注释（<!-- -->），仅输出 mxCell XML。"
        return True, None

    @classmethod
    def _validator_xml_valid(
        cls,
        content: str,
        parsed_result: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        xml_text = cls._extract_xml_text(content, parsed_result).strip()
        if not xml_text:
            return False, "输出不能为空，请返回 mxCell XML。"
        is_valid, errors = validate_xml(xml_text)
        if not is_valid:
            hint = "请确保所有 & < > 等特殊字符已转义（例如 &amp;），并只输出 mxCell 元素。"
            return False, f"XML 解析失败: {'; '.join(errors)}。{hint}"
        return True, None

    def update_state_result(
        self,
        state: Paper2DrawioState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        try:
            if isinstance(result, dict):
                xml_content = result.get("text", "") or result.get("xml", "") or result.get("drawio_xml", "")
            elif isinstance(result, str):
                xml_content = result
            else:
                xml_content = str(result)

            if xml_content:
                xml_content = xml_content.strip()
                if xml_content.startswith("```xml"):
                    xml_content = xml_content[6:]
                elif xml_content.startswith("```"):
                    xml_content = xml_content[3:]
                if xml_content.endswith("```"):
                    xml_content = xml_content[:-3]
                xml_content = sanitize_cells_xml(xml_content)
                xml_content = xml_content.strip()

            if xml_content:
                state.drawio_xml = xml_content
                state.drawio_xml_history.append(xml_content)
                log.info(f"[DiagramEditor] XML updated, length: {len(xml_content)}")
            else:
                log.warning("[DiagramEditor] No valid XML produced")
        except Exception as e:
            log.error(f"[DiagramEditor] Failed to update state: {e}")
