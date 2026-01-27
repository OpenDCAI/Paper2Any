"""
DrawioXmlGenerator agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成 draw.io XML 格式的图表

用于 Paper2Drawio 工作流
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from dataflow_agent.state import Paper2DrawioState
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.agentroles.cores.base_agent import BaseAgent, ValidatorFunc
from dataflow_agent.toolkits.drawio_tools import validate_xml, sanitize_cells_xml

log = get_logger(__name__)


@register("drawio_xml_generator")
class DrawioXmlGenerator(BaseAgent):
    """生成 draw.io XML 的 Agent"""

    def __init__(self, **kwargs):
        kwargs['parser_type'] = 'text'
        super().__init__(**kwargs)

    @property
    def role_name(self) -> str:
        return "drawio_xml_generator"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_drawio_xml_generator"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_drawio_xml_generator"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """从 pre_tool_results 中获取 prompt 参数"""
        diagram_plan = pre_tool_results.get("diagram_plan", "")
        diagram_type = pre_tool_results.get("diagram_type", "auto")
        diagram_style = pre_tool_results.get("diagram_style", "default")
        text_content = pre_tool_results.get("text_content", "")

        return {
            "diagram_plan": diagram_plan,
            "diagram_type": diagram_type,
            "diagram_style": diagram_style,
            "text_content": text_content,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """默认的 pre_tool_results"""
        return {
            "diagram_plan": "",
            "diagram_type": "auto",
            "diagram_style": "default",
            "text_content": "",
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
        """将生成的 XML 写入 state"""
        try:
            if isinstance(result, dict):
                xml_content = result.get("text", "") or result.get("xml", "") or result.get("drawio_xml", "")
            elif isinstance(result, str):
                xml_content = result
            else:
                xml_content = str(result)

            # 清理 markdown 代码块标记
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
                log.info(f"[DrawioXmlGenerator] XML 生成成功，长度: {len(xml_content)}")
            else:
                log.warning("[DrawioXmlGenerator] 未生成有效的 XML")

        except Exception as e:
            log.error(f"[DrawioXmlGenerator] 更新状态失败: {e}")
