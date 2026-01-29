"""
DiagramVlmValidator agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VLM-based diagram validation for draw.io outputs.
"""

from __future__ import annotations

from typing import Any, Dict

from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register


@register("diagram_vlm_validator")
class DiagramVlmValidator(BaseAgent):
    """Validate rendered diagram image with VLM."""

    @property
    def role_name(self) -> str:
        return "diagram_vlm_validator"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_drawio_vlm_validator"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_drawio_vlm_validator"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "diagram_type": pre_tool_results.get("diagram_type", "auto"),
            "diagram_xml": pre_tool_results.get("diagram_xml", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "diagram_type": "auto",
            "diagram_xml": "",
        }
