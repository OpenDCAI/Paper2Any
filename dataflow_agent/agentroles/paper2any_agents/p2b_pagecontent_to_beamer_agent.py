"""
P2bPagecontentToBeamer agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
从 paper2page_content 产出的 pagecontent（结构化大纲）生成 LaTeX Beamer 代码。
输入：pagecontent (list[dict]: title, layout_description, key_points, asset_ref)
输出：latex_code，写入 state.beamer_code_path。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.toolkits.p2vtool.p2v_tool import extract_beamer_code

log = get_logger(__name__)


# ----------------------------------------------------------------------
# Agent Definition
# ----------------------------------------------------------------------
@register("p2b_pagecontent_to_beamer")
class P2bPagecontentToBeamer(BaseAgent):
    """从 pagecontent（结构化大纲）生成 Beamer LaTeX 代码"""

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "p2b_pagecontent_to_beamer"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_p2b_pagecontent_to_beamer"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_p2b_pagecontent_to_beamer"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pagecontent": pre_tool_results.get("pagecontent", "[]"),
            "output_language": pre_tool_results.get("output_language", "English"),
            "pdf_images_working_dir": pre_tool_results.get("pdf_images_working_dir", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {}

    def _get_beamer_code_from_result(self, result: Dict[str, Any]) -> str:
        """从 result 中取出 Beamer 代码，兼容规范 dict 或解析失败时的 {"raw": content}。"""
        raw = result.get("latex_code", "") if isinstance(result, dict) else ""
        if isinstance(raw, str) and raw:
            code = extract_beamer_code(raw)
            if code:
                return code
        # 解析失败时 result 可能为 {"raw": content}，尝试从原始文本提取
        raw_content = result.get("raw", "") if isinstance(result, dict) else ""
        if isinstance(raw_content, str) and raw_content:
            code = extract_beamer_code(raw_content)
            if code:
                return code
            try:
                from dataflow_agent.utils import robust_parse_json
                parsed = robust_parse_json(raw_content)
                if isinstance(parsed, dict):
                    raw = parsed.get("latex_code", "")
                    if isinstance(raw, str) and raw:
                        code = extract_beamer_code(raw)
                        if code:
                            return code
            except Exception:
                pass
        return ""

    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        beamer_code = self._get_beamer_code_from_result(result)
        if not beamer_code:
            log.error("p2b_pagecontent_to_beamer: 未得到有效 Beamer 代码")
            super().update_state_result(state, result, pre_tool_results)
            return

        result_path = getattr(state, "result_path", "") or ""
        if result_path:
            base = Path(result_path).expanduser().resolve()
        else:
            req = getattr(state, "request", None)
            paper_pdf_path = getattr(req, "paper_pdf_path", "") if req else ""
            base = Path(paper_pdf_path).expanduser().resolve().parent if paper_pdf_path else Path(".").resolve()
        auto_dir = base / "auto"
        auto_dir.mkdir(parents=True, exist_ok=True)
        beamer_code_path = auto_dir / "beamer_code.tex"
        beamer_code_path.write_text(beamer_code, encoding="utf-8")
        state.beamer_code_path = str(beamer_code_path)
        log.info("p2b_pagecontent_to_beamer: Beamer 代码已写入 %s", beamer_code_path)
        super().update_state_result(state, result, pre_tool_results)
