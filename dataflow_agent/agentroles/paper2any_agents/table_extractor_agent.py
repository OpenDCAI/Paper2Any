"""
TableExtractor agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-17 22:53:23
生成位置: dataflow_agent/agentroles/common_agents/table_extractor_agent.py

本文件由 `dfa create --agent_name table_extractor` 自动生成。
1. 填写 prompt-template 名称
2. 根据需要完成 get_task_prompt_params / update_state_result
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

from pathlib import Path
import re

from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent, ValidatorFunc
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)

# ----------------------------------------------------------------------
# Agent Definition
# ----------------------------------------------------------------------
@register("table_extractor")
class TableExtractor(BaseAgent):
    """从 MinerU 输出中定位指定表格并生成 HTML（LLM），并将 HTML 渲染为 PNG 写入 state.table_img_path"""

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "table_extractor"

    @property
    def system_prompt_template_name(self) -> str:
        # TODO: 修改为真实的模板 id
        return "system_prompt_for_table_extractor"

    @property
    def task_prompt_template_name(self) -> str:
        # TODO: 修改为真实的模板 id
        return "task_prompt_for_table_extractor"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数
        提示词中的占位符：
        return {
            'text2img_prompt': pre_tool_results.get('prompt', ''),
            'image_size': pre_tool_results.get('size', '512x512'),
            'num_images': pre_tool_results.get('num_images', 1),
        }
        """
        # TODO: 按需补充
        return {
            'minueru_output': self.state.minueru_output,
            'table_num': self.state.asset_ref,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """若调用方未显式传入，返回默认前置工具结果"""
        return {"table_num": ""}

    # ---------- ReAct Validators ----------
    def get_react_validators(self) -> List[ValidatorFunc]:
        return [
            self._default_json_validator,
            self._validator_has_html_code,
            self._validator_html_not_markdown,
            self._validator_html_has_table_tag,
        ]

    @staticmethod
    def _validator_has_html_code(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(parsed_result, dict):
            return False, "返回结果必须是 JSON 对象。"
        if "html_code" not in parsed_result:
            return False, '缺少字段 "html_code"，请按格式返回：{"html_code":"..."}'
        if not isinstance(parsed_result.get("html_code"), str) or not parsed_result.get("html_code", "").strip():
            return False, '"html_code" 必须是非空字符串。'
        return True, None

    @staticmethod
    def _validator_html_not_markdown(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        html = (parsed_result.get("html_code") or "") if isinstance(parsed_result, dict) else ""
        if "```" in html or "```" in content:
            return False, "不要输出 markdown 代码块标记（```），只返回纯 JSON。"
        return True, None

    @staticmethod
    def _validator_html_has_table_tag(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        html = (parsed_result.get("html_code") or "") if isinstance(parsed_result, dict) else ""
        if "<table" not in html.lower():
            return False, 'html_code 中必须包含 <table ...> 表格结构。'
        if "</table>" not in html.lower():
            return False, 'html_code 中必须包含 </table> 闭合标签。'
        return True, None

    # ---------- Render helpers ----------
    @staticmethod
    def _normalize_table_num(raw: Any) -> str:
        """
        兼容：
        - "Table 2" / "table_2" / "2"
        输出统一 key：table_2
        """
        if raw is None:
            return ""
        s = str(raw).strip()
        if not s:
            return ""
        m = re.search(r"(\d+)", s)
        if m:
            return f"table_{m.group(1)}"
        return s.lower().replace(" ", "_")

    @staticmethod
    def _wrap_html_document(table_html: str) -> str:
        """
        兜底包装成完整 HTML 文档，保证 wkhtmltoimage 渲染稳定。
        """
        css = """
        <style>
          body { font-family: Arial, Helvetica, sans-serif; margin: 20px; }
          table { border-collapse: collapse; width: 100%; font-size: 14px; }
          th, td { border: 1px solid #333; padding: 6px 8px; vertical-align: top; }
          caption { caption-side: top; font-weight: 700; margin-bottom: 8px; }
        </style>
        """.strip()
        return f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{table_html}</body></html>"

    def _render_html_to_png(self, html_content: str, save_path: str) -> None:
        """
        使用 imgkit(wkhtmltoimage) 渲染 HTML -> PNG。
        """
        import imgkit

        options = {
            "format": "png",
            "encoding": "UTF-8",
            "quality": "100",
            "enable-local-file-access": "",
        }
        imgkit.from_string(html_content, save_path, options=options)

    # ---------- 结果写回 ----------
    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将推理结果写回 MainState，并将 html_code 渲染成图片写入 state.table_img_path"""
        super().update_state_result(state, result, pre_tool_results)

        if not isinstance(result, dict):
            return

        html_code = str(result.get("html_code") or "").strip()
        if not html_code:
            return

        # 输出目录：优先 state.result_path，其次项目 outputs/table_extractor
        base_dir = getattr(state, "result_path", "") or ""
        if base_dir:
            out_dir = Path(base_dir) / "tables"
        else:
            out_dir = get_project_root() / "outputs" / "table_extractor"
        out_dir.mkdir(parents=True, exist_ok=True)

        table_key = self._normalize_table_num(self.state.asset_ref)

        file_name = f"{table_key}.png" if table_key else "table.png"

        png_path = str((out_dir / file_name).resolve())

        html_content = self._wrap_html_document(html_code)

        try:
            self._render_html_to_png(html_content, png_path)

            state.table_img_path = png_path

            log.critical(f'[table_img_path 表格图像路径]:   {png_path}')

            # 同步到 result，方便下游直接读
            result["table_img_path"] = png_path
        except Exception as e:
            # 不抛异常，避免影响主流程；把错误写回结果
            log.error(f"渲染表格图片失败: {e}", exc_info=True)
            result["render_error"] = str(e)
        


# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
async def table_extractor(
    state: MainState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    tool_mode: str = "auto",
    react_mode: bool = False,
    react_max_retries: int = 3,
    parser_type: str = "json",
    parser_config: Optional[Dict[str, Any]] = None,
    use_vlm: bool = False,
    vlm_config: Optional[Dict[str, Any]] = None,
    use_agent: bool = False,
    **kwargs,
) -> MainState:
    """table_extractor 的异步入口
    
    Args:
        state: 主状态对象
        model_name: 模型名称，如 "gpt-4"
        tool_manager: 工具管理器实例
        temperature: 采样温度，控制随机性 (0.0-1.0)
        max_tokens: 最大生成token数
        tool_mode: 工具调用模式 ("auto", "none", "required")
        react_mode: 是否启用ReAct推理模式
        react_max_retries: ReAct模式下最大重试次数
        parser_type: 解析器类型 ("json", "xml", "text")，这个允许你在提示词中定义LLM不同的返回，xml还是json，还是直出；
        parser_config: 解析器配置字典（如XML的root_tag）
        use_vlm: 是否使用视觉语言模型，使用了视觉模型，其余的参数失效；
        vlm_config: VLM配置字典
        use_agent: 是否使用agent模式
        **kwargs: 其他传递给execute的参数
        
    Returns:
        更新后的MainState对象
    """
    agent = TableExtractor(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_mode=tool_mode,
        react_mode=react_mode,
        react_max_retries=react_max_retries,
        parser_type=parser_type,
        parser_config=parser_config,
        use_vlm=use_vlm,
        vlm_config=vlm_config,
    )
    return await agent.execute(state, use_agent=use_agent, **kwargs)


def create_table_extractor(
    tool_manager: Optional[ToolManager] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    tool_mode: str = "auto",
    react_mode: bool = False,
    react_max_retries: int = 3,
    parser_type: str = "json",
    parser_config: Optional[Dict[str, Any]] = None,
    use_vlm: bool = False,
    vlm_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> TableExtractor:
    return TableExtractor.create(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_mode=tool_mode,
        react_mode=react_mode,
        react_max_retries=react_max_retries,
        parser_type=parser_type,
        parser_config=parser_config,
        use_vlm=use_vlm,
        vlm_config=vlm_config,
        **kwargs,
    )
