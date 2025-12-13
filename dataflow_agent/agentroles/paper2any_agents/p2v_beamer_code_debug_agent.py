"""
P2vBeamerCodeDebug agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-11-29 15:41:18
生成位置: dataflow_agent/agentroles/common_agents/p2v_beamer_code_debug_agent.py

本文件由 `dfa create --agent_name p2v_beamer_code_debug` 自动生成。
1. 填写 prompt-template 名称
2. 根据需要完成 get_task_prompt_params / update_state_result
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

log = get_logger(__name__)

# ----------------------------------------------------------------------
# Agent Definition
# ----------------------------------------------------------------------
@register("p2v_beamer_code_debug")
class P2vBeamerCodeDebug(BaseAgent):
    """用来debug得到的beamer代码"""

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "p2v_beamer_code_debug"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_p2v_beamer_code_debug"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_p2v_beamer_code_debug"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数"""
        return {
            "beamer_code": pre_tool_results.get("beamer_code", ""),
            "code_debug_result": pre_tool_results.get("code_debug_result", ""), 
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """若调用方未显式传入，返回默认前置工具结果"""
        return {}

    # ---------- 结果写回 ----------
    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将推理结果 {latex_code: xxxx} 写回 MainState"""
        beamer_code = result.get("latex_code", '')
        beamer_code_path = state.beamer_code_path
        if beamer_code and beamer_code_path:
            from pathlib import Path

            tex_path = Path(beamer_code_path)
            tex_path.write_text(beamer_code, encoding='utf-8')
            # 编译最新的tex代码
            from dataflow_agent.toolkits.p2vtool.p2v_tool import compile_tex
            is_beamer_wrong, is_beamer_warning, code_debug_result = compile_tex(beamer_code_path)
            assert(is_beamer_wrong, False)
            state.ppt_path = beamer_code_path.replace(".tex", ".pdf")
            log.info(f"将更新好的beamer code写回 {beamer_code_path}")
        else:
            log.error(f"Failed to update beamer code: missing latex_code or beamer_code_path")

        super().update_state_result(state, result, pre_tool_results)


# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
async def p2v_beamer_code_debug(
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
    """p2v_beamer_code_debug 的异步入口
    
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
    agent = P2vBeamerCodeDebug(
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


def create_p2v_beamer_code_debug(
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
) -> P2vBeamerCodeDebug:
    return P2vBeamerCodeDebug.create(
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