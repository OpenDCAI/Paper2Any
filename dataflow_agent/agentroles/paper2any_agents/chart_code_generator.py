"""
ChartCodeGenerator agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
根据图表配置和表格数据生成 matplotlib 代码

用于 Paper2ExpFigure 工作流
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from dataflow_agent.state import Paper2ExpFigureState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.agentroles.cores.base_agent import BaseAgent

log = get_logger(__name__)


@register("chart_code_generator")
class ChartCodeGenerator(BaseAgent):
    """根据图表配置生成 matplotlib 代码的 Agent"""

    @property
    def role_name(self) -> str:
        return "chart_code_generator"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_chart_code_generator"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_chart_code_generator"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """从 pre_tool_results 中获取 prompt 参数"""
        paper_idea = pre_tool_results.get("paper_idea", "")
        chart_config = pre_tool_results.get("chart_config", {})
        table_caption = pre_tool_results.get("table_caption", "")
        # 将配置格式化为 JSON 字符串
        if isinstance(chart_config, dict):
            chart_config_str = (
                f"你要生成的图表类型：{chart_config.get('chart_type', '')}\n"
                f"为什么要选择这种类型：{chart_config.get('chart_type_reason', '')}\n"
                f"对图表预期的描述：{chart_config.get('chart_desc', '')}\n"
            )
        else:
            chart_config_str = str(chart_config)

        return {
            "paper_idea": paper_idea,
            "chart_config": chart_config_str,
            "table_caption": table_caption,
            
            # "table_headers": headers_str,
            # "table_rows": rows_str,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """默认的 pre_tool_results"""
        return {
            "paper_idea": "",
            "chart_config": {},
            "table_caption": "",
            # "table_headers": [],
            # "table_rows": [],
        }

    def update_state_result(
        self,
        state: Paper2ExpFigureState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将生成的代码写入 state.generated_codes"""
        try:
            if isinstance(result, dict):
                code = result.get("code", "")
                description = result.get("description", "")

                if code:
                    # 获取当前正在处理的 table_id
                    # 优先使用state里面传入的，因为vlm模式下，base_agent更新状态时不给提供pre_tool_results
                    if state.pre_tool_results.get("chart_config", {}):
                        chart_config = state.pre_tool_results.get("chart_config", {})
                    else:
                        chart_config = pre_tool_results.get("chart_config", {})
                    table_id = chart_config.get("table_id", f"table_{len(state.generated_codes)}")

                    code_entry = {
                        "table_id": table_id,
                        "code": code,
                        "description": description,
                    }
                    state.generated_codes[table_id] = code_entry
                    log.info(f"[ChartCodeGenerator] 生成代码: {table_id}, 长度: {len(code)}")
        except Exception as e:
            log.warning(f"[ChartCodeGenerator] 更新 state 失败: {e}")

        return super().update_state_result(state, result, pre_tool_results)


    async def execute_pre_tools(self, state: MainState) -> Dict[str, Any]:
        """重写 execute_pre_tools，方便并行调用时注入前置工具结果"""
        results = await super().execute_pre_tools(state)
        
        inject_results = state.pre_tool_results
        for key, value in inject_results.items():
            if value:
                results.update(
                    {key: value}
                )
        
        return results

# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
async def chart_code_generator(
    state: Paper2ExpFigureState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    vlm_config: Optional[Dict[str, Any]] = None,
    use_agent: bool = False,
    **kwargs,
) -> Paper2ExpFigureState:
    """ChartCodeGenerator 的异步入口"""
    inst = create_chart_code_generator(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        vlm_config=vlm_config,
    )
    return await inst.execute(state, use_agent=use_agent, **kwargs)


def create_chart_code_generator(
    tool_manager: Optional[ToolManager] = None,
    vlm_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ChartCodeGenerator:
    """创建 ChartCodeGenerator 实例"""
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return ChartCodeGenerator(
        tool_manager=tool_manager, 
        vlm_config=vlm_config,
        use_vlm=True,
        **kwargs
    )
