"""
ChartTypeRecommender agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
根据论文核心思想和表格数据推荐合适的图表类型

用于 Paper2ExpFigure 工作流
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from dataflow_agent.agentroles.cores.strategies import VLMStrategy
from dataflow_agent.state import Paper2ExpFigureState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.agentroles.cores.base_agent import BaseAgent

log = get_logger(__name__)


@register("chart_type_recommender")
class ChartTypeRecommender(BaseAgent):
    """根据论文思想和表格数据推荐图表类型的 Agent"""

    @property
    def role_name(self) -> str:
        return "chart_type_recommender"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_chart_type_recommender"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_chart_type_recommender"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """从 pre_tool_results 中获取 prompt 参数"""
        print("ChartTypeRecommender get_task_prompt_params", pre_tool_results)
        paper_idea = pre_tool_results.get("paper_idea", "")
        table_info = pre_tool_results.get("table_info", {})
        log.info(f"[ChartTypeRecommender] 获取 prompt 参数: paper_idea={paper_idea}, table_info={table_info}")
        
        # 格式化表格信息为字符串
        if isinstance(table_info, dict):
            table_info_str = json.dumps(table_info, ensure_ascii=False, indent=2)
        else:
            table_info_str = str(table_info)

        return {
            "paper_idea": paper_idea,
            "table_info": table_info_str,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """默认的 pre_tool_results"""
        return {
            "paper_idea": "",
            "table_info": {},
        }

    def update_state_result(
        self,
        state: Paper2ExpFigureState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将推荐结果写入 state.chart_configs"""
        try:
            if isinstance(result, dict):
                # 检查表格是否适合绘图
                is_suitable = result.get("is_suitable_for_chart", True)
                suitability_reason = result.get("suitability_reason", "")
                
                chart_type = result.get("chart_type", "bar")
                chart_type_reason = result.get("chart_type_reason", "")
                chart_desc = result.get("chart_desc", "")
                
                # 获取当前正在处理的 table_id
                # 优先使用state里面传入的，因为vlm模式下，base_agent更新状态时不给提供pre_tool_results
                if state.pre_tool_results.get("table_info", {}):
                    table_info = state.pre_tool_results.get("table_info", {})
                else:
                    table_info = pre_tool_results.get("table_info", {})
                table_id = table_info.get("table_id", f"table_{len(state.chart_configs)}")
                
                # 如果表格不适合绘图，记录原因并跳过
                if not is_suitable or chart_type == "none":
                    log.info(
                        f"[ChartTypeRecommender] 表格 {table_id} 不适合绘图: {suitability_reason}"
                    )
                    # 仍然添加配置，但标记为 chart_type="none"
                    chart_config = {
                        "table_id": table_id,
                        "is_suitable_for_chart": False,
                        "suitability_reason": suitability_reason,
                        "chart_type": "none",
                        "chart_type_reason": chart_type_reason,
                    }
                else:
                    # 构建完整的图表配置
                    chart_config = {
                        "table_id": table_id,
                        "is_suitable_for_chart": True,
                        "suitability_reason": suitability_reason,
                        "chart_type": chart_type,
                        "chart_type_reason": chart_type_reason,
                        "chart_desc": chart_desc,
                    }
                    log.info(f"[ChartTypeRecommender] 推荐图表类型: {table_id} -> {chart_type}")
                
                state.chart_configs[table_id] = chart_config
        except Exception as e:
            log.warning(f"[ChartTypeRecommender] 更新 state 失败: {e}")

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
async def chart_type_recommender(
    state: Paper2ExpFigureState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    vlm_config: Optional[Dict[str, Any]] = None,
    use_agent: bool = False,
    **kwargs,
) -> Paper2ExpFigureState:
    """ChartTypeRecommender 的异步入口"""
    inst = create_chart_type_recommender(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        vlm_config=vlm_config,
    )
    return await inst.execute(state, use_agent=use_agent, **kwargs)


def create_chart_type_recommender(
    tool_manager: Optional[ToolManager] = None,
    vlm_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ChartTypeRecommender:
    """创建 ChartTypeRecommender 实例"""
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    
    return ChartTypeRecommender(
        tool_manager=tool_manager,
        vlm_config=vlm_config,
        use_vlm=True, 
        **kwargs
    )
