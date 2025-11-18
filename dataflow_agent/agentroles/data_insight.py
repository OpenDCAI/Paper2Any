from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register
from dataflow_agent.state import DataInsightRequest, DataInsightState
from typing import Dict, Any
from dataflow_agent.logger import get_logger
logger = get_logger(__name__)

@register("data_insight")
class DataInsightAgent(BaseAgent):
    """DataInsight核心功能"""
    
    system_prompt_template_name = "system_prompt_for_data_analysis"
    task_prompt_template_name = "task_prompt_for_data_analysis"
    
    def __init__(self, tool_manager=None, model_name="gpt-4", **kwargs):
        super().__init__(tool_manager, model_name, **kwargs)
        # 保留原有配置
        self.temperature = kwargs.get('temperature', 0)
        self.n_retries = kwargs.get('n_retries', 2)
    
    def get_task_prompt_params(self, state: DataInsightState) -> Dict[str, Any]:
        """获取任务提示词参数"""
        return {
            "goal": state.goal,
            "context": state.context,
            "schema": state.temp_data.get("schema", ""),
            "question": state.current_question,
            "max_questions": state.max_questions
        }
    
    def get_default_pre_tool_results(self, state: DataInsightState) -> Dict[str, Any]:
        """执行前置工具获取数据"""
        # 调用数据加载工具
        schema_result = self.tool_manager.call_tool("get_data_schema", state)
        return {
            "schema": schema_result,
            "data_loaded": True
        }
    
    def update_state_result(self, state: DataInsightState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """更新状态结果"""
        if "insight" in result:
            state.insights_history.append({
                "question": state.current_question,
                "answer": result.get("answer", ""),
                "insight": result.get("insight", ""),
                "justification": result.get("justification", "")
            })
        
        # 调用父类方法保持兼容
        super().update_state_result(state, result, pre_tool_results)