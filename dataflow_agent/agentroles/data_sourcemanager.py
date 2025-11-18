# dataflow/dataflowagent/agentroles/data_source_agent.py
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register

@register("data_source")
class DataSourceAgent(BaseAgent):
    system_prompt_template_name = "system_prompt_for_data_source"
    task_prompt_template_name = "task_prompt_for_data_source"
    
    def __init__(self, tool_manager=None, model_name=None, name=None, data=None, **kwargs):
        super().__init__(tool_manager, model_name, **kwargs)
        self.name = name
        self.data = data
        self.data_description = ""
    
    def get_task_prompt_params(self, state):
        """获取任务提示词参数"""
        return {
            "agent_name": self.name,
            "data_description": self.data_description,
            "external_context": getattr(state, 'external_context', '')
        }
    
    def get_default_pre_tool_results(self, state):
        """获取前置工具结果"""
        return {
            "data_analysis": self.pre_tool_results.get("data_analysis"),
            "cross_reference": self.pre_tool_results.get("cross_reference")
        }
    
    def update_state_result(self, state, result, pre_tool_results):
        """更新状态结果"""
        parsed_result = self._parse_llm_result(result)
        
        # 更新多Agent状态
        if not hasattr(state, 'agent_reports'):
            state.agent_reports = {}
        state.agent_reports[self.name] = parsed_result
        
        state.cross_analysis_ideas = parsed_result.get("cross_analysis_ideas", [])
        
        super().update_state_result(state, result, pre_tool_results)