# dataflow/dataflowagent/agentroles/poirot_agent.py
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register

@register("data_funda")
class PoirotAgent(BaseAgent):
    system_prompt_template_name = "system_prompt_for_poirot"
    task_prompt_template_name = "task_prompt_for_poirot"
    
    def __init__(self, tool_manager=None, model_name=None, savedir=None, goal=None, **kwargs):
        super().__init__(tool_manager, model_name, **kwargs)
        self.savedir = savedir
        self.goal = goal
        self.insights_history = []
    
    def get_task_prompt_params(self, state):
        """获取任务提示词参数"""
        return {
            "goal": self.goal,
            "current_question": getattr(state, 'current_question', ''),
            "insights_history": getattr(state, 'insights_history', [])
        }
    
    def get_default_pre_tool_results(self, state):
        """获取前置工具结果"""
        return {
            "data_schema": self.pre_tool_results.get("data_schema"),
            "sample_data": self.pre_tool_results.get("sample_data")
        }
    
    def update_state_result(self, state, result, pre_tool_results):
        """更新状态结果"""
        # 解析LLM返回的JSON结果
        parsed_result = self._parse_llm_result(result)
        state.insights_history = parsed_result.get("insights_history", [])
        state.current_question = parsed_result.get("current_question", "")
        state.final_summary = parsed_result.get("final_summary", "")
        
        super().update_state_result(state, result, pre_tool_results)
    
    def _parse_llm_result(self, result):
        """解析LLM返回的JSON结果"""
        # 实现JSON解析逻辑
        try:
            import json
            return json.loads(result)
        except:
            return {"error": "Failed to parse LLM result"}