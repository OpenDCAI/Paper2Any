from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register
from dataflow_agent.state import DataInsightRequest, DataInsightState
from typing import Dict, Any
from dataflow_agent.logger import get_logger
logger = get_logger(__name__)

@register("question_recommender")
class QuestionRecommenderAgent(BaseAgent):
    """问题推荐Agent，使用DataAnalysisState"""
    
    system_prompt_template_name = "system_prompt_for_question_recommendation"
    task_prompt_template_name = "task_prompt_for_question_recommendation"
    
    def _get_data_analysis_task_params(self, state: DataInsightState) -> Dict[str, Any]:
        insights = state.insights_history[-1] if state.insights_history else None
        return {
            "goal": state.goal,
            "context": state.context,
            "schema": state.temp_data.get("schema", ""),
            "previous_question": insights["question"] if insights else None,
            "previous_answer": insights["answer"] if insights else None,
            "max_questions": state.max_questions
        }
    
    def update_state_result(self, state: DataInsightState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """更新推荐的问题"""
        if "questions" in result:
            state.root_questions = result["questions"]
            if result["questions"]:
                state.current_question = result["questions"][0]
        
        super().update_state_result(state, result, pre_tool_results)