from dataflow_agent.workflow.registry import register
from dataflow_agent.state import DataInsightState
from langgraph.graph import StateGraph, END

@register("single_insight_workflow")
def create_single_insight_workflow():
    """创建单洞察分析工作流"""
    from dataflow_agent.agentroles.registry import get_agent_cls
    
    workflow = StateGraph(DFState)
    
    # 定义节点
    async def question_recommendation_node(state: DFState) -> DFState:
        recommender = get_agent_cls("question_recommender")()
        return await recommender.execute(state, use_agent=False)
    
    async def data_analysis_node(state: DFState) -> DFState:
        analyzer = get_agent_cls("data_analyzer")()
        return await analyzer.execute(state, use_agent=False)
    
    async def insight_summary_node(state: DFState) -> DFState:
        summarizer = get_agent_cls("insight_summarizer")()
        return await summarizer.execute(state, use_agent=False)
    
    # 添加节点
    workflow.add_node("recommend_questions", question_recommendation_node)
    workflow.add_node("analyze_data", data_analysis_node) 
    workflow.add_node("summarize_insights", insight_summary_node)
    
    # 构建流程
    workflow.set_entry_point("recommend_questions")
    workflow.add_edge("recommend_questions", "analyze_data")
    workflow.add_conditional_edges(
        "analyze_data",
        should_continue_analysis,
        {
            "continue": "recommend_questions",
            "summarize": "summarize_insights"
        }
    )
    workflow.add_edge("summarize_insights", END)
    
    return workflow.compile()

def should_continue_analysis(state: DFState) -> str:
    """判断是否继续分析"""
    if len(state.insights_history) >= state.branch_depth:
        return "summarize"
    return "continue"