import prompts
from dataflow_agent.workflow.registry import register
from dataflow_agent.state import DataInsightState
from langgraph.graph import StateGraph, END
from dataflow_agent.logger import get_logger
logger = get_logger(__name__)

@register("multi_agent_analysis")
def create_multi_agent_workflow():
    """创建多Agent分析工作流 - 使用DataInsightState"""
    workflow = StateGraph(DataInsightState)
    
    # 节点函数现在接收DataInsightState
    async def initial_data_description_node(state: DataInsightState) -> DataInsightState:
        """Step 1A: 基础数据描述节点"""
        logger.info("\n===== STEP 1A: 基础数据描述 =====")
        for agent in state.data_agents:
            try:
                agent.data_description = agent.data.describe(include='all').to_string()
            except Exception as e:
                agent.data_description = f"生成描述失败: {e}"
        return state
    
    async def independent_analysis_node(state: DataInsightState) -> DataInsightState:
        """Step 1B: 独立分析节点"""
        logger.info("\n===== STEP 1B: 独立分析 =====")
        state.initial_reports = [agent.analyze_self() for agent in state.data_agents]
        return state
    
    async def background_crossover_node(state: DataInsightState) -> DataInsightState:
        """Step 2: 背景交叉节点"""
        logger.info("\n===== STEP 2: 背景交叉与思路生成 =====")
        # 原有的背景交叉逻辑，但使用DataInsightState
        initial_reports = state.initial_reports
        agents = state.data_agents
        
        annotated_reports = initial_reports.copy()
        context_for_ideation = ""

        for report in annotated_reports:
            for annotator_agent in agents:
                if annotator_agent.name != report['agent_name']:
                    annotation = annotator_agent.annotate_other_agent_summary(report)
                    if annotation['comment']:
                        report['annotations'].append(annotation)
                        context_for_ideation += f"  - 批注 by [{annotation['author_agent']}]: {annotation['comment']}\n"
            
        # 生成数值交叉分析思路
        orchestrator = state.temp_data.get("orchestrator_agent")
        prompt = prompts.NUMERICAL_CROSSOVER_IDEATION_PROMPT.format(context=context_for_ideation)
        response = orchestrator.chat_model(prompt)
        ideas = au.extract_html_tags(response.content, ["question"]).get("question", [])
        
        state.annotated_reports = annotated_reports
        state.numerical_crossover_ideas = ideas
        return state
    
    # 添加所有节点
    workflow.add_node("initial_description", initial_data_description_node)
    workflow.add_node("independent_analysis", independent_analysis_node) 
    workflow.add_node("background_crossover", background_crossover_node)
    workflow.add_node("numerical_crossover", numerical_crossover_node)
    workflow.add_node("viewpoint_crossover", viewpoint_crossover_node)
    
    # 构建流程
    workflow.set_entry_point("initial_description")
    workflow.add_edge("initial_description", "independent_analysis")
    workflow.add_edge("independent_analysis", "background_crossover")
    workflow.add_edge("background_crossover", "numerical_crossover")
    workflow.add_edge("numerical_crossover", "viewpoint_crossover")
    workflow.add_edge("viewpoint_crossover", END)
    
    return workflow.compile()

# 其他节点函数也需要调整为接收DataInsightState
async def numerical_crossover_node(state: DataInsightState) -> DataInsightState:
    """数值交叉节点"""
    # 实现逻辑...
    return state

async def viewpoint_crossover_node(state: DataInsightState) -> DataInsightState:
    """观点交叉节点"""
    # 实现逻辑...
    return state