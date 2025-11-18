# dataflow/dataflowagent/toolkits/data_tools.py
from langchain.tools import tool
from pydantic import BaseModel, Field
from dataflow_agent.toolkits.builder import pre_tool, post_tool

class DataAnalysisInput(BaseModel):
    file_path: str = Field(description="数据文件路径")
    analysis_type: str = Field(description="分析类型")

@pre_tool("data_analysis", "poirot")
def analyze_data_tool(state):
    """数据分析前置工具"""
    file_path = getattr(state, 'dataset_path', '')
    return {
        "schema": get_data_schema(file_path),
        "sample": get_data_sample(file_path, sample_size=5)
    }

@pre_tool("cross_reference", "data_source")  
def cross_reference_tool(state):
    """跨数据源引用前置工具"""
    other_agents_data = {}
    for agent_name, agent_data in getattr(state, 'agent_data', {}).items():
        if agent_name != state.current_agent:
            other_agents_data[agent_name] = agent_data
    return other_agents_data

@post_tool("recommender")
@tool(args_schema=DataAnalysisInput)
def recommend_analysis(file_path: str, analysis_type: str) -> str:
    """推荐分析路径的后置工具"""
    return f"针对 {file_path} 的 {analysis_type} 分析建议"