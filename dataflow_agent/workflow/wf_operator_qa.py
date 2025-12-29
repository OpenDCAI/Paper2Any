"""
OperatorQA Workflow - 算子问答工作流
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-01

本工作流实现了基于 Agentic RAG 的算子问答功能：
1. 前置工具：获取用户查询、对话历史
2. 后置工具：RAG 检索、获取算子信息/源码/参数（LLM 自主调用）
3. 支持多轮对话
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.tools import tool

from dataflow_agent.state import MainState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.toolkits.tool_manager import get_tool_manager
from dataflow_agent.agentroles.data_agents.operator_qa_agent import (
    OperatorQAAgent,
    OperatorRAGService,
    create_operator_qa_agent,
)
from dataflow_agent.toolkits.filetool.filetools import (
    read_file_content,
    list_directory_content,
)

from dataflow_agent.logger import get_logger

log = get_logger(__name__)


# ==============================================================================
# Workflow Factory
# ==============================================================================
@register("operator_qa")
def create_operator_qa_graph() -> GenericGraphBuilder:
    """
    Workflow factory: dfa run --wf operator_qa
    
    构建算子问答工作流图，支持：
    1. RAG 检索相关算子
    2. 获取算子详细信息
    3. 获取算子源码（后置工具）
    4. 多轮对话
    """
    builder = GenericGraphBuilder(
        state_model=MainState,
        entry_point="operator_qa_node"
    )
    
    # 创建共享的 RAG 服务实例
    rag_service = OperatorRAGService()
    
    # 创建共享的消息历史管理器（用于多轮对话）
    from dataflow_agent.graphbuilder.message_history import AdvancedMessageHistory
    shared_message_history = AdvancedMessageHistory()

    # ==========================================================================
    # 前置工具 (Pre-Tools) - 每次自动执行
    # ==========================================================================
    
    @builder.pre_tool("user_query", "operator_qa")
    def get_user_query(state: MainState) -> str:
        """获取用户查询"""
        return state.request.target or ""
    
    # 注意：对话历史由 BaseAgent 的 AdvancedMessageHistory 自动管理，
    # 不再通过 pre-tool 嵌入 prompt

    # ==========================================================================
    # 后置工具 (Post-Tools) - LLM 自主决定是否调用
    # ==========================================================================
    
    class SearchOperatorsInput(BaseModel):
        """搜索算子的输入参数"""
        query: str = Field(description="搜索查询，描述需要的算子功能，如 '过滤文本' '数据清洗'")
        top_k: int = Field(default=5, description="返回结果数量，默认5个")
    
    @builder.post_tool("operator_qa")
    @tool(args_schema=SearchOperatorsInput)
    def search_operators(query: str, top_k: int = 5) -> str:
        """
        根据功能描述搜索相关算子。这是主要的算子检索工具。
        当用户询问某类功能的算子、或需要查找算子时，首先使用此工具。
        如果对话历史中已有相关算子信息，可以不调用此工具直接回答。
        """
        result = rag_service.search_and_get_info(query, top_k=top_k)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    class GetOperatorInfoInput(BaseModel):
        """获取算子详细信息的输入参数"""
        operator_name: str = Field(description="要获取信息的算子名称，如 'PromptedFilter'")
    
    @builder.post_tool("operator_qa")
    @tool(args_schema=GetOperatorInfoInput)
    def get_operator_info(operator_name: str) -> str:
        """
        获取指定算子的详细描述信息。
        当用户询问某个特定算子的功能、用途时使用此工具。
        """
        info = rag_service.get_operator_info([operator_name])
        return info
    
    class GetOperatorSourceInput(BaseModel):
        """获取算子源码的输入参数"""
        operator_name: str = Field(description="要获取源码的算子名称，如 'PromptedFilter'")
    
    @builder.post_tool("operator_qa")
    @tool(args_schema=GetOperatorSourceInput)
    def get_operator_source_code(operator_name: str) -> str:
        """
        获取指定算子的完整源代码。
        当用户询问算子的具体实现细节、或需要了解算子内部逻辑时使用此工具。
        """
        return rag_service.get_operator_source(operator_name)
    
    class GetOperatorParamsInput(BaseModel):
        """获取算子参数的输入参数"""
        operator_name: str = Field(description="要获取参数信息的算子名称")
    
    @builder.post_tool("operator_qa")
    @tool(args_schema=GetOperatorParamsInput)
    def get_operator_parameters(operator_name: str) -> str:
        """
        获取指定算子的参数详情，包括 __init__ 和 run 方法的参数。
        当用户询问算子如何配置、参数含义时使用此工具。
        """
        params = rag_service.get_operator_params(operator_name)
        return json.dumps(params, ensure_ascii=False, indent=2)
    
    # ==========================================================================
    # 文件操作工具 (File Tools) - LLM 自主决定是否调用
    # ==========================================================================
    
    class ReadFileInput(BaseModel):
        """读取文件内容的输入参数"""
        file_path: str = Field(description="文件路径，可以是相对路径（相对于项目根目录）或绝对路径")
        start_line: int = Field(default=None, description="起始行号（从1开始，可选）。不指定则从第1行开始")
        end_line: int = Field(default=None, description="结束行号（包含，可选）。不指定则读取到文件末尾")
    
    @builder.post_tool("operator_qa")
    @tool(args_schema=ReadFileInput)
    def read_text_file(file_path: str, start_line: int = None, end_line: int = None) -> str:
        """
        读取文本文件内容。
        
        支持读取项目内的任意文本文件，可指定读取的行范围。
        出于安全考虑，只能读取项目根目录内的文件。
        
        当用户需要查看某个文件的内容、或需要了解项目中某个文件的具体实现时使用此工具。
        
        Examples:
            >>> read_text_file("README.md")  # 读取整个文件
            >>> read_text_file("src/main.py", start_line=10, end_line=20)  # 读取第10-20行
        """
        result = read_file_content(file_path, start_line, end_line)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    class ListDirectoryInput(BaseModel):
        """查看目录内容的输入参数"""
        dir_path: str = Field(description="目录路径，可以是相对路径（相对于项目根目录）或绝对路径")
        show_hidden: bool = Field(default=False, description="是否显示隐藏文件（以.开头的文件），默认 False")
        recursive: bool = Field(default=False, description="是否递归显示子目录内容，默认 False")
    
    @builder.post_tool("operator_qa")
    @tool(args_schema=ListDirectoryInput)
    def list_directory(dir_path: str, show_hidden: bool = False, recursive: bool = False) -> str:
        """
        查看目录内容。
        
        列出指定目录下的文件和子目录，支持 Windows 和 Linux 系统。
        出于安全考虑，只能查看项目根目录内的目录。
        
        当用户需要了解项目结构、查看某个目录下有哪些文件时使用此工具。
        
        Examples:
            >>> list_directory(".")  # 列出项目根目录
            >>> list_directory("src", show_hidden=True)  # 列出 src 目录，包含隐藏文件
            >>> list_directory("dataflow_agent", recursive=True)  # 递归列出目录
        """
        result = list_directory_content(dir_path, show_hidden, recursive)
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    # ==========================================================================
    # 节点定义 (Nodes)
    # ==========================================================================
# DATAFLOW_LOG_LEVEL=DEBUG python /mnt/DataFlow/lz/proj/agentgroup/ziyi/DataFlow-Agent/script/run_dfa_operator_qa.py --query "GeneralFilter算子有什么作用？"

    async def operator_qa_node(state: MainState) -> MainState:
        """
        算子问答节点
        
        使用 OperatorQAAgent 处理用户查询，支持：
        - LLM 自主决定是否调用 RAG 检索
        - 多轮对话（由 BaseAgent 通过 messages 数组管理）
        - 工具调用（检索算子、获取源码、参数等）
        """
        # 多轮对话：如果有历史消息，追加新的用户问题
        # 第一轮时 state.messages 为空，由 build_messages 构建 system + user
        # 后续轮次手动追加用户问题，避免 build_messages 不被调用导致新问题丢失
        if state.messages:
            user_query = state.request.target or ""
            if user_query:
                from langchain_core.messages import HumanMessage
                state.messages = state.messages + [HumanMessage(content=user_query)]
                log.debug(f"追加用户问题到历史，当前共 {len(state.messages)} 条消息")
        
        tm = get_tool_manager()
        
        # 创建 Agent（使用共享的消息历史管理器）
        agent = create_operator_qa_agent(
            tool_manager=tm,
            rag_service=rag_service,
            model_name=state.request.model or "gpt-4o",
            temperature=0.1,
            max_tokens=4096,
            parser_type="json",
            tool_mode="auto",
            message_history=shared_message_history,  # 共享消息历史
        )
        
        # 执行 Agent（use_agent=True 启用工具调用）
        state = await agent.execute(state, use_agent=True)
        
        # 记录结果
        result = state.agent_results.get("operator_qa", {})
        log.info(f"OperatorQA 执行结果: {result}")
        
        return state
    
    # ==========================================================================
    # 图结构 (Graph Structure)
    # ==========================================================================
    
    nodes = {
        "operator_qa_node": operator_qa_node,
        "_end_": lambda state: state,
    }

    edges = [
        ("operator_qa_node", "_end_"),
    ]

    # 关键：将节点 role 映射为 "operator_qa"，与 Agent 的 role_name 和前置工具的 role 保持一致
    builder.add_nodes(nodes, role_mapping={"operator_qa_node": "operator_qa"}).add_edges(edges)
    return builder


# ==============================================================================
# 便捷执行函数
# ==============================================================================
async def run_operator_qa(
    query: str,
    chat_api_url: str = "http://123.129.219.111:3000/v1/",
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    执行算子问答
    
    Args:
        query: 用户查询
        chat_api_url: Chat API 地址
        api_key: API 密钥
        model: 模型名称
        
    Returns:
        问答结果字典
    """
    import os
    from dataflow_agent.state import DFRequest
    
    # 构建请求
    req = DFRequest(
        language="zh",
        chat_api_url=chat_api_url,
        api_key=api_key or os.getenv("DF_API_KEY", ""),
        model=model,
        target=query,
    )
    
    # 构建状态
    state = MainState(request=req, messages=[])
    
    # 执行工作流
    graph_builder = create_operator_qa_graph()
    graph = graph_builder.build()
    final_state = await graph.ainvoke(state)
    
    # 提取结果
    result = final_state.get("agent_results", {}).get("operator_qa", {})
    return {
        "answer": result.get("results", {}).get("answer", ""),
        "related_operators": result.get("results", {}).get("related_operators", []),
        "code_snippet": result.get("results", {}).get("code_snippet", ""),
    }
