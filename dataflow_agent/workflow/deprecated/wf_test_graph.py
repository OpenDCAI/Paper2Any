"""
test_graph workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-01 20:16:43

1. 在 **TOOLS** 区域定义需要暴露给 Prompt 的前置工具
2. 在 **NODES**  区域实现异步节点函数 (await-able)
3. 在 **EDGES**  区域声明有向边
4. 最后返回 builder.compile() 或 GenericGraphBuilder
"""

from __future__ import annotations
import json
from dataclasses import Field
from pydantic import BaseModel
from dataflow_agent.states.test_graph_state import TestGraphState
# from dataflow_agent.state import TestGraphState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import (
    create_agent,
    create_simple_agent,
    create_react_agent,
    create_graph_agent,
    create_vlm_agent,
    SimpleConfig,
    ReactConfig,
    GraphConfig,
    VLMConfig,
    ExecutionMode,
)

from dataflow_agent.toolkits.tool_manager import get_tool_manager
from langchain.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

@register("test_graph")
def create_test_graph_graph() -> GenericGraphBuilder:  # noqa: N802
    """
    Workflow factory: dfa run --wf test_graph
    """
    builder = GenericGraphBuilder(state_model=TestGraphState,
                                  entry_point="test_graph")  # 自行修改入口

    # ----------------------------------------------------------------------
    # TOOLS (pre_tool definitions)
    # ----------------------------------------------------------------------
    # 例:
    @builder.pre_tool("purpose", "test_graph")
    def _purpose(state: TestGraphState):
        return "请问日期 11-29 的天气是什么？？？如果天气晴朗，请帮我购买这天的火车票！！"
    

    @builder.post_tool("test_graph")
    @tool
    def _get_tomorrow_weather(data_str: str):
        """
        获取明天天气

        Args:
            data_str: 日期字符串，格式为 "MM-DD"
        
        """
        return "明天天气晴朗!!!!!!!!!!!!!"
    
    @builder.post_tool("test_graph")
    @tool
    def _get_ticket(data_str: str):
        """
        购买日期的火车票

        Args:
            data_str: 日期字符串，格式为 "MM-DD"
        
        """
        return "购买1张11-29的火车票!!!!!!!!!!!!!"


    # ----------------------------------------------------------------------

    # ==============================================================
    # NODES
    # ==============================================================
    async def test_graph_node(state: TestGraphState) -> TestGraphState:
        """
        示例节点 1: 使用新的策略模式创建和执行 Agent
        
        新版 Agent 创建方式推荐使用 `create_agent` 配合配置对象 (Config)
        或使用便捷函数 `create_simple_agent`, `create_react_agent` 等。
        
        执行模式说明:
        - SimpleConfig: 简单模式，单次 LLM 调用
        - ReactConfig:  ReAct 模式，带验证和重试的循环
        - GraphConfig:  图模式，用于执行带工具的子图 (LangGraph)
        - VLMConfig:    视觉语言模型模式
        """
        
        agent = create_graph_agent(
            name="test_graph",        
            model_name="gpt-4o",
            temperature=0.1,
            max_tokens=16384,
            parser_type="json",
        )
        
        state = await agent.execute(state=state)

        log.critical(f"state.messages: {state.messages}")
        
        # 可选：处理执行结果
        agent_result = state.agent_results.get(agent.role_name, {})
        log.info(f"Agent {agent.role_name} 执行结果: {agent_result}")
        
        return state
    
    async def step2(state: TestGraphState) -> TestGraphState:
        """
        示例节点 2: 处理agent执行结果
        
        Args:
            state: 主状态对象
        """
        # TODO: 替换为真正的业务逻辑
        state.agent_results["step2"] = {"msg": "hello step2"}
        
        # 示例：从 step1 的结果中提取数据
        # if "code_reviewer" in state.agent_results:
        #     review_result = state.agent_results["code_reviewer"]
        #     # 处理审查结果...
        
        return state

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    nodes = {
        "test_graph": test_graph_node,
        "step2": step2,
        '_end_': lambda state: state,  # 终止节点
    }

    # ------------------------------------------------------------------
    # EDGES  (从节点 A 指向节点 B)
    # ------------------------------------------------------------------
    edges = [
        ("test_graph", "step2"),
        ("step2", "_end_"),  # 指向终止节点
    ]

    builder.add_nodes(nodes).add_edges(edges)
    return builder