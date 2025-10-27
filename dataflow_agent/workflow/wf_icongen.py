"""
icongen workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-10-27 11:11:56

1. 在 **TOOLS** 区域定义需要暴露给 Prompt 的前置工具
2. 在 **NODES**  区域实现异步节点函数 (await-able)
3. 在 **EDGES**  区域声明有向边
4. 最后返回 builder.compile() 或 GenericGraphBuilder
"""

from __future__ import annotations
import json
from dataflow_agent.state import MainState
from dataflow_agent.graghbuilder.gragh_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register   # <- 核心装饰器


@register("icongen")
def create_icongen_graph() -> GenericGraphBuilder:  # noqa: N802
    """
    Workflow factory: dfa run --wf icongen
    """
    builder = GenericGraphBuilder(state_model=MainState,
                                  entry_point="step1")  # 自行修改入口

    # ----------------------------------------------------------------------
    # TOOLS (pre_tool definitions)
    # ----------------------------------------------------------------------
    # 例:
    # @builder.pre_tool("purpose", "step1")
    # def _purpose(state: MainState):
    #     return "这里放入字符串 / 数值 / 列表 / 字典等供 prompt 使用"

    # @builder.post_tool('','')
    # def _post_tool1():
    # ----------------------------------------------------------------------

    # ==============================================================
    # NODES
    # ==============================================================
    async def step1(state: MainState) -> MainState:
        """
        示例节点 1
        """
        # TODO: 替换为真正的业务逻辑
        state.agent_results["step1"] = {"msg": "hello step1"}
        return state

    async def step2(state: MainState) -> MainState:
        """
        示例节点 2
        """
        state.agent_results["step2"] = {"msg": "hello step2"}
        return state

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    nodes = {
        "step1": step1,
        "step2": step2,
    }

    # ------------------------------------------------------------------
    # EDGES  (从节点 A 指向节点 B)
    # ------------------------------------------------------------------
    edges = [
        ("step1", "step2"),
    ]

    builder.add_nodes(nodes).add_edges(edges)
    return builder