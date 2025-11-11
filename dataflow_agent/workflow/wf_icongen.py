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
import asyncio
import json
import os
from dataflow_agent.state import MainState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder


from dataflow_agent.workflow.registry import register


from dataflow_agent.toolkits.tool_manager import get_tool_manager
from langchain.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger

from dataflow_agent.toolkits.imtool.req_img import generate_or_edit_and_save_image_async
from dataflow_agent.toolkits.imtool.bg_tool import local_tool_for_bg_remove

log = get_logger(__name__)

@register("icongen")
def create_icongen_graph() -> GenericGraphBuilder:  # noqa: N802
    """
    Workflow factory: dfa run --wf icongen
    """
    builder = GenericGraphBuilder(state_model=MainState,
                                  entry_point="icon_prompt_generator")  # 自行修改入口

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
    @builder.pre_tool("keywords", "icon_prompt_generator")
    def _keywords(state: MainState):
        return state.request.get("keywords", "")
    
    @builder.pre_tool("style", "icon_prompt_generator")
    def _keywords(state: MainState):
        return state.request.get("style", "")


    # ==============================================================
    # NODES
    # ==============================================================
    async def icon_prompt_generator_node(state: MainState) -> MainState:
        """
        图标提示词生成器节点
        """
        from dataflow_agent.agentroles import create_agent  # 延迟导入以避免循环依赖
        icon_prompt_generator = create_agent("icon_prompt_generator", tool_manager=get_tool_manager())
        state = await icon_prompt_generator.execute(state, use_agent=False)
        return state

    # async def icon_prompt_generator_node(state: MainState) -> MainState:
    #     return state
    async def gen_img_node(state: MainState) -> MainState:
        b64 = await generate_or_edit_and_save_image_async(
            prompt=state.agent_results["icon_prompt_generator"]["results"]["icon_prompt"],
            save_path="./icon.png",
            api_url=state.request.chat_api_url,
            api_key=os.getenv("DF_API_KEY"), 
            model="gemini-2.5-flash-image-preview"
        )
        state.agent_results["gen_img"] = {"base64": b64}
        return state
    async def bg_remove_node(state: MainState) -> MainState:
        from dataflow_agent.toolkits.imtool.bg_tool import local_tool_for_bg_remove
        output_path = local_tool_for_bg_remove({
            "image_path": "./icon.png",
            "model_path": None,
            "output_dir": "./"
        })
        state.agent_results["bg_removed"] = {"path": output_path}
        return state


    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    nodes = {
        "icon_prompt_generator": icon_prompt_generator_node,
        "gen_img": gen_img_node,
        "bg_remove": bg_remove_node,
        '_end_': lambda state: state,  # 终止节点
    }

    # ------------------------------------------------------------------
    # EDGES  (从节点 A 指向节点 B)
    # ------------------------------------------------------------------
    edges = [
        ("icon_prompt_generator", "gen_img"),
        ("gen_img", "bg_remove"),
        ("bg_remove", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges)
    return builder