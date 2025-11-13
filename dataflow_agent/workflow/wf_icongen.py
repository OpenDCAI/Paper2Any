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
import time
from dataflow_agent.state import IconGenState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder


from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import get_agent_cls, create_agent

from dataflow_agent.toolkits.tool_manager import get_tool_manager
from langchain.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger

from dataflow_agent.toolkits.imtool.req_img import generate_or_edit_and_save_image_async
from dataflow_agent.toolkits.imtool.bg_tool import local_tool_for_bg_remove

log = get_logger(__name__)

def _ts_name(stem: str, ext: str = ".png") -> str:
    return f"./{stem}_{int(time.time()*1000)%10_000_000}{ext}"

@register("icongen")
def create_icongen_graph() -> GenericGraphBuilder:  # noqa: N802
    """
    Workflow factory: dfa run --wf icongen
    """
    builder = GenericGraphBuilder(state_model=IconGenState,
                                  entry_point="should_generate_prompt")  # 修改入口为条件节点

    # ----------------------------------------------------------------------
    # TOOLS (pre_tool definitions)
    # ----------------------------------------------------------------------
    @builder.pre_tool("keywords", "icon_prompt_generator")
    def _keywords(state: IconGenState):
        return state.request.get("keywords", "")
    
    @builder.pre_tool("style", "icon_prompt_generator")
    def _style(state: IconGenState):
        return state.request.get("style", "")
    
    @builder.pre_tool("edit_prompt", "icon_prompt_generator")
    def _edit_prompt(state: IconGenState):
        return state.request.get("edit_prompt", "")

    @builder.pre_tool("prev_image", "icon_prompt_generator")
    def _prev_image(state: IconGenState):
        return state.request.get("prev_image", "")


    # ==============================================================
    # NODES
    # ==============================================================
    async def icon_prompt_generator_node(state: IconGenState) -> IconGenState:
        """
        图标提示词生成器节点
        """
        from dataflow_agent.agentroles import create_agent  # 延迟导入以避免循环依赖
        icon_prompt_generator = create_agent("icon_prompt_generator", model_name = "gpt-4o")
        state = await icon_prompt_generator.execute(state, use_agent=False)
        return state

    async def gen_img_node(state: IconGenState) -> IconGenState:
        """
        图像生成或编辑节点
        """
        prompt = (
            state.agent_results.get("icon_prompt_generator", {})
            .get("results", {})
            .get("icon_prompt")
        )
        
        edit_prompt = state.request.get("edit_prompt")
        image_path = state.request.get("prev_image")

        # 如果是二次编辑，prompt可以为空
        final_prompt = edit_prompt if image_path else prompt

        log.critical(f'final_prompt{final_prompt} - edit_prompt：{edit_prompt} - image_path：{image_path} - prompt：{prompt}')

        save_path = _ts_name("paper_icon")

        # log.critical(f'use_edit: {False if image_path == "" else True}')

        b64 = await generate_or_edit_and_save_image_async(
            prompt=final_prompt,
            save_path=save_path,
            api_url=state.request.chat_api_url,
            api_key=os.getenv("DF_API_KEY"), 
            model=state.request.model,
            image_path=image_path,
            use_edit= False if image_path == "" else True
            # edit_prompt=edit_prompt,
        )
        state.agent_results["gen_img"] = {"path": save_path, "base64": b64}
        return state

    async def bg_remove_node(state: IconGenState) -> IconGenState:
        from dataflow_agent.toolkits.imtool.bg_tool import local_tool_for_bg_remove
        
        image_to_process = (state.agent_results.get("gen_img") or {}).get("path")
        if not image_to_process:
            log.warning("[bg_remove] No image found to process, skipping.")
            return state

        try:
            output_path = local_tool_for_bg_remove({
                "image_path": image_to_process,
                "model_path": os.getenv("RM_MODEL_PATH"),
                "output_dir": "./"
            })
            if output_path:
                state.agent_results["bg_removed"] = {"path": output_path}
            else:
                log.warning("[bg_remove] bg tool returned None.")
        except Exception as e:
            log.error(f"[bg_remove] background removal failed: {e}", exc_info=True)
        
        return state

    def should_generate_prompt(state: IconGenState) -> str:
        """
        条件路由：判断是否需要生成prompt
        """
        if state.request.get("prev_image"):
            return "gen_img"  # 如果有上一张图，直接去生成/编辑
        return "icon_prompt_generator" # 否则，先生成prompt

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    nodes = {
        "should_generate_prompt": lambda state: state, 
        "icon_prompt_generator": icon_prompt_generator_node,
        "gen_img": gen_img_node,
        "bg_remove": bg_remove_node,
        '_end_': lambda state: state,
    }

    edges = [
        ("icon_prompt_generator", "gen_img"),
        ("gen_img", "bg_remove"),
        ("bg_remove", "_end_"),
    ]

    # 条件边从 should_generate_prompt 节点出发
    conditional_edges = {
        "should_generate_prompt": should_generate_prompt
    }

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edges(conditional_edges)
    return builder