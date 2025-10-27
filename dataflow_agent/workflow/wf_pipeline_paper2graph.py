from __future__ import annotations

from dataflow_agent.state import DFState, GraphState
from dataflow_agent.graghbuilder.gragh_builder import GenericGraphBuilder
from dataflow_agent.toolkits.optool.op_tools import (
    local_tool_for_get_purpose,
    get_operator_content_str,
)
from dataflow_agent.agentroles.text_outline_agent import create_text_outline_generator
from dataflow_agent.agentroles.visual_outline_agent import create_visual_outline_agent
from dataflow_agent.utils import draw_pic_entry
from dataflow_agent.toolkits.tool_manager import get_tool_manager
import os, time

def create_paper_visualize_graph() -> GenericGraphBuilder:
    """Build the operator write workflow graph.

    Flow: match_operator -> write_the_operator -> operator_executor
          -> (code_debugger -> rewriter -> after_rewrite -> operator_executor)*
    """
    builder = GenericGraphBuilder(state_model=GraphState, entry_point="text_outline_generator")

    # ---------------- 前置工具：match_operator ----------------
    # @builder.pre_tool("get_operator_content", "match_operator")
    # def pre_get_operator_content(state: DFState):
    #     cat = state.category.get("category") or state.request and getattr(state.request, "category", None)
    #     data_type = cat or state.temp_data.get("category") or "Default"
    #     return get_operator_content_str(data_type=data_type)

    @builder.pre_tool('input_text', 'text_outline_generator')
    def get_input_text(state: GraphState):
        return state.input_text
    
    @builder.pre_tool('num_of_blocks', 'text_outline_generator')
    def get_block_num(state: GraphState):
        return state.graph_width * state.graph_height
    
    @builder.pre_tool('text_outline', 'visual_outline_generator')
    def get_input_text(state: GraphState):
        return state.text_outline
    
    @builder.pre_tool('color_palette', 'visual_outline_generator')
    def get_input_text(state: GraphState):
        return state.color_pallete
    
    @builder.pre_tool('illustration_style', 'visual_outline_generator')
    def get_input_text(state: GraphState):
        return state.illustration_style
    
    
    # ---------------- 节点实现 ----------------
    async def text_outline_node(s: GraphState) -> GraphState:
        agent = create_text_outline_generator(tool_manager=get_tool_manager())
        return await agent.execute(s, use_agent=False)
    
    async def visual_outline_node(s: GraphState) -> GraphState:
        agent = create_visual_outline_agent(tool_manager=get_tool_manager())
        return await agent.execute(s, use_agent=False)
    
    async def draw_pics_node(s: GraphState) -> GraphState:
        visual_outline = s.visual_outline
        IMAGE_ASSETS_DIR = os.path.join(os.getcwd(), "assets")
        timestamp_dir = os.path.join(IMAGE_ASSETS_DIR, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(timestamp_dir, exist_ok=True)
        for block_id, block_data in visual_outline.items():
            for icon in block_data.get("icons", []):
                # 拼接描述 + 色调 + 风格信息，构成完整 prompt
                full_description = (
                    f"{icon['description'].strip()} "
                    f"(Color palette: {s.color_pallete.strip()} | "
                    f"Illustration style: {s.illustration_style.strip()})"
                )

                # 生成图片路径并保存
                img_path = draw_pic_entry(timestamp_dir, full_description)

                # 为每个 icon 添加 img_path 字段
                icon['img_path'] = img_path

        # return await agent.execute(s, use_agent=False)

    # ---------------- 条件边（复用 pipeline 的循环思路） ----------------
    def exec_condition(s: GraphState):
        if s.request.need_debug:
            if s.execution_result.get("success"):
                return "__end__"
            if s.temp_data.get("round", 0) >= s.request.max_debug_rounds:
                return "__end__"
            return "code_debugger"
        else:
            return "__end__"

    # 修改后的节点，只保留 text_outline_node, visual_outline_node 和 draw_pics_node
    nodes = {
        "text_outline_generator": text_outline_node,   # 先执行 text_outline_node
        "visual_outline_generator": visual_outline_node, # 然后执行 visual_outline_node
        "draw_pics": draw_pics_node,                     # 最后执行 draw_pics_node
    }

    # 修改后的边的顺序，只保留这些节点的执行顺序
    edges = [
        ("text_outline_generator", "visual_outline_generator"),  # text_outline_generator -> visual_outline_generator
        ("visual_outline_generator", "draw_pics"),               # visual_outline_generator -> draw_pics
    ]   


    builder.add_nodes(nodes).add_edges(edges).add_conditional_edges({"draw_pics": exec_condition})
    return builder

