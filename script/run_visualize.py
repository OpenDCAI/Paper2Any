from __future__ import annotations

import asyncio
import os
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow_agent.state import DFRequest, DFState, GraphState
from dataflow_agent.style import COLOR_PALETTES, ILLUSTRATION_STYLES
# from pipeline_nodes import create_pipeline_graph
from dataflow_agent.workflow.wf_pipeline_paper2graph import create_paper_visualize_graph
import json
from dataclasses import dataclass
from typing import Dict, Any

# from IPython.display import Image, display

def pretty_report(state):
    print("\n" + "="*80)
    print("INPUT TEXT:")
    print("-"*80)
    input_text = state.get("input_text", "")
    print(input_text)

    print("\n" + "="*80)
    print("COLOR PALETTE:")
    print("-"*80)
    color_palette = state.get("color_pallete") or state.get("color_palette")
    print(color_palette.strip() if color_palette else "N/A")

    print("\n" + "="*80)
    print("ILLUSTRATION STYLE:")
    print("-"*80)
    illustration_style = state.get("illustration_style") or state.get("illustraion_style")
    print(illustration_style.strip() if illustration_style else "N/A")

    print("\n" + "="*80)
    print("TEXT OUTLINE:")
    print("-"*80)
    text_outline = state.get("text_outline", {}).get("blocks", {})
    for block_id, block_lines in text_outline.items():
        print(f"{block_id}:")
        for line in block_lines:
            print(f"  - {line}")

    print("\n" + "="*80)
    print("VISUAL OUTLINE:")
    print("-"*80)
    visual_outline = state.get("visual_outline", {})
    for block_id, block_info in visual_outline.items():
        print(f"{block_id}: {block_info.get('content', '')}")
        icons = block_info.get("icons", [])
        for icon in icons:
            print(f"  Icon ({icon.get('x')},{icon.get('y')}, z={icon.get('z-level')}): {icon.get('description')}")
            print(f"    img_path: {icon.get('img_path')}")
    print("="*80 + "\n")


async def main() -> None:
    # 初始请求
    DATAFLOW_DIR = DataFlowPath.get_dataflow_dir().parent
    req = DFRequest(
        language="en",
        chat_api_url="http://123.129.219.111:3000/v1/",
        api_key=os.getenv("TAB_API_KEY", "sk-dummy"),
        model="gpt-4o",
        # json_file=f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl",
        # target="根据数据，推荐pipeline，必须要包含 ReasoningQuestionFilter算子，总数不超过4个！",
        # python_file_path = f"{DATAFLOW_DIR}/dataflow/dataflowagent/tests/my_pipeline.py",  # pipeline的输出脚本路径
        # need_debug = False, #是否需要Debug
        # max_debug_rounds = 3, #Debug的轮次数量
    )
    state = GraphState(request=req, messages=[])
    state.input_text = str(input("请输入需要配图的文字: "))
    state.graph_width = int(input("请输入配图的横向子块数量: "))
    state.graph_height = int(input("请输入配图的纵向子块数量: "))

    # Display options for color palettes and illustration styles
    def get_user_input_with_choices(prompt, options_dict):
        options = " / ".join([f"{key}" for key, value in options_dict.items()])
        user_input = input(f"{prompt}\nAvailable options:\n{options}\nYour choice (default is the first option): ")
        return user_input.strip() or list(options_dict.keys())[0]

    # 获取用户输入并根据用户输入选择配图色调
    state.color_pallete = COLOR_PALETTES[
        get_user_input_with_choices("请输入配图的色调:", COLOR_PALETTES) 
    ]

    # 获取用户输入并根据用户输入选择配图风格
    state.illustration_style = ILLUSTRATION_STYLES[
        get_user_input_with_choices("请输入配图的风格:", ILLUSTRATION_STYLES) 
    ]


    # 构建并运行图
    graph_builder = create_paper_visualize_graph()
    graph = graph_builder.build()
    # try:
    #     png_image = graph.get_graph().draw_mermaid_png()
    #     display(Image(png_image))

    #     with open("my_graph.png", "wb") as f:
    #         f.write(png_image)
    #     print("\n图已保存为 my_graph.png")

    # except Exception as e:
    #     print(f"生成PNG失败，请确保已正确安装 pygraphviz 和 Graphviz：{e}")
    
    final_state: GraphState = await graph.ainvoke(state)

    pretty_report(final_state)
    
if __name__ == "__main__":
    asyncio.run(main())