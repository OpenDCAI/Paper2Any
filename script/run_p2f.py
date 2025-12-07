from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any
import time

# from IPython.display import Image, display

from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow import run_workflow


# ====================== 通用工具函数 ====================== #
def to_serializable(obj: Any):
    """递归将对象转成可 JSON 序列化结构"""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if hasattr(obj, "__dict__"):
        return to_serializable(obj.__dict__)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# def save_final_state_json(final_state: dict, out_dir: Path, filename: str = "final_state.json") -> None:
#     """
#     把 DFState 序列化写入 <项目根>/dataflow_agent/tmps/(session_id?)/final_state.json
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / filename
#     with out_path.open("w", encoding="utf-8") as f:
#         json.dump(to_serializable(final_state), f, ensure_ascii=False, indent=2)
#     print(f"final_state 已保存到 {out_path}")

def save_final_state_json(final_state: dict, out_dir: Path, filename: str = "final_state.json") -> None:
    """
    直接把 final_state 用 json.dump 存到 <项目根>/dataflow_agent/tmps/(session_id?)/final_state.json
    遇到无法序列化的对象用 str 兜底。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)
    print(f"final_state 已保存到 {out_path}")

# ====================== 主函数 ====================== #
async def main() -> None:
    # -------- 基础路径与 session 处理 -------- #
    PROJECT_ROOT: Path = get_project_root()  # e.g. /mnt/DataFlow/lz/proj/dataflow-agent-kupasi
    TMPS_DIR: Path = PROJECT_ROOT / "dataflow_agent" / "tmps"

    session_id = base64.urlsafe_b64encode("username=zhaoyang".encode()).decode()
    SESSION_DIR: Path = TMPS_DIR / session_id
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    # -------- 构造请求 DFRequest -------- #
    python_file_path = SESSION_DIR / "my_pipeline.py"

    req = Paper2FigureRequest(
        language="en",
        chat_api_url="http://123.129.219.111:3000/v1",
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model="gpt-4o",
        sam2_model="/data0/hzy/paper2figure/models/facebook/sam2.1-hiera-tiny",
        gen_fig_model="gemini-3-pro-image-preview",
        json_file=f"{PROJECT_ROOT}/tests/test.jsonl",
        target="Create Figures For Papers",
        python_file_path=str(python_file_path),  # pipeline 的输出脚本路径
        need_debug= False,  # 是否需要 Debug
        max_debug_rounds= 3,
        session_id=session_id,
        cache_dir="dataflow_cache"
    )

    # -------- 初始化 Paper2FigureState -------- #
    state = Paper2FigureState(request=req, messages=[])
    state.temp_data["round"] = 0
    state.paper_file = 'data/GovBench_ACL_Template.pdf'
    # state.input_type = 'FIGURE'
    # state.fig_draft_path = '/home/ubuntu/liuzhou/hzy/paper2figure/gen_gemini2.png'
    state.result_path = f'./outputs/{time.strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(state.result_path, exist_ok=True)

    state.mask_detail_level = 3
    
    # state.size = '1024x1024'
    # -------- 构建并运行流水线图 -------- #
    # graph_builder = create_p2fig_graph()
    # graph = graph_builder.build()

    # # （可选）展示 mermaid 图
    # try:
    #     png_image = graph.get_graph().draw_mermaid_png()
    #     display(Image(png_image))
    #     (SESSION_DIR / "graph.png").write_bytes(png_image)
    #     print(f"\n流水线图已保存到 {SESSION_DIR / 'graph.png'}")
    # except Exception as e:
    #     print(f"生成 PNG 失败，请确保已正确安装 pygraphviz 和 Graphviz：{e}")

    # -------- 异步执行 -------- #
    final_state: Paper2FigureState = await run_workflow("paper2fig", state)

    # -------- 保存最终 State -------- #
    save_final_state_json(final_state=final_state, out_dir=SESSION_DIR)


if __name__ == "__main__":
    asyncio.run(main())