from __future__ import annotations

"""
脚本用途: 运行 paper2fig_with_sam 工作流，将论文 PDF 转为图示并导出 PPT。

使用方式:
  python script/run_paper2figure_with_sam.py
"""

import asyncio

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root


# ================== 可配置参数 ==================
# 论文 PDF 文件路径（默认使用 tests 目录下示例 PDF）
PAPER_FILE: str = f"{get_project_root()}/tests/2506.02454v1.pdf"

# 图像生成模型名称（用于生成实验图 / 示意图）
# 可选示例:
#   - "gemini-3-pro-image-preview"
#   - "gemini-2.5-flash-image-preview"
GEN_FIG_MODEL: str = "gemini-3-pro-image-preview"

# 工作流名称（需与 workflow 注册名称一致）
WORKFLOW_NAME: str = "paper2fig_with_sam"
# =================================================


async def run_paper2figure_with_sam() -> Paper2FigureState:
    """
    执行 paper2figure_with_sam 工作流的主流程。

    返回:
        Paper2FigureState: 工作流执行结束后的最终状态对象。
    """
    req = Paper2FigureRequest(
        gen_fig_model=GEN_FIG_MODEL,
    )

    state = Paper2FigureState(
        messages=[],
        agent_results={},
        request=req,
        paper_file=PAPER_FILE,
    )

    final_state: Paper2FigureState = await run_workflow(WORKFLOW_NAME, state)
    return final_state


def main() -> None:
    """
    同步入口: 运行异步主流程并打印关键输出路径与调试信息。
    """
    final_state = asyncio.run(run_paper2figure_with_sam())

    print("\n=== Workflow finished ===")
    print(f"paper_file      : {getattr(final_state, 'paper_file', None)}")
    print(f"ppt_path        : {getattr(final_state, 'ppt_path', None)}")
    print(f"fig_draft_path  : {getattr(final_state, 'fig_draft_path', None)}")
    print(f"fig_layout_path : {getattr(final_state, 'fig_layout_path', None)}")

    print("\n=== layout_items (len) ===")
    print(len(getattr(final_state, "layout_items", []) or []))

    print("\n=== fig_mask (len) ===")
    print(len(getattr(final_state, "fig_mask", []) or []))

    # 可选: 打印 agent_results 以便调试
    print("\n=== agent_results ===")
    print(getattr(final_state, "agent_results", {}))


if __name__ == "__main__":
    main()
