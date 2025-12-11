"""
测试 paper2figure_with_sam workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
运行方式:
  pytest tests/test_paper2figure_with_sam.py -v -s
  或直接: python tests/test_paper2figure_with_sam.py
"""

from __future__ import annotations
import asyncio
import pytest

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root

# ============ 核心异步流程 ============
async def run_paper2figure_with_sam_pipeline() -> Paper2FigureState:
    """
    执行 paper2figure_with_sam 工作流的测试流程
    """
    req = Paper2FigureRequest(
        gen_fig_model = "gemini-3-pro-image-preview"
    )
# gemini-3-pro-image-preview gemini-2.5-flash-image-preview
    state = Paper2FigureState(
        messages=[],
        agent_results={},
        # paper_idea="This is a test description for paper2figure_with_sam.",
        request=req,
        paper_file=f"{get_project_root()}/tests/2506.02454v1.pdf",
    )

    # 对应 wf_paper2figure_with_sam.py 中的 @register("paper2fig_with_sam")
    final_state: Paper2FigureState = await run_workflow("paper2fig_with_sam", state)
    return final_state


# ============ pytest 入口 ============
@pytest.mark.asyncio
async def test_paper2figure_with_sam_pipeline():
    """
    测试 paper2fig_with_sam 工作流的完整流程
    """
    final_state = await run_paper2figure_with_sam_pipeline()

    assert final_state is not None, "final_state 不应为 None"
    assert hasattr(final_state, "agent_results"), "state 应包含 agent_results"

    # 关键产物的弱检查
    # 原始带内容图
    if hasattr(final_state, "fig_draft_path") and final_state.fig_draft_path:
        assert isinstance(final_state.fig_draft_path, str)

    # 空框模板图
    if hasattr(final_state, "fig_layout_path") and final_state.fig_layout_path:
        assert isinstance(final_state.fig_layout_path, str)

    # SAM 背景布局元素
    if hasattr(final_state, "layout_items") and final_state.layout_items is not None:
        assert isinstance(final_state.layout_items, list)

    # MinerU 内容元素
    if hasattr(final_state, "fig_mask") and final_state.fig_mask is not None:
        assert isinstance(final_state.fig_mask, list)

    # PPT 输出
    if hasattr(final_state, "ppt_path") and final_state.ppt_path:
        assert isinstance(final_state.ppt_path, str)

    # -- 调试输出，可按需保留 --
    print("\n=== agent_results ===")
    print(final_state.agent_results)

    print("\n=== fig_draft_path ===")
    print(getattr(final_state, "fig_draft_path", None))

    print("\n=== fig_layout_path ===")
    print(getattr(final_state, "fig_layout_path", None))

    print("\n=== layout_items (len) ===")
    print(len(getattr(final_state, "layout_items", []) or []))

    print("\n=== fig_mask (len) ===")
    print(len(getattr(final_state, "fig_mask", []) or []))

    print("\n=== ppt_path ===")
    print(getattr(final_state, "ppt_path", None))


# ============ 直接 python 执行 ============
if __name__ == "__main__":
    asyncio.run(run_paper2figure_with_sam_pipeline())
