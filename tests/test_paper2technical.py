"""
测试 paper2technical workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
运行方式:
  pytest tests/test_paper2technical.py -v -s
  或直接: python tests/test_paper2technical.py
"""

from __future__ import annotations
import asyncio
import pytest

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow


# ============ 核心异步流程 ============
async def run_paper2technical_pipeline() -> Paper2FigureState:
    """
    执行 paper2technical 工作流的测试流程
    """
    # TEXT 模式下，跳过 PDF 抽取节点，直接进入 technical_route_desc_generator
    req = Paper2FigureRequest()

    state = Paper2FigureState(
        messages=[],
        agent_results={},
        paper_idea="This is a test description for technical route.",
        request=req,
        paper_file = f"{get_project_root()}/tests/2506.02454v1.pdf"
    )

    # final_state: Paper2FigureState = await run_workflow("paper2technical", state)
    final_state: Paper2FigureState = await run_workflow("paper2technical", state)
    return final_state


# ============ pytest 入口 ============
@pytest.mark.asyncio
async def test_paper2technical_pipeline():
    """
    测试 paper2technical 工作流的完整流程
    """
    final_state = await run_paper2technical_pipeline()

    assert final_state is not None, "final_state 不应为 None"
    assert hasattr(final_state, "agent_results"), "state 应包含 agent_results"

    # -- 调试输出，可按需保留 --
    print("\n=== agent_results ===")
    print(final_state.agent_results)

    if hasattr(final_state, "messages") and final_state.messages:
        print("\n=== messages ===")
        for msg in final_state.messages:
            print(f"- {msg}")

    # 可选弱断言: 如果 workflow 已经实现了 PPT 生成逻辑，可以检查 ppt_path 类型
    if hasattr(final_state, "ppt_path") and final_state.ppt_path:
        assert isinstance(final_state.ppt_path, str)


# ============ 直接 python 执行 ============
if __name__ == "__main__":
    asyncio.run(run_paper2technical_pipeline())
