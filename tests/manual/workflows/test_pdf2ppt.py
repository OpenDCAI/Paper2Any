"""
测试 pdf2ppt_with_sam workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
运行方式:
  cd dev_2/DataFlow-Agent
  pytest tests/test_pdf2ppt.py -v -s
  或直接: python tests/test_pdf2ppt.py
"""

from __future__ import annotations
import asyncio
import os
from pathlib import Path

import pytest

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root


# ============ 核心异步流程 ============

async def run_pdf2ppt_with_sam_pipeline() -> xxState:
    """
    执行 pdf2ppt_with_sam 工作流的最小测试流程：
    - 使用 tests 目录下的一个 PDF 作为 slides
    - 跑完整 workflow，拿到最终 state
    """
    root = get_project_root()
    pdf_path = Path(f"{root}/tests/test_03.pdf")
    assert pdf_path.exists(), f"测试 PDF 不存在: {pdf_path}"

    # 构造最简 request / state
    req = Paper2FigureRequest()
    state = Paper2FigureState(
        messages=[],
        request=req,
    )
    # 把 pdf 路径挂到 state 上，供 workflow 使用
    state.pdf_file = str(pdf_path)
    print(state.pdf_file)
# pdf2ppt_with_sam_ocr_mineru pdf2ppt_with_sam
    final_state: Paper2FigureState = await run_workflow("pdf2ppt_with_sam_ocr_mineru", state)
    return final_state


# ============ pytest 入口 ============

@pytest.mark.asyncio
async def test_pdf2ppt_with_sam_pipeline():
    """
    测试 pdf2ppt_with_sam 工作流的完整流程：
    - 能跑完不异常
    - 最终 state 上有 ppt_path 且文件存在
    """
    final_state = await run_pdf2ppt_with_sam_pipeline()

    assert final_state is not None, "final_state 不应为 None"

    ppt_path = getattr(final_state, "ppt_path", None)
    assert ppt_path, "workflow 应在 state.ppt_path 上写入输出 PPT 路径"
    assert os.path.exists(ppt_path), f"PPT 文件不存在: {ppt_path}"

    print("\n=== pdf2ppt_with_sam 输出 PPT ===")
    print(ppt_path)


# ============ 直接 python 执行 ============

if __name__ == "__main__":
    asyncio.run(run_pdf2ppt_with_sam_pipeline())
