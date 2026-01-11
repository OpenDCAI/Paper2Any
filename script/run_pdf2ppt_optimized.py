from __future__ import annotations

"""
脚本用途: 运行 pdf2ppt_optimized 工作流，将输入 PDF 转换为 PPT 文件。
(Hybrid Text Extraction + Dynamic Sizing + Smart Style)

使用方式:
  python script/run_pdf2ppt_optimized.py
"""

import asyncio
import os
from pathlib import Path

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root


# ================== 可配置参数 ==================
# 输入 PDF 文件路径（默认使用 tests 目录下示例 PDF）
PDF_PATH: str = str(Path(get_project_root()) / "tests" / "test_03.pdf")

# 工作流名称（需与 workflow 注册名称一致）
WORKFLOW_NAME: str = "pdf2ppt_optimized"
# =================================================


async def run_pdf2ppt_optimized() -> Paper2FigureState:
    """
    执行 pdf2ppt_optimized 工作流的主流程：
    - 使用配置的 PDF 作为输入
    - 跑完整 workflow，拿到最终 state

    返回:
        Paper2FigureState: 工作流执行结束后的最终状态对象。
    """
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        print(f"Warning: Default PDF not found at {pdf_path}, trying to find any pdf in tests/")
        tests_dir = Path(get_project_root()) / "tests"
        pdfs = list(tests_dir.glob("*.pdf"))
        if pdfs:
            pdf_path = pdfs[0]
            print(f"Using fallback PDF: {pdf_path}")
        else:
            print("No PDF found for testing.")
            return None

    # 构造最简 request / state
    req = Paper2FigureRequest()
    state = Paper2FigureState(
        messages=[],
        request=req,
    )
    # 把 pdf 路径挂到 state 上，供 workflow 使用
    state.pdf_file = str(pdf_path)
    print(f"输入 PDF: {state.pdf_file}")

    final_state: Paper2FigureState = await run_workflow(WORKFLOW_NAME, state)
    return final_state


def main() -> None:
    """
    同步入口: 运行异步主流程并打印输出 PPT 路径等关键信息。
    """
    final_state = asyncio.run(run_pdf2ppt_optimized())

    print("\n=== Workflow finished ===")
    if final_state is None:
        print("Workflow aborted due to missing input.")
        return

    ppt_path = getattr(final_state, "ppt_path", None)
    print(f"ppt_path: {ppt_path}")
    if ppt_path and os.path.exists(ppt_path):
        print("PPT 已生成且文件存在。")
    else:
        print("未检测到有效的 PPT 输出文件，请检查 workflow 配置。")


if __name__ == "__main__":
    main()
