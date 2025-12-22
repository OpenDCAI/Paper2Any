from __future__ import annotations

"""
脚本用途: 运行 paper2technical 工作流，根据论文构建技术路线图并导出结果（如 PPT 等）。

使用方式:
  python script/run_paper2technical.py
"""

import asyncio

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root


# ================== 可配置参数 ==================
# 论文 PDF 文件路径（默认使用 tests 目录下示例 PDF）
PAPER_FILE: str = f"{get_project_root()}/tests/2506.02454v1.pdf"

# 技术路线描述的默认输入（TEXT 模式下可直接用这段文字生成技术路线图）
TECHNICAL_ROUTE_DESC: str = "This is a test description for technical route."

# 工作流名称（需与 workflow 注册名称一致）
WORKFLOW_NAME: str = "paper2technical"
# =================================================


async def run_paper2technical() -> Paper2FigureState:
    """
    执行 paper2technical 工作流的主流程。

    返回:
        Paper2FigureState: 工作流执行结束后的最终状态对象。
    """
    # 如果 workflow 支持 TEXT 模式，可以只用文字描述跳过 PDF 抽取节点
    req = Paper2FigureRequest()

    state = Paper2FigureState(
        messages=[],
        agent_results={},
        paper_idea=TECHNICAL_ROUTE_DESC,
        request=req,
        paper_file=PAPER_FILE,
    )

    final_state: Paper2FigureState = await run_workflow(WORKFLOW_NAME, state)
    return final_state


def main() -> None:
    """
    同步入口: 运行异步主流程并打印关键输出信息。
    """
    final_state = asyncio.run(run_paper2technical())

    print("\n=== Workflow finished ===")
    print(f"paper_file      : {getattr(final_state, 'paper_file', None)}")
    print(f"paper_idea      : {getattr(final_state, 'paper_idea', None)}")

    # 如果 workflow 已经实现了 PPT 生成逻辑，可以查看 ppt_path
    ppt_path = getattr(final_state, "ppt_path", None)
    print(f"ppt_path        : {ppt_path}")

    print("\n=== agent_results ===")
    print(getattr(final_state, "agent_results", {}))

    if hasattr(final_state, "messages") and final_state.messages:
        print("\n=== messages ===")
        for msg in final_state.messages:
            print(f"- {msg}")


if __name__ == "__main__":
    main()
