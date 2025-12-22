from __future__ import annotations

"""
脚本用途: 运行 paper2page_content + paper2ppt 工作流，将论文 PDF 转为按页内容 + PPT。

使用方式:
  python script/run_paper2page_content.py
"""

import asyncio
from pathlib import Path

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root


# ================== 可配置参数 ==================
# 输入论文 PDF 文件路径（默认使用 tests 目录下示例 PDF）
PDF_PATH: str = str(Path(get_project_root()) / "tests" / "2512.16676v1.pdf")

# 文本大模型名称，用于生成页内容 / PPT 文案
LLM_MODEL_NAME: str = "gpt-5.1"

# Chat API 地址
CHAT_API_URL: str = "https://api.apiyi.com/v1"

# PPT 风格描述（如: 学校模板、语言、版式偏好等）
PPT_STYLE: str = "北京大学风格；英文；"

# MinerU 抽取后最大处理页数（避免长文档成本过高）
MAX_PAGE_COUNT: int = 3

# 第一步内容抽取工作流名称 (需与 workflow 注册名称一致)
WORKFLOW_NAME_PAGE_CONTENT: str = "paper2page_content"

# 第二步生成 PPT 工作流名称 (需与 workflow 注册名称一致)
WORKFLOW_NAME_PPT: str = "paper2ppt"
# =================================================


async def run_paper2page_content_pipeline() -> Paper2FigureState:
    """
    执行 paper2page_content + paper2ppt 工作流的主流程。

    返回:
        Paper2FigureState: 工作流执行结束后的最终状态对象。
    """
    req = Paper2FigureRequest()
    # 输入类型为 PDF，内部会先走 MinerU 抽取结构化内容
    req.input_type = "PDF"
    # 使用的文本大模型名称
    req.model = LLM_MODEL_NAME
    # 需要处理的最大页数
    req.page_count = MAX_PAGE_COUNT
    # Chat API URL
    req.chat_api_url = CHAT_API_URL
    # PPT 风格描述
    req.style = PPT_STYLE
    # 是否已经完成人工编辑（True 表示直接生成最终 PPT）
    req.all_edited_down = True

    state = Paper2FigureState(
        messages=[],
        agent_results={},
        request=req,
        # 论文 PDF 文件路径
        paper_file=PDF_PATH,
    )

    # 第一步: PDF -> MinerU -> 页内容抽取
    final_state: Paper2FigureState = await run_workflow(
        WORKFLOW_NAME_PAGE_CONTENT, state
    )
    # 第二步: 基于页内容生成 PPT
    final_state = await run_workflow(WORKFLOW_NAME_PPT, final_state)
    return final_state


def main() -> None:
    """
    同步入口: 运行异步主流程并打印关键结果。
    """
    final_state = asyncio.run(run_paper2page_content_pipeline())

    print("\n=== Workflow finished ===")
    print(f"paper_file      : {getattr(final_state, 'paper_file', None)}")
    print(f"minueru_output_len: {len(getattr(final_state, 'minueru_output', '') or '')}")
    print(f"pagecontent_len : {len(getattr(final_state, 'pagecontent', '') or '')}")
    print(f"ppt_path        : {getattr(final_state, 'ppt_path', None)}")

    # 可选: 打印部分 pagecontent 预览
    pagecontent = getattr(final_state, "pagecontent", None)
    if isinstance(pagecontent, str) and pagecontent:
        print("\n=== pagecontent preview (前 500 字) ===")
        print(pagecontent[:500])


if __name__ == "__main__":
    main()
