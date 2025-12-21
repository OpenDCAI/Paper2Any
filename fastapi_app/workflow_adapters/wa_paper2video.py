from __future__ import annotations

"""
paper2video 工作流封装。

从原来的单文件 workflow_adapters.py / paper2_modules.py 拆分而来，
只保留与 paper2video 相关的封装逻辑。
"""

import os

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2VideoRequest, Paper2VideoState

from fastapi_app.schemas import (
    FeaturePaper2VideoRequest,
    FeaturePaper2VideoResponse,
)

log = get_logger(__name__)


# ------------------- paper2video 工作流封装 -------------------


async def run_paper_to_video_api(
    req: FeaturePaper2VideoRequest,
) -> FeaturePaper2VideoResponse:
    """
    基于 wf_paper2video.create_paper2video_graph 的封装。

    对标 gradio_app.pages.paper2video.run_paper2video_workflow。
    """
    # 设置环境变量
    if req.api_key:
        os.environ["DF_API_KEY"] = req.api_key
    else:
        req.api_key = os.getenv("DF_API_KEY", "sk-dummy")

    # 构造 DFRequest / DFState
    paper_req = Paper2VideoRequest(
        chat_api_url=req.chat_api_url,
        api_key=req.api_key,
        model=req.model,
        paper_pdf_path=req.pdf_path,
        user_imgs_path=req.img_path,
        language=req.language,
    )
    state = Paper2VideoState(request=paper_req, messages=[])

    from dataflow_agent.workflow.wf_paper2video import create_paper2video_graph

    graph = create_paper2video_graph().build()
    final_state: Paper2VideoState = await graph.ainvoke(state)

    result = {
        "success": True,
        "final_state": final_state,
    }

    # 提取输出的 ppt 文件
    try:
        if isinstance(final_state, dict):
            ppt_path = final_state.get("ppt_path", [])
        else:
            ppt_path = getattr(final_state, "ppt_path", [])
        result["ppt_path"] = ppt_path or []
    except Exception as e:  # pragma: no cover - 仅日志
        if "log" in globals():
            log.warning(f"提取 pdf 的 ppt 失败: {e}")
        result["ppt_path"] = []

    return FeaturePaper2VideoResponse(
        success=result.get("success", False),
        ppt_path=result.get("ppt_path", ""),
    )
