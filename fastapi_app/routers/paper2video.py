from __future__ import annotations

from fastapi import APIRouter

from fastapi_app.schemas import FeaturePaper2VideoRequest, FeaturePaper2VideoResponse
from fastapi_app.workflow_adapters import run_paper_to_video_api

router = APIRouter()

@router.post(
    "/paper2video",
    response_model=FeaturePaper2VideoResponse,
    summary="将 Paper 自动化转变成汇报 video",
    description=(
        "封装 wf_paper2video 工作流，将学术论文自动化变成学术汇报视频"
        "行为对齐 gradio_app.pages.paper2video 页面。"
    ),
)
async def paper2video_endpoint(body: FeaturePaper2VideoRequest) -> FeaturePaper2VideoResponse:
    """
    算子编写接口。

    - 输入参数与 gradio_app 页面上的表单字段一致（target/json_file/chat_api_url/...）
    - 内部调用 dataflow_agent.workflow.wf_pipeline_write.create_operator_write_graph
    - 返回生成的代码、匹配算子列表、执行结果、调试信息等
    """
    return await run_paper_to_video_api(body)
