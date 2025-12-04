from __future__ import annotations

from fastapi import APIRouter

from fastapi_app.schemas import (
    PipelineRecommendRequest,
    PipelineRecommendResponse,
)
from fastapi_app.workflow_adapters import run_pipeline_recommend_api

router = APIRouter()


@router.post(
    "/recommend",
    response_model=PipelineRecommendResponse,
    summary="基于工作流的 Pipeline 推荐与导出",
    description=(
        "封装 wf_pipeline_recommend_extract_json 工作流，"
        "根据目标描述与样例数据自动推荐算子组合并生成可执行的 pipeline.py，"
        "行为对齐 gradio_app.utils.wf_pipeine_rec.run_pipeline_workflow。"
    ),
)
async def pipeline_recommend_endpoint(
    body: PipelineRecommendRequest,
) -> PipelineRecommendResponse:
    """
    流水线推荐接口。

    - 输入参数与 gradio_app 中的 run_pipeline_workflow 一致（target/json_file/chat_api_url/...）
    - 内部调用 dataflow_agent.workflow.wf_pipeline_recommend_extract_json.create_pipeline_graph
    - 返回生成的 python_file 路径、debug_history、agent_results 等
    """
    return await run_pipeline_recommend_api(body)
