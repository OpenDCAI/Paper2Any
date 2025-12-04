from __future__ import annotations

from fastapi import APIRouter

from fastapi_app.schemas import OperatorWriteRequest, OperatorWriteResponse
from fastapi_app.workflow_adapters import run_operator_write_pipeline_api

router = APIRouter()


@router.post(
    "/write",
    response_model=OperatorWriteResponse,
    summary="基于工作流的算子编写",
    description=(
        "封装 wf_pipeline_write 工作流，按照目标描述自动匹配参考算子并生成新的算子代码，"
        "行为对齐 gradio_app.pages.operator_write 页面。"
    ),
)
async def operator_write_endpoint(body: OperatorWriteRequest) -> OperatorWriteResponse:
    """
    算子编写接口。

    - 输入参数与 gradio_app 页面上的表单字段一致（target/json_file/chat_api_url/...）
    - 内部调用 dataflow_agent.workflow.wf_pipeline_write.create_operator_write_graph
    - 返回生成的代码、匹配算子列表、执行结果、调试信息等
    """
    return await run_operator_write_pipeline_api(body)
