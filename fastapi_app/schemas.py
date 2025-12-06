from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ============================================================
# 与 dataflow_agent.state 中 Request / State 的对应关系说明
# ------------------------------------------------------------
# 1. 本文件中的 *Request 模型，尽量与 dataflow_agent.state 中
#    对应的 xxxRequest / xxxState.request 字段保持语义对齐。
#
#    - OperatorWriteRequest
#        ≈ dataflow_agent.state.PromptWritingState.request
#        （其类型为 DFRequest）
#
#    - PipelineRecommendRequest
#        ≈ dataflow_agent.state.DFRequest
#
# 2. 本文件中的 *Response 模型，与前缀相同的 *Request 一一对应：
#
#    - OperatorWriteResponse        ↔ OperatorWriteRequest
#    - PipelineRecommendResponse    ↔ PipelineRecommendRequest
#
# 3. 设计原则：
#    - Request 层：用于 FastAPI 入参，字段含义应与对应 State.request
#      中的字段保持一致或可直接映射（如 target / language / model /
#      chat_api_url / need_debug / max_debug_rounds / session_id 等）。
#    - Response 层：用于 FastAPI 出参，其字段语义应尽量复用
#      dataflow_agent.state 中 DFState / PromptWritingState 等的字段
#      （如 matched_ops / execution_result / agent_results 等），
#      使 API 层与 Agent 层之间的状态转换清晰、可追踪。
# ============================================================


# ===================== 通用基础模型 =====================


class APIError(BaseModel):
    code: str
    message: str


# ===================== 算子编写相关 =====================


class OperatorWriteRequest(BaseModel):
    """
    对标 gradio_app.pages.operator_write 中 generate_operator 的输入参数。
    """
    target: str
    category: str = "Default"
    json_file: Optional[str] = None

    chat_api_url: str = "http://123.129.219.111:3000/v1/"
    api_key: str = ""
    model: str = "gpt-4o"
    language: str = "en"

    need_debug: bool = False
    max_debug_rounds: int = 3

    # 若提供，则用于保存生成的算子代码
    output_path: Optional[str] = None


class OperatorWriteResponse(BaseModel):
    success: bool
    code: str
    matched_ops: List[str]
    execution_result: Dict[str, Any]
    debug_runtime: Dict[str, Any]
    agent_results: Dict[str, Any]
    log: str


# ===================== 流水线推荐 / 导出相关 =====================


class PipelineRecommendRequest(BaseModel):
    """
    对标 gradio_app.utils.wf_pipeine_rec.run_pipeline_workflow 的参数。
    """
    target: str
    json_file: str

    need_debug: bool = False
    session_id: str = "default"

    chat_api_url: str = "http://123.129.219.111:3000/v1/"
    api_key: str = ""
    model_name: str = "gpt-4o"
    max_debug_rounds: int = 2

    chat_api_url_for_embeddings: str = ""
    embedding_model_name: str = "text-embedding-3-small"
    update_rag_content: bool = True


class PipelineRecommendResponse(BaseModel):
    success: bool
    python_file: str
    execution_result: Dict[str, Any]
    agent_results: Dict[str, Any]


# ===================== paper2video相关 =====================
class FeaturePaper2VideoRequest(BaseModel):
    model: str = "gpt-4o",
    chat_api_url: str = "http://123.129.219.111:3000/v1/",
    api_key: str = "",
    pdf_path: str = "",
    img_path: str = "",

class FeaturePaper2VideoResponse(BaseModel):
    success: bool
    ppt_path: str

