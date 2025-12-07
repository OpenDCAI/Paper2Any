from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

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



class Paper2FigureRequest(BaseModel):
    """
    Paper2Figure 的请求参数定义。
    """

    # ---------------------- 基础 LLM 设置 ----------------------
    chat_api_url: str = "http://123.129.219.111:3000/v1/"
    # 与大模型交互使用的 API URL

    chat_api_key: str = "fill the key"
    # chat_api_url 对应的 API KEY；用于访问后端 LLM 服务

    api_key: str = ""
    # 如果使用第三方外部 API（如 OpenAI），在此填写外部 API Key；为空则使用内部服务

    llm_model: str = "gpt-4o"
    # 用于执行理解、抽象、描述生成的文本模型名称

    gen_fig_model: str = "gemini-3-pro-image-preview"
    # 用于生成插图 / 构图草图的图像模型名称
    # 模型名和雨茶官网一致


    # ---------------------- 输入类型设置 ----------------------
    input_type: Literal["PDF", "TEXT", "FIGURE"] = "PDF"
    # 指定输入内容的形式：
    # - "PDF": 输入为 PDF 文件路径
    # - "TEXT": 输入为纯文本内容
    # - "FIGURE": 输入为图片文件路径（如 JPG/PNG），用于图像解析或转图


    input_content: str = ''
    # 输入内容本体（字符串类型），含义由 input_type 决定：
    # - 当 input_type = "PDF"   时：input_content 为 PDF **文件路径**
    # - 当 input_type = "FIGURE" 时：input_content 为 图片 **文件路径**
    # - 当 input_type = "TEXT"   时：input_content 为 **纯文本内容本身**
    # 注意：此参数始终为字符串，不做类型变化。


    # ---------------------- 输出图像比例设置 ----------------------
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"] = "16:9"
    # 指定生成图像的长宽比，例如：
    # 1:1（正方形）、16:9（横向宽屏）、9:16（竖屏）、4:3、3:4 以及 21:9 超宽屏。


class Paper2FigureResponse(BaseModel):
    success: bool
    ppt_filename: str = '' # 生成PPT的路径