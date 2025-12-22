from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from dataflow_agent.utils import get_project_root
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


# ===================== paper2video相关 =====================


class FeaturePaper2VideoRequest(BaseModel):
    model: str = "gpt-4o",
    chat_api_url: str = "http://123.129.219.111:3000/v1/",
    api_key: str = "",
    pdf_path: str = "",
    img_path: str = "",
    language: str = "",


class FeaturePaper2VideoResponse(BaseModel):
    success: bool
    ppt_path: str


# ===================== paper2figure 相关 =====================


class Paper2FigureRequest(BaseModel):
    """
    Paper2Figure 的请求参数定义。

    注意：
    - 为了兼容 dataflow_agent 内部对 state.request 的访问，
      这里额外提供 language 字段，并实现一个简单的 get 方法，
      使其既能通过属性访问（.language），也能通过 dict 风格访问（.get）。
    """

    # ---------------------- 基础 LLM 设置 ----------------------
    language: str = "en"
    # 工作流内部有角色会访问 state.request.language

    chat_api_url: str = "http://123.129.219.111:3000/v1/"
    # 与大模型交互使用的 API URL

    # ---------------------- 图类型 & 难度设置 ----------------------
    figure_complex: str = "easy"
    # 绘图难度：仅在 graph_type == "model_arch" 时生效，前端透传 easy/mid/hard

    chat_api_key: str = "fill the key"
    # chat_api_url 对应的 API KEY；用于访问后端 LLM 服务

    api_key: str = ""
    # 如果使用第三方外部 API（如 OpenAI），在此填写外部 API Key；为空则使用内部服务

    model: str = "gpt-4o"
    # 用于执行理解、抽象、描述生成的文本模型名称

    gen_fig_model: str = "gemini-3-pro-image-preview"
    # 用于生成插图 / 构图草图的图像模型名称
    # 模型名和雨茶官网一致

    bg_rm_model: str = f"{get_project_root()}/models/RMBG-2.0"

    # ---------------------- 输入类型设置 ----------------------
    input_type: Literal["PDF", "TEXT", "FIGURE"] = "PDF"
    # 指定输入内容的形式：
    # - "PDF": 输入为 PDF 文件路径
    # - "TEXT": 输入为纯文本内容
    # - "FIGURE": 输入为图片文件路径（如 JPG/PNG），用于图像解析或转图

    input_content: str = ""
    # 输入内容本体（字符串类型），含义由 input_type 决定：
    # - 当 input_type = "PDF"   时：input_content 为 PDF **文件路径**
    # - 当 input_type = "FIGURE" 时：input_content 为 图片 **文件路径**
    # - 当 input_type = "TEXT"   时：input_content 为 **纯文本内容本身**
    # 注意：此参数始终为字符串，不做类型变化。

    # ---------------------- 输出图像比例设置 ----------------------
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"] = "16:9"
    # 图类型：模型架构图 / 技术路线图 / 实验数据图
    graph_type: Literal["model_arch", "tech_route", "exp_data"] = "model_arch"
    # 风格：卡通 / 写实（具体取值前端透传）
    style: str = "cartoon"
    # 指定生成图像的长宽比，例如：
    # 1:1（正方形）、16:9（横向宽屏）、9:16（竖屏）、4:3、3:4 以及 21:9 超宽屏。

    invite_code: str = ""

    # ---------------------- 兼容 dict 风格访问 ----------------------
    def get(self, key: str, default=None):
        """
        兼容 dataflow_agent 内部对 request 使用 dict.get("key") 的写法。
        未找到属性时返回 default。
        """
        return getattr(self, key, default)


class Paper2FigureResponse(BaseModel):
    success: bool
    ppt_filename: str = ""  # 生成PPT的路径
    svg_filename: str = ""  # 技术路线 SVG 源文件路径（graph_type=tech_route 时有效）
    svg_image_filename: str = ""  # 技术路线 PNG 渲染图路径（graph_type=tech_route 时有效）
    all_output_files: List[str] = []  # 本次任务产生的所有输出文件路径（稍后在路由层转换为 URL）


# ===================== paper2ppt 相关 =====================


class Paper2PPTRequest(BaseModel):
    """
    Paper2PPT 的请求参数定义。

    目前直接复用 Paper2FigureRequest 的字段语义，仅名称区分，
    方便在 FastAPI 层与具体 workflow 解耦。
    """

    # ---------------------- 基础 LLM 设置 ----------------------
    language: str = "en"
    chat_api_url: str = "http://123.129.219.111:3000/v1/"

    # ---------------------- 图类型 & 难度设置 ----------------------
    chat_api_key: str = "fill the key"
    api_key: str = ""
    # 用于对话的模型
    model: str = "gpt-5.1"

    gen_fig_model: str = "gemini-3-pro-image-preview"
    # bg_rm_model: str = f"{get_project_root()}/models/RMBG-2.0"

    # ---------------------- 输入类型设置 ----------------------
    input_type: Literal["PDF", "TEXT", "PPT"] = "PDF"
    input_content: str = ""

    # ---------------------- 输出图像比例设置 ----------------------
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"] = "16:9"
    style: str = " "

    reference_img: str = ""

    invite_code: str = ""
    # 生成的ppt页数；
    page_count: int = 5

    all_edited_down: bool = False
    use_ai_edit: bool = False

    def get(self, key: str, default=None):
        """
        兼容 dataflow_agent 内部对 request 使用 dict.get("key") 的写法。
        未找到属性时返回 default。
        """
        return getattr(self, key, default)


class Paper2PPTResponse(BaseModel):
    """
    Paper2PPT 的响应模型。

    workflow_adapters.paper2ppt 会返回这些字段（或其中子集）：
    - pagecontent: paper2page_content 的结构化结果
    - result_path: 本次任务输出目录（后端内部路径；路由层通常会再转 URL）
    - ppt_pdf_path / ppt_pptx_path: paper2ppt 导出的最终文件路径
    - all_output_files: 本次任务输出目录下扫描到的相关文件（路由层转 URL 后返回）
    """
    success: bool = True

    ppt_pdf_path: str = ""
    ppt_pptx_path: str = ""
    pagecontent: List[Dict[str, Any]] = []
    result_path: str = ""
    all_output_files: List[str] = []
