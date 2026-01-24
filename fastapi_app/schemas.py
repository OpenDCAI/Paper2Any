from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from dataflow_agent.utils import get_project_root
from pydantic import BaseModel, Field
from fastapi_app.config import settings

# ===================== 通用基础模型 =====================


class APIError(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """统一错误响应模型"""
    error: str
    code: str = "INTERNAL_ERROR"  # 错误代码，如VALIDATION_ERROR, WORKFLOW_ERROR等
    details: Optional[Dict] = None


# ===================== paper2video相关 =====================


class FeaturePaper2VideoRequest(BaseModel):
    model: str = settings.PAPER2VIDEO_DEFAULT_MODEL,
    chat_api_url: str = settings.DEFAULT_LLM_API_URL,
    api_key: str = "",
    pdf_path: str = "",
    img_path: str = "",
    language: str = "",


class FeaturePaper2VideoResponse(BaseModel):
    success: bool
    ppt_path: str


# ===================== LLM Verification =====================


class VerifyLlmRequest(BaseModel):
    api_url: str
    api_key: str
    model: str = settings.MODEL_GPT_4O


class VerifyLlmResponse(BaseModel):
    success: bool
    error: Optional[str] = None


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

    chat_api_url: str = settings.DEFAULT_LLM_API_URL
    # 与大模型交互使用的 API URL

    # ---------------------- 图类型 & 难度设置 ----------------------
    figure_complex: str = "easy"
    # 绘图难度：仅在 graph_type == "model_arch" 时生效，前端透传 easy/mid/hard

    chat_api_key: str = "fill the key"
    # chat_api_url 对应的 API KEY；用于访问后端 LLM 服务

    api_key: str = ""
    # 如果使用第三方外部 API（如 OpenAI），在此填写外部 API Key；为空则使用内部服务

    model: str = settings.PAPER2FIGURE_TEXT_MODEL
    # 用于执行理解、抽象、描述生成的文本模型名称

    gen_fig_model: str = settings.PAPER2FIGURE_IMAGE_MODEL
    # 用于生成插图 / 构图草图的图像模型名称

    bg_rm_model: str = f"{get_project_root()}/models/RMBG-2.0"

    # 新增模型参数
    vlm_model: str = settings.PAPER2FIGURE_VLM_MODEL
    chart_model: str = settings.PAPER2FIGURE_CHART_MODEL
    fig_desc_model: str = settings.PAPER2FIGURE_DESC_MODEL
    technical_model: str = settings.PAPER2FIGURE_TECHNICAL_MODEL

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

    email: str = ""

    # ---------------------- 重新生成/编辑相关 ----------------------
    edit_prompt: str = ""
    # 用户重新生成时提供的提示词

    prev_image: str = ""
    # 上一次生成的图片路径（用于 image-to-image 或 edit 模式）

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

class PageContentRequest(BaseModel):
    """专用于pagecontent生成的请求模型"""
    chat_api_url: str
    api_key: str
    email: Optional[str] = None
    input_type: Literal["text", "pdf", "pptx", "topic"]
    file: Optional[Any] = None  # UploadFile 在路由层处理，这里用Any占位
    text: Optional[str] = None
    model: str = settings.PAPER2PPT_OUTLINE_MODEL
    language: str = "zh"
    style: str = ""
    reference_img: Optional[Any] = None
    gen_fig_model: str = Field(...)
    page_count: int = 5
    use_long_paper: str = "false"


class OutlineRefineRequest(BaseModel):
    """Refine outline based on user feedback without re-parsing input."""
    chat_api_url: str
    api_key: str
    email: Optional[str] = None
    model: str = settings.PAPER2PPT_OUTLINE_MODEL
    language: str = "zh"
    result_path: Optional[str] = None
    outline_feedback: str
    pagecontent: str


class PPTGenerationRequest(BaseModel):
    """专用于PPT生成/编辑的请求模型"""
    img_gen_model_name: str
    chat_api_url: str
    api_key: str
    email: Optional[str] = None
    style: str = ""
    reference_img: Optional[Any] = None
    aspect_ratio: str = "16:9"
    language: str = "en"
    model: str = settings.PAPER2PPT_CONTENT_MODEL
    get_down: str = "false"
    all_edited_down: str = "false"
    result_path: str
    pagecontent: Optional[str] = None
    page_id: Optional[int] = None
    edit_prompt: Optional[str] = None


class FullPipelineRequest(BaseModel):
    """专用于完整流水线的请求模型"""
    img_gen_model_name: str
    chat_api_url: str
    api_key: str
    email: Optional[str] = None
    input_type: Literal["text", "pdf", "pptx"]
    file: Optional[Any] = None
    text: Optional[str] = None
    language: str = "zh"
    aspect_ratio: str = "16:9"
    style: str = ""
    model: str = settings.PAPER2PPT_DEFAULT_MODEL
    use_long_paper: str = "false"


class Paper2PPTRequest(BaseModel):
    """
    Paper2PPT 的请求参数定义。

    目前直接复用 Paper2FigureRequest 的字段语义，仅名称区分，
    方便在 FastAPI 层与具体 workflow 解耦。
    """

    # ---------------------- 基础 LLM 设置 ----------------------
    language: str = "en"
    chat_api_url: str = settings.DEFAULT_LLM_API_URL

    # ---------------------- 图类型 & 难度设置 ----------------------
    chat_api_key: str = "fill the key"
    api_key: str = ""
    # 用于对话的模型
    model: str = settings.PAPER2PPT_DEFAULT_MODEL

    ref_img : str = ""

    gen_fig_model: str = settings.PAPER2PPT_IMAGE_GEN_MODEL
    # bg_rm_model: str = f"{get_project_root()}/models/RMBG-2.0"

    # 新增模型参数
    vlm_model: str = settings.PAPER2PPT_VLM_MODEL
    chart_model: str = settings.PAPER2PPT_CHART_MODEL
    fig_desc_model: str = settings.PAPER2PPT_DESC_MODEL
    technical_model: str = settings.PAPER2PPT_TECHNICAL_MODEL

    # ---------------------- 输入类型设置 ----------------------
    input_type: Literal["PDF", "TEXT", "PPT", "TOPIC", "FIGURE"] = "PDF"
    input_content: str = ""

    # ---------------------- 输出图像比例设置 ----------------------
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"] = "16:9"
    style: str = " "
    use_long_paper: bool = False # 不使用 长文

    email: str = ""
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
