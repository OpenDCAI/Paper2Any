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
    model: str = settings.PAPER2VIDEO_DEFAULT_MODEL
    chat_api_url: str = settings.DEFAULT_LLM_API_URL
    api_key: str = ""
    pdf_path: str = ""
    img_path: str = ""
    language: str = ""


class FeaturePaper2VideoResponse(BaseModel):
    success: bool
    ppt_path: str


# --------------- paper2video 两步流程：生成脚本 + 生成视频 ---------------


class ScriptPageItem(BaseModel):
    """单页脚本项，用于 generate-subtitle 响应与 generate-video 请求。"""
    page_num: int = 0
    image_url: str = ""   # 前端展示用 URL；generate-video 请求可不传或传空
    script_text: str = ""  # 该页语音脚本正文（用户可编辑）


class GenerateSubtitleResponse(BaseModel):
    """generate-subtitle 接口响应：解析论文后得到的脚本页列表、任务目录与 state_snapshot（供第二步复用 state）。"""
    success: bool = True
    message: Optional[str] = None
    result_path: str = ""   # 本次任务输出根目录（后端路径，前端后续原样回传）
    script_pages: List[Dict[str, Any]] = []  # [{ "page_num", "image_url", "script_text" }, ...]
    state_snapshot: Optional[Dict[str, Any]] = None  # 第一步 state 序列化，第二步请求时原样回传以复用 state
    all_output_files: List[str] = []  # 可选，本次任务产出文件 URL 列表，便于前端预加载


class GenerateVideoRequest(BaseModel):
    """generate-video 接口请求体（从 Form 解析后组装）。"""
    result_path: str = ""
    script_pages: str = ""  # JSON 字符串，列表 [{ "page_num", "script_text" }, ...]
    state_snapshot: Optional[str] = None  # 第一步返回的 state_snapshot 的 JSON 字符串，可选
    email: Optional[str] = None


class GenerateVideoResponse(BaseModel):
    """generate-video 接口响应：最终视频地址。"""
    success: bool = True
    message: Optional[str] = None
    video_url: str = ""   # 浏览器可访问的完整 URL，优先返回
    video_path: str = ""  # 后端本地路径，当 video_url 为空时前端可据此拼 URL



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

    resolution: str = "2K"
    # 图像分辨率：2K 或 4K，影响生成图像的质量和生成时间

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
    tec_vlm_desc_model: str = settings.PAPER2FIGURE_REF_IMG_DESC_MODEL
    chart_model: str = settings.PAPER2FIGURE_CHART_MODEL
    fig_desc_model: str = settings.PAPER2FIGURE_DESC_MODEL
    technical_model: str = settings.PAPER2FIGURE_TECHNICAL_MODEL
    tech_route_template: str = ""
    tech_route_palette: str = ""

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

    # ---------------------- 技术路线图参考图相关 ----------------------
    reference_image_path: str = ""
    # 参考图路径（用于 VLM 理解后生成类似风格的技术路线图）

    tech_route_edit_prompt: str = ""
    # 技术路线图二次编辑提示词

    # ---------------------- 兼容 dict 风格访问 ----------------------
    def get(self, key: str, default=None):
        """
        兼容 dataflow_agent 内部对 request 使用 dict.get("key") 的写法。
        未找到属性时返回 default。
        """
        return getattr(self, key, default)


class Paper2FigureResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    ppt_filename: str = ""  # 生成PPT的路径
    drawio_filename: str = ""  # DrawIO 源文件路径（model_arch image2drawio 时有效）
    svg_filename: str = ""  # 技术路线 SVG 源文件路径（graph_type=tech_route 时有效）
    svg_image_filename: str = ""  # 技术路线 PNG 渲染图路径（graph_type=tech_route 时有效）
    svg_bw_filename: str = ""  # 技术路线黑白 SVG 源文件路径（选配色时有效）
    svg_bw_image_filename: str = ""  # 技术路线黑白 PNG 渲染图路径（选配色时有效）
    svg_color_filename: str = ""  # 技术路线彩色 SVG 源文件路径（选配色时有效）
    svg_color_image_filename: str = ""  # 技术路线彩色 PNG 渲染图路径（选配色时有效）
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
    # 当 input_type=pdf 时，是否按“幻灯片图片”模式解析（跳过 MinerU 解析）
    pdf_as_slides: str = "false"
    # PPT/PDF 转图片时的渲染 DPI（None 表示使用默认值）
    render_dpi: Optional[int] = None


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


# ===================== KB Deep Research 相关 =====================

class DeepResearchRequest(BaseModel):
    mode: Literal["llm", "web"] = "llm"
    topic: str = ""
    file_paths: List[str] = []
    api_url: str
    api_key: str
    model: str = settings.MODEL_GPT_4O
    language: str = "zh"
    email: Optional[str] = None
    user_id: Optional[str] = None
    notebook_id: Optional[str] = None
    search_provider: Literal["serpapi", "google_cse", "brave"] = "serpapi"
    search_api_key: str = ""
    search_engine: Literal["google", "baidu"] = "google"
    search_num: int = 10
    google_cse_id: str = ""
    brave_summarizer: bool = True
    search_depth: int = 2
    max_queries: int = 6
    top_k_per_query: int = 5
    fetch_top_n: int = 8
    max_page_chars: int = 8000
    enable_agentic: bool = True


class DeepResearchResponse(BaseModel):
    success: bool
    report_markdown: str = ""
    report_path: str = ""
    search_results: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    output_file_id: str = ""


# ===================== Paper2Citation 相关 =====================


class CitationAuthorCandidate(BaseModel):
    openalex_author_id: str = ""
    dblp_id: str = ""
    orcid: str = ""
    display_name: str = ""
    affiliations: List[str] = []
    works_count: int = 0
    cited_by_count: int = 0
    source: str = ""


class CitationAuthorItem(BaseModel):
    openalex_author_id: str = ""
    display_name: str = ""
    affiliations: List[str] = []
    citing_works_count: int = 0


class CitationInstitutionStat(BaseModel):
    openalex_institution_id: str = ""
    display_name: str = ""
    country_code: str = ""
    type: str = ""
    citing_works_count: int = 0


class CitationWorkItem(BaseModel):
    openalex_work_id: str = ""
    doi: str = ""
    title: str = ""
    year: Optional[int] = None
    publication_date: str = ""
    venue: str = ""
    type: str = ""
    cited_by_count: int = 0
    authors: List[str] = []
    institutions: List[str] = []
    landing_page_url: str = ""


class CitationContextItem(BaseModel):
    section: str = ""
    sentence: str = ""
    paragraph: str = ""
    marker: str = ""
    confidence: float = 0.0


class CitationHonorStat(BaseModel):
    honor_label: str = ""
    count: int = 0
    matched_authors: List[Dict[str, Any]] = []


class Paper2CitationAuthorSearchRequest(BaseModel):
    author_name: str
    max_author_candidates: int = 12


class Paper2CitationAuthorSearchResponse(BaseModel):
    success: bool = True
    mode: str = "author_search"
    query: str = ""
    candidates: List[CitationAuthorCandidate] = []


class Paper2CitationAuthorDetailRequest(BaseModel):
    openalex_author_id: str = ""
    dblp_id: str = ""
    display_name: str = ""
    affiliation_hint: str = ""
    candidate_source: str = ""
    max_publications: int = 25
    max_citing_works: int = 60
    publication_page: int = 1
    publication_page_size: int = 20


class Paper2CitationAuthorDetailResponse(BaseModel):
    success: bool = True
    mode: str = "author_detail"
    query: str = ""
    best_effort_notice: str = ""
    author_profile: Dict[str, Any] = {}
    publication_stats: Dict[str, Any] = {}
    citation_stats: Dict[str, Any] = {}
    publication_pagination: Dict[str, Any] = {}
    publications: List[CitationWorkItem] = []
    citing_works: List[CitationWorkItem] = []
    citing_authors: List[CitationAuthorItem] = []
    citing_institutions: List[CitationInstitutionStat] = []
    honors_stats: List[CitationHonorStat] = []
    matched_honorees: List[Dict[str, Any]] = []


class Paper2CitationAuthorPublicationsRequest(BaseModel):
    openalex_author_id: str = ""
    dblp_id: str = ""
    display_name: str = ""
    affiliation_hint: str = ""
    candidate_source: str = ""
    max_publications: int = 25
    publication_page: int = 1
    publication_page_size: int = 20


class Paper2CitationAuthorPublicationsResponse(BaseModel):
    success: bool = True
    mode: str = "author_publications"
    query: str = ""
    best_effort_notice: str = ""
    publication_stats: Dict[str, Any] = {}
    publication_pagination: Dict[str, Any] = {}
    publications: List[CitationWorkItem] = []


class Paper2CitationPaperDetailRequest(BaseModel):
    doi_or_url: str
    max_citing_works: int = 60


class Paper2CitationPaperDetailResponse(BaseModel):
    success: bool = True
    mode: str = "paper_detail"
    query: str = ""
    best_effort_notice: str = ""
    paper_detail: Dict[str, Any] = {}
    citation_stats: Dict[str, Any] = {}
    citing_works: List[CitationWorkItem] = []
    citing_authors: List[CitationAuthorItem] = []
    citing_institutions: List[CitationInstitutionStat] = []
    honors_stats: List[CitationHonorStat] = []
    matched_honorees: List[Dict[str, Any]] = []


class Paper2CitationPaperContextRequest(BaseModel):
    target_doi_or_url: str
    citing_work_openalex_id: str = ""
    citing_work_doi_or_url: str = ""
    citing_work_title: str = ""


class Paper2CitationPaperContextResponse(BaseModel):
    success: bool = True
    mode: str = "paper_context"
    query: str = ""
    best_effort_notice: str = ""
    source_url: str = ""
    target_reference_match: Dict[str, Any] = {}
    citing_paper: Dict[str, Any] = {}
    contexts: List[CitationContextItem] = []
    summary: str = ""
    citation_intents: List[str] = []


# ===================== KB Report 相关 =====================

class KBReportRequest(BaseModel):
    file_paths: List[str] = []
    api_url: str
    api_key: str
    model: str = "gpt-5.1"
    language: str = "zh"
    report_style: Literal["insight", "analysis"] = "insight"
    length: Literal["short", "standard", "long"] = "standard"
    email: Optional[str] = None
    user_id: Optional[str] = None
    notebook_id: Optional[str] = None


class KBReportResponse(BaseModel):
    success: bool
    report_markdown: str = ""
    report_path: str = ""
    output_file_id: str = ""


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
    # 图像生成分辨率（1K/2K/4K 等）
    image_resolution: Optional[str] = None


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
    # PPT/PDF 转图片时的渲染 DPI（None 表示使用默认值）
    render_dpi: Optional[int] = None

    # ---------------------- 输出图像比例设置 ----------------------
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"] = "16:9"
    style: str = " "
    use_long_paper: bool = False # 不使用 长文
    # 图像生成分辨率（1K/2K/4K 等）
    image_resolution: str = "2K"

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
