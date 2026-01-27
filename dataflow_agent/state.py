from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
current_file = Path(__file__).resolve()
PROJDIR = current_file.parent.parent
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ==================== 最基础的 Request ====================
@dataclass
class MainRequest:
    """所有Request的基类，只包含核心字段"""
    # ① 用户偏好的自然语言
    language: str = "en"  # "en" | "zh" | ...

    # ② LLM 接口
    chat_api_url: str = os.getenv("DF_API_URL", "test")
    api_key: str = os.getenv("DF_API_KEY", "test")
    chat_api_key: str = os.getenv("DF_API_KEY", "test") #没区别，但是不想改之前代码了；

    # ③ 选用的 LLM 名称
    model: str = "gpt-4o"

    # ④ 需求描述
    target: str = ""

    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


# ==================== 最基础的 State（所有State的祖先）====================
@dataclass
class MainState:
    """所有State的基类，只包含核心字段"""
    request: MainRequest = field(default_factory=MainRequest)
    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)
    # 通用字段
    agent_results: Dict[str, Any] = field(default_factory=dict)
    temp_data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __setitem__(self, key, value):
        setattr(self, key, value)


# ==================== 主流程 Request ====================
@dataclass
class DFRequest(MainRequest):
    """主流程的Request，继承自MainRequest"""
    # ⑤ 测试样例文件（仅 CLI 批量跑用）
    json_file: str = ""

    # ⑥ Python 代码文件位置
    python_file_path: str = ""

    # ⑦ Debug 相关
    need_debug: bool = False
    max_debug_rounds: int = 3

    # ⑧ 本地模型相关
    use_local_model: bool = False
    local_model_path: str = ""

    # ⑨ 缓存和会话
    cache_dir: str = f"{PROJDIR}/cache_dir"
    session_id: str = "default_session"

    # embeddings url
    chat_api_url_for_embeddings : str = ""
    embedding_model_name: str = "text-embedding-3-small"
    update_rag_content: bool = True

# ==================== 主流程 State ====================
@dataclass
class DFState(MainState):
    """主流程的State，继承自MainState"""
    # 重写request类型为DFRequest
    request: DFRequest = field(default_factory=DFRequest)

    
    # 主流程特有字段
    category: Dict[str, Any] = field(default_factory=dict)
    recommendation: Dict[str, Any] = field(default_factory=dict)
    matched_ops: list[str] = field(default_factory=list)
    debug_mode: bool = False
    pipeline_structure_code: Dict[str, Any] = field(default_factory=dict)
    execution_result: Dict[str, Any] = field(default_factory=dict)
    code_debug_result: Dict[str, Any] = field(default_factory=dict)
    debug_history: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    opname_and_params: List[Dict[str, Dict[str, Any]]] = field(default_factory=list)

# Paper2Video 相关 State 和 Request 定义
# ==================== Paper2Video 生成 Request ====================

@dataclass
class Paper2VideoRequest(MainRequest):
    paper_pdf_path: str = ""
    user_imgs_path: str = ""
    
    ref_audio_path: str = ""

# ==================== Paper2Video 生成 State ======================
@dataclass
class Paper2VideoState(MainState):
    # 重写 request
    request: Paper2VideoRequest = field(default_factory=Paper2VideoRequest)
    
    # paper2video 特有字段
    beamer_code_path: str = ""
    is_beamer_wrong: bool = False
    is_beamer_warning: bool = False
    code_debug_result: str = ""
    ppt_path: str = ""
    
    # 生成字幕 + cursor的位置信息
    slide_img_dir: str = ""
    subtitle_and_cursor: List[str] = field(default_factory=list)
    subtitle_and_cursor_path: str = ""
    
    # 生成的音频路径
    speech_save_dir: str = ""



# ==================== Planning Agent 相关 State ====================
@dataclass
class PlanningRequest(MainRequest):
    """Planning Agent 的 Request"""
    # 规划器配置
    planner_model: Optional[str] = None
    planner_temperature: float = 0.0
    
    # 执行器配置
    executor_model: Optional[str] = None
    executor_as_react: bool = True
    
    # 重规划器配置 (仅 Plan-and-Execute 模式)
    replanner_model: Optional[str] = None
    max_replanning_rounds: int = 3
    
    # Human-in-the-Loop 配置
    require_plan_approval: bool = True      # 是否需要计划审批
    interrupt_before_step: bool = True      # 每步执行前是否中断
    interrupt_after_step: bool = False      # 每步执行后是否中断
    
    # 执行配置
    max_plan_steps: int = 10
    planning_mode: str = "plan_solve"       # "plan_solve" | "plan_execute"


@dataclass
class PlanStep:
    """单个计划步骤"""
    index: int                              # 步骤索引
    description: str                        # 步骤描述
    status: str = "pending"                 # pending | running | completed | failed | skipped
    result: Optional[str] = None            # 执行结果
    error: Optional[str] = None             # 错误信息
    started_at: Optional[str] = None        # 开始时间
    completed_at: Optional[str] = None      # 完成时间


@dataclass
class PlanningState(MainState):
    """
    Planning Agent 的状态类
    
    支持两种模式:
    - Plan-and-Solve: 一次性生成计划，按顺序执行
    - Plan-and-Execute (Replanning): 动态调整计划
    """
    request: PlanningRequest = field(default_factory=PlanningRequest)
    
    # ===== 计划相关 =====
    plan: List[str] = field(default_factory=list)                    # 计划步骤列表 (简单字符串)
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)   # 详细计划步骤
    current_step_index: int = 0                                       # 当前执行步骤索引
    past_steps: List[tuple] = field(default_factory=list)            # [(步骤描述, 执行结果), ...]
    
    # ===== 状态控制 =====
    plan_approved: bool = False                     # 计划是否已审批
    is_replanning_needed: bool = False              # 是否需要重新规划
    replanning_count: int = 0                       # 重规划次数
    final_response: str = ""                        # 最终响应
    is_finished: bool = False                       # 是否已完成
    
    # ===== Human-in-the-Loop 相关 =====
    awaiting_human_input: bool = False              # 是否等待人类输入
    human_feedback: Optional[str] = None            # 人类反馈
    interrupt_reason: Optional[str] = None          # 中断原因
    
    # ===== 执行上下文 =====
    original_task: str = ""                         # 原始任务描述
    executor_tools: List[str] = field(default_factory=list)  # 可用工具列表
    
    def get_current_step(self) -> Optional[str]:
        """获取当前待执行的步骤"""
        if 0 <= self.current_step_index < len(self.plan):
            return self.plan[self.current_step_index]
        return None
    
    def get_remaining_steps(self) -> List[str]:
        """获取剩余未执行的步骤"""
        return self.plan[self.current_step_index:]
    
    def get_completed_steps(self) -> List[tuple]:
        """获取已完成的步骤及结果"""
        return self.past_steps
    
    def mark_step_complete(self, result: str):
        """标记当前步骤完成"""
        if self.current_step_index < len(self.plan):
            step = self.plan[self.current_step_index]
            self.past_steps.append((step, result))
            self.current_step_index += 1
    
    def reset_plan(self):
        """重置计划状态（用于重规划）"""
        self.plan = []
        self.plan_steps = []
        self.current_step_index = 0
        self.is_replanning_needed = False
        # 保留 past_steps，因为重规划需要参考历史执行结果
    
    def to_planning_context(self) -> Dict[str, Any]:
        """生成规划上下文（供 LLM 使用）"""
        return {
            "original_task": self.original_task or self.request.target,
            "past_steps": [
                {"step": step, "result": result} 
                for step, result in self.past_steps
            ],
            "remaining_steps": self.get_remaining_steps(),
            "replanning_count": self.replanning_count,
            "available_tools": self.executor_tools,
        }

@dataclass
class Paper2FigureRequest(MainRequest):
    gen_fig_model: str = "gemini-2.5-flash-image-preview"
    # gen_fig_model: str = "gemini-3-pro-image-preview"
    sam2_model: str = "models/facebook/sam2.1-hiera-tiny"
    bg_rm_model: str = "models/RMBG-2.0"

    # 新增：用于 wf_pdf2ppt_qwenvl.py 的 VLM 模型
    vlm_model: str = "qwen-vl-ocr-2025-11-20"
    
    # 新增：用于 wf_paper2expfigure.py 的图表相关模型
    chart_model: str = "gpt-4o"
    
    # 新增：用于 wf_paper2figure_image_only.py 的描述生成模型
    fig_desc_model: str = "gpt-5.1"
    
    # 新增：用于 wf_paper2technical.py 的技术路线生成模型
    technical_model: str = "claude-haiku-4-5-20251001"
    # 技术路线图模板/配色（可选）
    tech_route_template: str = ""
    tech_route_palette: str = ""

    input_type: str = "PDF"
    #  科研绘图复杂度    
    figure_complex: str = "hard"
    style: str = "kartoon"

    # PPT的页面数量 
    page_count: int = 10
    # 是否编辑完毕，也就是是否需要重新生成完整的 PPT
    all_edited_down: bool = False

    # pdf2ppt是否使用AI编辑
    use_ai_edit: bool = False

    # paper2ppt的参考图路径：
    ref_img: str = ''

@dataclass
class Paper2FigureState(MainState):
    request: Paper2FigureRequest = field(default_factory=Paper2FigureRequest)
    fig_desc: str = ''
    aspect_ratio: str = '16:9'
    paper_file: str = ''
    # 原始带内容的图像路径
    fig_draft_path: str = ''
    # MinerU 解析得到的内容元素（文本 / 图片 / 表格等）
    fig_mask: List[Dict[str, Any]] = field(default_factory=list)
    # 二次编辑后的空框模板图（仅外层矩形和箭头）
    fig_layout_path: str = ''
    # SAM + SVG + EMF 形成的布局元素（仅背景框架层）
    layout_items: List[Dict[str, Any]] = field(default_factory=list)
    result_path: str = ''
    ppt_path: str = ''
    mask_detail_level: int = 2
    paper_idea: str = ''
    input_type: str = 'PDF'

    # 技术路线图使用属性 ==============================
    figure_tec_svg_content: str = ""
    figure_tec_svg_bw_content: str = ""
    figure_tec_svg_color_content: str = ""
    svg_img_path: str = ""
    mineru_port: int = 8010
    svg_file_path: str = ""  # svg 带文字图的 地址
    svg_bg_file_path: str = ""
    svg_bw_file_path: str = ""
    svg_bw_img_path: str = ""
    svg_color_file_path: str = ""
    svg_color_img_path: str = ""
    # 带文字版本的svg图片
    svg_full_img_path: str = ""
    # 背景svg code：
    svg_bg_code : str = ""
    
    # 实验统计图使用属性 ==============================
    # ===== 输入 =====
    pre_tool_results: Dict[str, Any] = field(default_factory=dict)  # 前置工具结果注入

    # ===== 中间结果 =====
    paper_idea: str = ''                                          # 论文核心思想
    extracted_tables: List[Dict[str, Any]] = field(default_factory=list)  # 从 MinerU 提取的表格列表
    # 每个表格格式: {"table_id": str, "headers": List[str], "rows": List[List[str]], "caption": str}

    chart_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)     # 图表配置字典
    # 每个配置格式: {table_id: {"table_id": str, "chart_type": str, "x_column": str, "y_columns": List[str], ...}}

    generated_codes: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # 生成的代码字典
    # 每个代码格式: {table_id: {"table_id": str, "code": str}}

    # ===== 输出 =====
    generated_charts: Dict[str, str] = field(default_factory=dict)             # 生成的图表路径字典
    stylize_results: Dict[str, list] = field(default_factory=dict)             # 风格化后的图表路径字典

    svg_bg_code: str = ""

    # paper2ppt 专用 ==============================
    # 首次生成整套页面图是否已完成；False 走批量生成，True 走按页二次编辑
    gen_down: bool = False
    # 0-based: 要二次编辑的页号
    edit_page_num: int = -1
    # 二次编辑提示词（用于 edit_page_num 对应页）
    edit_page_prompt: str = ""
    # 批量生成出来的页面图片路径（0-based 对齐 pagecontent）
    generated_pages: List[str] = field(default_factory=list)
    table_img_path: str = ""

    # pagecontent: 既可为结构化 slide 描述，也可为 [{"ppt_img_path": "..."}] 的图片列表
    pagecontent: list[dict] = field(default_factory=list)
    minueru_output: str = ""
    mineru_root: str = ""
    text_content: str = ""
    outline_feedback: str = ""
    # 生成的 PPT PDF 路径
    ppt_pdf_path: str = ""
    ppt_pptx_path: str = ""

    # 长文PPT专用：
    long_text: str = ""
    target_pages: int = 60
    pages_per_batch: int = 10
    pages_to_generate: int = 12
    max_rounds: int = 1
    current_chunk: str = ""
    current_text: str = ""

    # pdf2ppt 专用 ==============================
    pdf_file: str = ""
    slide_images: List[str] = field(default_factory=list)
    ocr_pages: List[str] = field(default_factory=list)
    sam_pages: List[str] = field(default_factory=list)
    mineru_pages: List[Dict[str, Any]] = field(default_factory=list)
    # pdf2ppt是否使用AI编辑
    use_ai_edit: bool = False
    use_global_font_clustering: bool = False # 是否使用单页聚类器


    # img2ppt 专用 ==============================
    bbox_result: List[str] = field(default_factory=list)
    vlm_pages: List[Dict[str, Any]] = field(default_factory=list)

# ==================== Intelligent QA 相关 State ====================

@dataclass
class IntelligentQARequest(MainRequest):
    """
    智能问答请求
    """
    files: List[str] = field(default_factory=list)  # 文件路径列表
    query: str = ""  # 用户问题
    history: List[Dict[str, str]] = field(default_factory=list)  # 历史记录 [{"role": "user", "content": "..."}]

@dataclass
class IntelligentQAState(MainState):
    """
    智能问答状态
    """
    request: IntelligentQARequest = field(default_factory=IntelligentQARequest)

    # 解析后的上下文内容
    context_content: str = ""

    # 新增: 存储每个文件的分析结果
    file_analyses: List[Dict[str, Any]] = field(default_factory=list)

    # 最终回答
    answer: str = ""

@dataclass
class KBPodcastRequest(MainRequest):
    """
    知识播客请求
    """
    files: List[str] = field(default_factory=list)  # 文件路径列表
    tts_model: str = "gemini-2.5-pro-preview-tts"
    voice_name: str = "Kore"
    language: str = "zh"

@dataclass
class KBPodcastState(MainState):
    """
    知识播客状态
    """
    request: KBPodcastRequest = field(default_factory=KBPodcastRequest)
    result_path: str = ""
    file_contents: List[Dict[str, Any]] = field(default_factory=list)
    podcast_script: str = ""
    audio_path: str = ""


# ==================== KBMindMap 相关 State ====================

@dataclass
class KBMindMapRequest(MainRequest):
    """
    知识库思维导图请求
    """
    files: List[str] = field(default_factory=list)  # 文件路径列表
    mindmap_style: str = "default"  # default | flowchart | tree
    max_depth: int = 3  # 思维导图最大深度

@dataclass
class KBMindMapState(MainState):
    """
    知识库思维导图状态
    """
    request: KBMindMapRequest = field(default_factory=KBMindMapRequest)
    result_path: str = ""
    file_contents: List[Dict[str, Any]] = field(default_factory=list)
    content_structure: str = ""  # LLM提取的内容结构
    mermaid_code: str = ""  # 生成的Mermaid代码
    mindmap_svg_path: str = ""  # SVG输出路径（可选）


# ==================== Paper2Drawio 相关 State ====================

@dataclass
class Paper2DrawioRequest(MainRequest):
    """Paper2Drawio 请求参数"""
    # 输入类型: "PDF" | "TEXT"
    input_type: str = "TEXT"

    # 图表类型: "flowchart" | "architecture" | "sequence" | "mindmap" | "er" | "auto"
    diagram_type: str = "auto"

    # 图表风格: "minimal" | "sketch" | "default"
    diagram_style: str = "default"

    # 是否启用 VLM 验证
    enable_vlm_validation: bool = False

    # 最大重试次数
    max_retries: int = 3

    # 当前 XML (用于编辑模式)
    current_xml: str = ""

    # 编辑指令 (用于编辑模式)
    edit_instruction: str = ""

    # 会话历史 (用于多轮对话)
    chat_history: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Paper2DrawioState(MainState):
    """Paper2Drawio 工作流状态"""
    request: Paper2DrawioRequest = field(default_factory=Paper2DrawioRequest)

    # 输入内容
    paper_file: str = ""           # PDF 文件路径
    text_content: str = ""         # 文本内容

    # 中间结果
    paper_summary: str = ""        # 论文摘要/核心内容
    diagram_plan: str = ""         # 图表规划描述

    # 图表 XML
    drawio_xml: str = ""           # 当前 draw.io XML
    drawio_xml_history: List[str] = field(default_factory=list)  # XML 历史

    # 输出路径
    result_path: str = ""          # 结果目录
    output_xml_path: str = ""      # XML 文件路径
    output_png_path: str = ""      # PNG 导出路径
    output_svg_path: str = ""      # SVG 导出路径
