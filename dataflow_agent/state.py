from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataflow.cli_funcs.paths import DataFlowPath
current_file = Path(__file__).resolve()

BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent
STATICS_DIR = DataFlowPath.get_dataflow_statics_dir()
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
    chat_api_url: str = "http://123.129.219.111:3000/v1"
    api_key: str = os.getenv("DF_API_KEY", "test")

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



# ==================== 数据采集 Request ====================
@dataclass
class DataCollectionRequest(MainRequest):
    """数据采集任务的Request，继承自MainRequest"""
    # 重写language默认值
    language: str = "English"
    
    # 数据采集特有的字段
    download_dir: str = os.path.join(STATICS_DIR, "data_collection")
    dataset_size_category: str = '1K<n<10K'
    dataset_num_limit: int = 5
    category: str = "PT"
    max_dataset_size: int = None  # 数据集大小限制（字节数），None表示不限制
    max_download_subtasks: Optional[int] = None  # 下载子任务执行数量上限，None 表示不限制
    rag_api_url: Optional[str] = None
    rag_api_key: Optional[str] = None
    rag_embed_model: Optional[str] = None
    tavily_api_key: Optional[str] = None


# ==================== 数据采集 State ====================
@dataclass
class DataCollectionState(MainState):
    """数据采集任务的State，继承自MainState"""
    # 重写request类型为DataCollectionRequest
    request: DataCollectionRequest = field(default_factory=DataCollectionRequest)
    
    # 数据采集特有的字段
    keywords: list[str] = field(default_factory=list)
    datasets: Dict[str, list] = field(default_factory=dict)
    downloads: Dict[str, list] = field(default_factory=dict)
    sources: Dict[str, Dict] = field(default_factory=dict)

# Iconagent相关 State 和 Request 定义
# ==================== Icon 生成 Request ====================
@dataclass
class IconGenRequest(MainRequest):      
    keywords: str = ""
    style: str = ""
    prev_image: str = ""
    edit_prompt: str = ""

# ==================== Icon 生成 State ======================
@dataclass
class IconGenState(MainState):
    request: IconGenRequest = field(default_factory=IconGenRequest)

    # 下面是 icongen 自己的产物 / 临时数据
    icon_prompt: str = ""                                 # 生成的图标提示词
    img_save_path: str = ""                              # 生成的图标保存路径


# ==================== Web 爬取/研究 Request ====================
@dataclass
class WebCrawlRequest(MainRequest):
    """Web 爬取任务的 Request，继承自 MainRequest"""
    # 初始需求与下载目录
    initial_request: str = ""
    download_dir: str = os.path.join(STATICS_DIR, "web_crawl")

    # 爬取/研究配置
    search_engine: str = "tavily"     # 'tavily' | 'duckduckgo' | 'jina'
    use_jina_reader: bool = False
    enable_rag: bool = True
    max_download_subtasks: Optional[int] = None


# ==================== Web 爬取/研究 State ====================
@dataclass
class WebCrawlState(MainState):
    """管理网络爬取与研究过程的状态"""
    # 重写 request 类型为 WebCrawlRequest
    request: WebCrawlRequest = field(default_factory=WebCrawlRequest)

    # 直通字段（为兼容调用方直接从 state 访问这些配置项）
    initial_request: str = ""
    download_dir: str = os.path.join(STATICS_DIR, "web_crawl")
    search_engine: str = "tavily"
    use_jina_reader: bool = False
    enable_rag: bool = True
    rag_manager: Any = None
    max_download_subtasks: Optional[int] = None

    # 研究/爬取过程中的临时与产出数据
    sub_tasks: list[Dict[str, Any]] = field(default_factory=list)
    completed_sub_tasks: list[Dict[str, Any]] = field(default_factory=list)
    research_summary: Dict[str, Any] = field(default_factory=dict)
    search_results_text: str = ""
    filtered_urls: list[str] = field(default_factory=list)
    crawled_data: list[Dict[str, Any]] = field(default_factory=list)
    visited_urls: set[str] = field(default_factory=set)
    url_queue: list[str] = field(default_factory=list)
    is_finished: bool = False
    supervisor_feedback: str = "Process has not started."
    # 控制参数
    max_crawl_cycles_per_task: int = 5
    max_crawl_cycles_for_research: int = 15
    max_dataset_size: Optional[int] = None
    current_cycle: int = 0
    download_successful_for_current_task: bool = False
    completed_download_tasks: int = 0

    def reset_for_new_task(self):
        self.search_results_text = ""
        self.filtered_urls = []
        self.visited_urls = set()
        self.url_queue = []
        self.current_cycle = 0
        self.download_successful_for_current_task = False
        

    
@dataclass
class PromptWritingState(MainState):
    """提示词生成任务的State，继承自MainState"""
    request: DFRequest = field(default_factory=DFRequest)
    
    # 提示词生成特有的字段
    prompt_op_name: str = ""
    prompt_args: Dict[str, Any] = field(default_factory=dict)
    prompt_output_format: Dict[str, Any] = field(default_factory=dict)
    delete_test_files: bool = True


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
    gen_fig_model: str = "gemini-2.5-flash-image-preview",
    # gen_fig_model: str = "gemini-3-pro-image-preview",
    sam2_model: str = "models/facebook/sam2.1-hiera-tiny"
    bg_rm_model: str = "models/RMBG-2.0"

@dataclass
class Paper2FigureState(MainState):
    request: Paper2FigureRequest = field(default_factory=Paper2FigureRequest)
    fig_desc: str = ''
    aspect_ratio: str = '16:9'
    paper_file: str = ''
    fig_draft_path: str = ''
    fig_mask: List[Dict[str, Any]] = field(default_factory=dict)
    result_path: str = ''
    ppt_path: str = ''
    mask_detail_level: int = 2
    paper_idea: str = ''
    input_type: str = 'PDF'


    # 技术路线图使用属性
    figure_tec_svg_content: str = ""
    svg_img_path: str = ""
    svg_file_path: str = "" # svg 带文字图的 地址
    svg_bg_file_path: str = ""
    # 带文字版本的svg图片
    svg_full_img_path: str = "" 
    # 背景svg code：
    svg_bg_code : str = "" 