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