from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict
from dataflow.cli_funcs.paths import DataFlowPath

BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent
STATICS_DIR = DataFlowPath.get_dataflow_statics_dir()

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
    chat_api_url: str = "http://123.129.219.111:3000/v1"
    api_key: str = os.getenv("DF_API_KEY", "test")

    # ③ 选用的 LLM 名称
    model: str = "gpt-4o"

    # ④ 需求描述
    target: str = ""


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
    json_file: str = (
        f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl"
    )

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

# ==================== Icon 生成 State ======================
@dataclass
class IconGenState(MainState):
    request: IconGenRequest = field(default_factory=IconGenRequest)

    # 下面是 icongen 自己的产物 / 临时数据
    icon_prompts: list[str] = field(default_factory=list)
    svg_results: list[str] = field(default_factory=list)   # 或 base64, 路径等