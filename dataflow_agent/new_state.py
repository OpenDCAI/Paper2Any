# /mnt/DataFlow/lz/proj/DataFlow-Agent/dataflow_agent/new_state.py

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Type, Optional, TypeVar, Generic
import os
from pathlib import Path

# ===== 基础路径配置 =====
class DataFlowPath:
    @staticmethod
    def get_dataflow_dir():
        # 按你实际项目结构修改
        return Path(__file__).parent.parent

    @staticmethod
    def get_dataflow_statics_dir():
        return Path(__file__).parent.parent / "statics"

BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent
STATICS_DIR = DataFlowPath.get_dataflow_statics_dir()
current_file = Path(__file__).resolve()
PROJDIR = current_file.parent.parent

# ===== 基础Request和State类 =====
T = TypeVar('T', bound='RequestBase')

@dataclass
class RequestBase:
    language: str = "en"
    chat_api_url: str = "http://123.129.219.111:3000/v1"
    api_key: str = os.getenv("DF_API_KEY", "test")

@dataclass
class StateBase(Generic[T]):
    request: T
    temp_data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def clear_temp(self):
        self.temp_data.clear()

    def asdict(self):
        return asdict(self)

# ===== DFRequest / DFState (主Agent流程) =====
@dataclass
class DFRequest(RequestBase):
    model: str = "gpt-4o"
    json_file: str = f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl"
    target: str = ""
    python_file_path: str = ""
    need_debug: bool = False
    max_debug_rounds: int = 3
    use_local_model: bool = False
    local_model_path: str = ""
    cache_dir: str = f"{PROJDIR}/cache_dir"

@dataclass
class DFState(StateBase[DFRequest]):
    messages: List[Any] = field(default_factory=list)
    agent_results: Dict[str, Any] = field(default_factory=dict)
    category: Dict[str, Any] = field(default_factory=dict)
    recommendation: Dict[str, Any] = field(default_factory=dict)
    matched_ops: List[str] = field(default_factory=list)
    debug_mode: bool = False
    pipeline_structure_code: Dict[str, Any] = field(default_factory=dict)
    execution_result: Dict[str, Any] = field(default_factory=dict)
    code_debug_result: Dict[str, Any] = field(default_factory=dict)

# ===== DataCollectionRequest / DataCollectionState (数据集收集流程) =====
@dataclass
class DataCollectionRequest(RequestBase):
    model: str = "gpt-4o"
    target: str = ""
    download_dir: str = os.path.join(STATICS_DIR, "data_collection")
    dataset_size_category: str = '1K<n<10K'
    dataset_num_limit: int = 5
    category: str = "PT"

@dataclass
class DataCollectionState(StateBase[DataCollectionRequest]):
    keywords: List[str] = field(default_factory=list)
    datasets: Dict[str, List[Any]] = field(default_factory=dict)
    downloads: Dict[str, List[Any]] = field(default_factory=dict)
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)

# ===== StateManager 工厂类（多Agent/多流程统一管理State） =====
class StateManager:
    def __init__(self):
        self._states: Dict[str, StateBase] = {}

    def create_state(self, name: str, state_cls: Type[StateBase], request: RequestBase) -> StateBase:
        state = state_cls(request)
        self._states[name] = state
        return state

    def get_state(self, name: str) -> Optional[StateBase]:
        return self._states.get(name)

    def clear_state(self, name: str):
        if name in self._states:
            del self._states[name]

    def clear_all(self):
        self._states.clear()

# ===== 用法示例 =====
if __name__ == "__main__":
    # 创建请求与状态
    df_req = DFRequest(language="zh")
    df_state = DFState(request=df_req)
    print(df_state.asdict())

    # 使用StateManager统一管理
    manager = StateManager()
    manager.create_state("df", DFState, df_req)
    state = manager.get_state("df")
    print(state.request.language)
    state.temp_data['foo'] = 'bar'
    print(state.temp_data)