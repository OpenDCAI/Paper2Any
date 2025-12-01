"""
TestGraph State and Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-01 20:17:07

本文件由 `dfa create --state_name test_graph` 自动生成。
用于创建继承MainState和MainRequest的自定义state和request。
"""

from dataclasses import dataclass, field
from dataflow_agent.state import MainState, MainRequest


# ==================== TestGraph Request ====================
@dataclass
class TestGraphRequest(MainRequest):
    """TestGraph任务的Request，继承自MainRequest"""
    pass


# ==================== TestGraph State ====================
@dataclass
class TestGraphState(MainState):
    """TestGraph任务的State，继承自MainState"""
    
    # 重写request类型为TestGraphRequest
    request: TestGraphRequest = field(default_factory=TestGraphRequest)