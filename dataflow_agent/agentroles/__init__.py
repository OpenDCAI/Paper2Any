# dataflow_agent/agentroles/__init__.py
import importlib
from pathlib import Path

from .registry import AgentRegistry

# 1) 自动 import 所有 .py 文件
_pkg_path = Path(__file__).resolve().parent
for py in _pkg_path.glob("*.py"):
    if py.stem not in {"__init__", "registry", "base_agent"}:
        importlib.import_module(f"{__name__}.{py.stem}")

# 2) 对外接口
def get_agent_cls(name: str):
    return AgentRegistry.get(name)

def create_agent(name: str, tool_manager=None, **kwargs):
    cls = get_agent_cls(name)
    return cls.create(tool_manager=tool_manager, **kwargs)

list_agents = AgentRegistry.all