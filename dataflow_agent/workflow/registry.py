# dataflow_agent/workflow/registry.py
from typing import Callable, Dict

class RuntimeRegistry:
    _workflows: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, factory: Callable):
        # 同一个对象重复登记 → 忽略
        if name in cls._workflows:
            if cls._workflows[name] is factory: 
                return
            raise ValueError(
                f"Workflow '{name}' already registered by "
                f"{cls._workflows[name]} (now trying {factory})"
            )
        cls._workflows[name] = factory

    @classmethod
    def get(cls, name: str) -> Callable:
        try:
            return cls._workflows[name]
        except KeyError:
            raise KeyError(
                f"Workflow '{name}' 不存在，可选值: {list(cls._workflows)}"
            )

    @classmethod
    def all(cls) -> Dict[str, Callable]:
        return dict(cls._workflows)

def register(name: str):
    def _decorator(func_or_cls):
        RuntimeRegistry.register(name, func_or_cls)
        return func_or_cls
    return _decorator