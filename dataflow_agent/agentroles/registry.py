# dataflow_agent/agentroles/registry.py
from typing import Dict, Type
from importlib import import_module

from dataflow_agent.agentroles.base_agent import BaseAgent

class AgentRegistry:
    _agents: Dict[str, Type["BaseAgent"]] = {}

    @classmethod
    def register(cls, name: str, agent_cls: Type["BaseAgent"]):
        if name in cls._agents:
            raise ValueError(f"Agent '{name}' already registered")
        cls._agents[name] = agent_cls

    @classmethod
    def get(cls, name: str) -> Type["BaseAgent"]:
        try:
            return cls._agents[name]
        except KeyError:
            raise KeyError(f"Agent '{name}' 未注册，可选: {list(cls._agents)}")

    @classmethod
    def all(cls) -> Dict[str, Type["BaseAgent"]]:
        return dict(cls._agents)


def register(name: str):
    """装饰器：@register("writer")"""
    def _decorator(agent_cls: Type["BaseAgent"]):
        from dataflow_agent.agentroles.base_agent import BaseAgent   # 规避循环引用
        if not issubclass(agent_cls, BaseAgent):
            raise TypeError("只允许注册 BaseAgent 的子类")
        AgentRegistry.register(name, agent_cls)
        return agent_cls
    return _decorator