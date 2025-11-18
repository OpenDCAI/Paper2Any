# dataflow_agent/agentroles/__init__.py
import importlib
from pathlib import Path
from typing import Optional
from dataflow_agent.toolkits.tool_manager import get_tool_manager, ToolManager

# 1) 自动 import 所有 .py 文件
_pkg_path = Path(__file__).resolve().parent
for py in _pkg_path.glob("*.py"):
    if py.stem not in {"__init__", "registry", "base_agent", "configs", "strategies"}:
        importlib.import_module(f"{__name__}.{py.stem}")

from .registry import AgentRegistry
from .cores.configs import (
    BaseAgentConfig, 
    SimpleConfig, 
    ReactConfig, 
    GraphConfig, 
    VLMConfig,
    ExecutionMode
)

# ==================== 核心函数（增强版） ====================

def get_agent_cls(name: str):
    """获取 Agent 类"""
    return AgentRegistry.get(name)


def create_agent(
    name: str, 
    config: Optional[BaseAgentConfig] = None,
    tool_manager: Optional[ToolManager] = None,
    **legacy_kwargs
):
    """
    统一 Agent 创建入口（增强版，向后兼容）
    
    Args:
        name: 代理角色名称
        config: 执行配置对象（SimpleConfig/ReactConfig/GraphConfig/VLMConfig）
        tool_manager: 工具管理器实例
        **legacy_kwargs: 兼容旧参数（如果不传 config，则使用这些参数）
            - model_name: 模型名称
            - temperature: 采样温度 (0.0-1.0)
            - max_tokens: 最大生成token数
            - tool_mode: 工具调用模式 ("auto", "none", "required")
            - react_mode: 是否启用ReAct推理模式
            - react_max_retries: ReAct模式最大重试次数
            - parser_type: 解析器类型 ("json", "xml", "text")
            - parser_config: 解析器配置字典
            - use_vlm: 是否使用视觉语言模型
            - vlm_config: VLM配置字典
            - ignore_history: 是否忽略历史消息
            - message_history: 消息历史管理器
    
    Returns:
        Agent: 代理角色实例
    
    Examples:
        # 新方式（推荐）：使用配置对象
        agent = create_agent(
            "writer",
            config=ReactConfig(max_retries=5, temperature=0.7)
        )
        
        # 旧方式（兼容）：直接传参数
        agent = create_agent(
            "writer",
            react_mode=True,
            react_max_retries=5,
            temperature=0.7
        )
    """
    cls = get_agent_cls(name)
    
    # 如果没有提供 tool_manager，使用默认的
    if tool_manager is None and config is None:
        tool_manager = get_tool_manager()
    
    # 兼容旧参数：自动转换为配置对象
    if config is None and legacy_kwargs:
        config = _convert_legacy_params(legacy_kwargs)
        if tool_manager is None:
            tool_manager = get_tool_manager()
    
    # 如果仍然没有配置，使用简单模式
    if config is None:
        config = SimpleConfig()
        if tool_manager is None:
            tool_manager = get_tool_manager()
    
    # 合并 tool_manager
    if tool_manager:
        config.tool_manager = tool_manager
    
    # 调用原有的 cls.create，但传入 execution_config
    return cls.create(
        tool_manager=config.tool_manager,
        execution_config=config
    )


def _convert_legacy_params(kwargs: dict) -> BaseAgentConfig:
    """将旧参数转换为配置对象（内部函数）"""
    # 提取通用参数
    common_params = {
        "model_name": kwargs.get("model_name"),
        "temperature": kwargs.get("temperature", 0.0),
        "max_tokens": kwargs.get("max_tokens", 16384),
        "tool_mode": kwargs.get("tool_mode", "auto"),
        "parser_type": kwargs.get("parser_type", "json"),
        "parser_config": kwargs.get("parser_config"),
        "ignore_history": kwargs.get("ignore_history", True),
        "message_history": kwargs.get("message_history"),
    }
    
    # 根据关键参数判断模式
    if kwargs.get("use_vlm"):
        vlm_cfg = kwargs.get("vlm_config", {})
        return VLMConfig(
            **common_params,
            vlm_mode=vlm_cfg.get("mode", "understanding"),
            image_detail=vlm_cfg.get("image_detail", "auto"),
            max_image_size=vlm_cfg.get("max_image_size", (2048, 2048)),
            additional_params={k: v for k, v in vlm_cfg.items() 
                             if k not in {"mode", "image_detail", "max_image_size"}}
        )
    elif kwargs.get("react_mode"):
        return ReactConfig(
            **common_params,
            max_retries=kwargs.get("react_max_retries", 3),
            validators=kwargs.get("validators")
        )
    else:
        return SimpleConfig(**common_params)


# ==================== 便捷创建函数 ====================

def create_simple_agent(name: str, tool_manager: Optional[ToolManager] = None, **kwargs):
    """创建简单模式 Agent"""
    config = SimpleConfig(**kwargs)
    return create_agent(name, config=config, tool_manager=tool_manager)


def create_react_agent(
    name: str, 
    max_retries: int = 3,
    tool_manager: Optional[ToolManager] = None,
    **kwargs
):
    """创建 ReAct 模式 Agent"""
    config = ReactConfig(max_retries=max_retries, **kwargs)
    return create_agent(name, config=config, tool_manager=tool_manager)


def create_graph_agent(name: str, tool_manager: Optional[ToolManager] = None, **kwargs):
    """创建图模式 Agent"""
    config = GraphConfig(**kwargs)
    return create_agent(name, config=config, tool_manager=tool_manager)


def create_vlm_agent(
    name: str,
    vlm_mode: str = "understanding",
    image_detail: str = "auto",
    tool_manager: Optional[ToolManager] = None,
    **kwargs
):
    """创建 VLM 模式 Agent"""
    config = VLMConfig(
        vlm_mode=vlm_mode,
        image_detail=image_detail,
        **kwargs
    )
    return create_agent(name, config=config, tool_manager=tool_manager)


# ==================== 导出 ====================

list_agents = AgentRegistry.all

__all__ = [
    # 核心函数
    "get_agent_cls",
    "create_agent",
    "list_agents",
    
    # 便捷函数
    "create_simple_agent",
    "create_react_agent",
    "create_graph_agent",
    "create_vlm_agent",
    
    # 配置类
    "BaseAgentConfig",
    "SimpleConfig",
    "ReactConfig",
    "GraphConfig",
    "VLMConfig",
    "ExecutionMode",
    
    # 注册表
    "AgentRegistry",
]