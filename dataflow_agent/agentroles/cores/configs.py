from typing import Any, Dict, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ExecutionMode(Enum):
    """执行模式枚举"""
    SIMPLE = "simple"           # 简单模式：单次LLM调用
    REACT = "react"             # ReAct模式：带验证的循环
    GRAPH = "graph"             # 图模式：子图+工具调用
    VLM = "vlm"                 # 视觉语言模型模式
    PARALLEL = "parallel"       # 并行模式：同时调用多个LLM
    CUSTOM = "custom"           # 自定义模式（预留）


@dataclass
class BaseAgentConfig:
    """Agent基础配置"""
    # 核心参数
    model_name: Optional[str] = None
    chat_api_url: Optional[str] = None  # 新增chat_api_url参数
    temperature: float = 0.0
    max_tokens: int = 16384
    
    # 工具相关
    tool_mode: str = "auto"
    tool_manager: Optional[Any] = None  # ToolManager
    
    # 解析器相关
    parser_type: str = "json"
    parser_config: Optional[Dict[str, Any]] = None
    
    # 消息历史
    ignore_history: bool = True
    message_history: Optional[Any] = None  # AdvancedMessageHistory


@dataclass
class SimpleConfig(BaseAgentConfig):
    """简单模式配置"""
    mode: ExecutionMode = field(default=ExecutionMode.SIMPLE, init=False)


@dataclass
class ReactConfig(BaseAgentConfig):
    """ReAct模式配置"""
    mode: ExecutionMode = field(default=ExecutionMode.REACT, init=False)
    max_retries: int = 3
    validators: Optional[List[Callable]] = None  # 自定义验证器


@dataclass
class GraphConfig(BaseAgentConfig):
    """图模式配置"""
    mode: ExecutionMode = field(default=ExecutionMode.GRAPH, init=False)
    enable_react_validation: bool = False  # 是否在图模式中启用ReAct验证
    react_max_retries: int = 3


@dataclass
class VLMConfig(BaseAgentConfig):
    """视觉语言模型配置"""
    mode: ExecutionMode = field(default=ExecutionMode.VLM, init=False)
    vlm_mode: str = "understanding"  # understanding/generation/edit
    image_detail: str = "auto"       # low/high/auto
    max_image_size: tuple = (1024, 1024)
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelConfig(BaseAgentConfig):
    """并行模式配置"""
    mode: ExecutionMode = field(default=ExecutionMode.PARALLEL, init=False)
    concurrency_limit: int = 5  # 并行度限制，默认同时执行5个任务