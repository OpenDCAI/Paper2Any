"""
BaseAgent 模块 - Agent 系统的核心基类

本模块定义了 BaseAgent 抽象基类，它是所有 Agent 角色的基础。
BaseAgent 提供了统一的执行模式、工具管理、消息构建和结果解析等核心功能。

主要功能：
- 多种执行模式：简单模式、ReAct模式、并行模式、图模式
- 工具管理：前置工具和后置工具的执行
- 消息构建：系统提示词和任务提示词的生成
- 结果解析：支持 JSON、XML、文本等多种格式
- Agent-as-Tool：将 Agent 包装为可被其他 Agent 调用的工具
- VLM 支持：视觉语言模型的集成

使用示例：
    class MyAgent(BaseAgent):
        @property
        def role_name(self) -> str:
            return "MyAgent"
        
        @property
        def system_prompt_template_name(self) -> str:
            return "my_agent_system"
        
        @property
        def task_prompt_template_name(self) -> str:
            return "my_agent_task"

作者: Zhou Liu
版本: 1.0.0
"""

from __future__ import annotations

# =============================================================================
# 标准库导入
# =============================================================================
import asyncio
import datetime
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable, Tuple

# =============================================================================
# 第三方库导入
# =============================================================================
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

# =============================================================================
# 项目内部导入
# =============================================================================
from dataflow_agent.llm_callers.base import BaseLLMCaller
from dataflow_agent.parsers.parsers import BaseParser
from dataflow_agent.graphbuilder.message_history import AdvancedMessageHistory
from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import MainState
from dataflow_agent.utils import robust_parse_json, get_project_root
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.strategies import ExecutionStrategy

# =============================================================================
# 常量定义
# =============================================================================
PROJDIR = get_project_root()
"""项目根目录路径"""

log = get_logger(__name__)
"""模块日志记录器"""

# =============================================================================
# 类型定义
# =============================================================================
ValidatorFunc = Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str]]]
"""
验证器函数类型定义

验证器用于 ReAct 模式中验证 LLM 输出的正确性。

参数:
    content (str): LLM 原始输出内容
    parsed_result (Dict[str, Any]): 解析后的结果字典

返回:
    Tuple[bool, Optional[str]]: (是否通过验证, 错误信息)
        - 如果通过验证，返回 (True, None)
        - 如果未通过验证，返回 (False, "错误描述")
"""


class BaseAgent(ABC):
    """
    Agent 基类 - 定义通用的 Agent 执行模式
    
    BaseAgent 是所有 Agent 角色的抽象基类，提供了完整的 Agent 生命周期管理，
    包括初始化、消息构建、LLM 调用、结果解析和状态更新等核心功能。
    
    核心特性：
        1. 自动注册：子类定义时自动注册到 AgentRegistry
        2. 多执行模式：支持简单、ReAct、并行、图等多种执行模式
        3. 工具集成：支持前置工具和后置工具的管理和执行
        4. 灵活解析：支持 JSON、XML、文本等多种输出格式解析
        5. VLM 支持：集成视觉语言模型能力
        6. Agent-as-Tool：可将 Agent 包装为工具供其他 Agent 调用
    
    子类必须实现的抽象属性：
        - role_name: Agent 角色名称
        - system_prompt_template_name: 系统提示词模板名称
        - task_prompt_template_name: 任务提示词模板名称
    
    Attributes:
        tool_manager (ToolManager): 工具管理器实例
        model_name (str): LLM 模型名称
        temperature (float): LLM 温度参数
        max_tokens (int): 最大 token 数
        tool_mode (str): 工具调用模式
        react_mode (bool): 是否启用 ReAct 模式
        react_max_retries (int): ReAct 最大重试次数
        parser_type (str): 解析器类型
        use_vlm (bool): 是否使用视觉语言模型
        ignore_history (bool): 是否忽略消息历史
        message_history (AdvancedMessageHistory): 消息历史管理器
    
    Example:
        >>> class WriterAgent(BaseAgent):
        ...     @property
        ...     def role_name(self) -> str:
        ...         return "Writer"
        ...     
        ...     @property
        ...     def system_prompt_template_name(self) -> str:
        ...         return "writer_system"
        ...     
        ...     @property
        ...     def task_prompt_template_name(self) -> str:
        ...         return "writer_task"
        >>> 
        >>> agent = WriterAgent(tool_manager=tm)
        >>> result = await agent.execute(state)
    """

    # =========================================================================
    # A. 类初始化与工厂方法
    # =========================================================================

    def __init_subclass__(cls, **kwargs):
        """
        子类注册钩子
        
        当定义 BaseAgent 的子类时自动调用，将子类注册到 AgentRegistry 中。
        这使得可以通过角色名称动态创建 Agent 实例。
        
        注册过程：
            1. 创建子类的临时实例（使用 tool_manager=None）
            2. 获取实例的 role_name
            3. 将 (role_name.lower(), cls) 注册到 AgentRegistry
        
        Args:
            **kwargs: 传递给父类的关键字参数
        
        Note:
            如果子类初始化失败（例如缺少必要参数），注册会静默失败。
            这是为了允许抽象子类的定义。
        """
        super().__init_subclass__(**kwargs)
        try:
            # 创建临时实例以获取 role_name
            tmp = cls(tool_manager=None)
            name = tmp.role_name
            # 注册到 AgentRegistry
            from dataflow_agent.agentroles.registry import AgentRegistry
            AgentRegistry.register(name.lower(), cls)
        except Exception as e:
            # 静默失败，允许抽象子类
            pass
    
    def __init__(self, 
                 tool_manager: Optional[ToolManager] = None,
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 16384,
                 tool_mode: str = "auto",
                 react_mode: bool = False,
                 react_max_retries: int = 3,
                 parser_type: str = "json",
                 parser_config: Optional[Dict[str, Any]] = None,
                 use_vlm: bool = False,
                 vlm_config: Optional[Dict[str, Any]] = None,
                 ignore_history: bool = True,
                 message_history: Optional[AdvancedMessageHistory] = None,
                 chat_api_url: Optional[str] = None,
                 execution_config: Optional[Any] = None):
        """
        初始化 BaseAgent 实例
        
        Args:
            tool_manager (ToolManager, optional): 工具管理器，用于管理前置和后置工具
            model_name (str, optional): LLM 模型名称，如 "gpt-4"、"claude-3"
            temperature (float): LLM 温度参数，控制输出随机性，默认 0.0
            max_tokens (int): 最大输出 token 数，默认 16384
            tool_mode (str): 工具调用模式，可选 "auto"、"required"、"none"
            react_mode (bool): 是否启用 ReAct 模式（带验证的循环调用）
            react_max_retries (int): ReAct 模式最大重试次数，默认 3
            parser_type (str): 解析器类型，可选 "json"、"xml"、"text"
            parser_config (dict, optional): 解析器配置，如 XML 的 root_tag
            use_vlm (bool): 是否使用视觉语言模型
            vlm_config (dict, optional): VLM 配置，包含 mode、image_path 等
            ignore_history (bool): 是否忽略消息历史，默认 True
            message_history (AdvancedMessageHistory, optional): 消息历史管理器
            chat_api_url (str, optional): 自定义 API 端点 URL
            execution_config (Any, optional): 执行策略配置，用于高级执行控制
        """
        # ----- 基础配置 -----
        self.tool_manager = tool_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_mode = tool_mode
        self.react_mode = react_mode
        self.react_max_retries = react_max_retries
        self.chat_api_url = chat_api_url
        
        # ----- 解析器配置 -----
        self.parser_type = parser_type
        self.parser_config = parser_config or {}
        self._parser = None  # 懒加载，首次访问时创建
        
        # ----- VLM 配置 -----
        self.use_vlm = use_vlm
        self.vlm_config = vlm_config or {}
        
        # ----- 消息历史配置 -----
        self.ignore_history = ignore_history
        self.message_history = message_history or AdvancedMessageHistory()

        # ----- 策略模式支持 -----
        self._execution_strategy: Optional[ExecutionStrategy] = None
        if execution_config:
            # 从执行配置中更新 agent 属性
            # 这解决了通过 create_simple_agent 等函数创建时参数不生效的问题
            for f in execution_config.__dataclass_fields__:
                config_value = getattr(execution_config, f)
                if hasattr(self, f) and config_value is not None:
                    setattr(self, f, config_value)

            # 创建执行策略
            from dataflow_agent.agentroles.cores.strategies import StrategyFactory
            self._execution_strategy = StrategyFactory.create(
                execution_config.mode.value,
                self,
                execution_config
            )

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs) -> "BaseAgent":
        """
        工厂方法 - 统一的 Agent 创建入口
        
        提供一个标准化的方式来创建 Agent 实例，确保所有 Agent
        都通过相同的接口创建，便于依赖注入和测试。
        
        Args:
            tool_manager (ToolManager, optional): 工具管理器实例
            **kwargs: 传递给 __init__ 的其他参数
        
        Returns:
            BaseAgent: 创建的 Agent 实例（实际类型为调用此方法的子类）
        
        Example:
            >>> agent = WriterAgent.create(tool_manager=tm, temperature=0.7)
        """
        return cls(tool_manager=tool_manager, **kwargs)

    # =========================================================================
    # B. 抽象属性 - 子类必须实现
    # =========================================================================
    
    @property
    @abstractmethod
    def role_name(self) -> str:
        """
        角色名称 - 子类必须实现
        
        返回 Agent 的唯一标识名称，用于：
        - 注册到 AgentRegistry
        - 存储执行结果到 state.agent_results
        - 日志记录和调试
        - 工具管理器中的角色匹配
        
        Returns:
            str: Agent 角色名称，如 "Classifier"、"Writer"、"Recommender"
        
        Example:
            >>> @property
            ... def role_name(self) -> str:
            ...     return "Writer"
        """
        pass
    
    @property
    @abstractmethod
    def system_prompt_template_name(self) -> str:
        """
        系统提示词模板名称 - 子类必须实现
        
        返回用于生成系统提示词的模板名称。
        模板文件应位于 promptstemplates/resources 目录下。
        
        Returns:
            str: 模板名称，如 "writer_system"、"classifier_system"
        
        Note:
            模板使用 Jinja2 语法，可以包含变量占位符。
        """
        pass
    
    @property
    @abstractmethod
    def task_prompt_template_name(self) -> str:
        """
        任务提示词模板名称 - 子类必须实现
        
        返回用于生成任务提示词的模板名称。
        任务提示词包含具体的任务指令和上下文信息。
        
        Returns:
            str: 模板名称，如 "writer_task"、"classifier_task"
        """
        pass

    # =========================================================================
    # C. 解析器相关
    # =========================================================================
    
    @property
    def parser(self) -> BaseParser:
        """
        获取解析器实例（懒加载）
        
        根据 parser_type 和 parser_config 创建对应的解析器。
        解析器用于将 LLM 的原始输出转换为结构化数据。
        
        支持的解析器类型：
        - json: JSON 格式解析，支持 schema 验证
        - xml: XML 格式解析，支持自定义 root_tag
        - text: 纯文本，不做解析
        
        Returns:
            BaseParser: 解析器实例
        
        Note:
            解析器在首次访问时创建，之后复用同一实例。
        """
        if self._parser is None:
            from dataflow_agent.parsers import ParserFactory
            
            # 合并 parser_config 和快捷配置
            config = self.parser_config.copy() if self.parser_config else {}
            
            # 如果是 JSON 解析器，合并 schema 相关配置
            if self.parser_type == "json":
                if hasattr(self, 'response_schema') and self.response_schema:
                    config['schema'] = self.response_schema
                if hasattr(self, 'response_schema_description') and self.response_schema_description:
                    config['schema_description'] = self.response_schema_description
                if hasattr(self, 'response_example') and self.response_example:
                    config['example'] = self.response_example
                if hasattr(self, 'required_fields') and self.required_fields:
                    config['required_fields'] = self.required_fields
            
            self._parser = ParserFactory.create(self.parser_type, **config)
        return self._parser
    
    def parse_result(self, content: str) -> Dict[str, Any]:
        """
        使用配置的解析器解析 LLM 输出结果
        
        将 LLM 的原始文本输出转换为结构化的字典格式。
        
        Args:
            content (str): LLM 原始输出内容
        
        Returns:
            Dict[str, Any]: 解析后的结果字典
                - 成功时返回解析后的数据
                - 失败时返回 {"raw": content, "error": error_message}
        
        Example:
            >>> result = agent.parse_result('{"status": "success", "data": [1, 2, 3]}')
            >>> print(result)
            {'status': 'success', 'data': [1, 2, 3]}
        """
        try:
            parsed = self.parser.parse(content)
            log.info(f"{self.role_name} 使用 {self.parser_type} 解析器解析成功")
            log.critical(f"[parse_result 解析结果] : {parsed}")
            return parsed
        except Exception as e:
            log.exception(f"解析失败: {e}")
            return {"raw": content, "error": str(e)}

    # =========================================================================
    # D. 消息构建
    # =========================================================================
    
    def build_messages(self, 
                       state: MainState, 
                       pre_tool_results: Dict[str, Any]) -> List[BaseMessage]:
        """
        构建 LLM 输入消息列表
        
        根据系统提示词模板和任务提示词模板生成完整的消息列表，
        包括格式说明（如果使用解析器）。
        
        Args:
            state (MainState): 当前状态对象，包含请求信息
            pre_tool_results (Dict[str, Any]): 前置工具执行结果
        
        Returns:
            List[BaseMessage]: 消息列表，包含 SystemMessage 和 HumanMessage
        
        消息结构：
            1. SystemMessage: 系统提示词 + 格式说明
            2. HumanMessage: 任务提示词（包含前置工具结果）
        """
        log.info("构建提示词消息...")
        
        # 创建提示词生成器
        ptg = PromptsTemplateGenerator(state.request.language)
        
        # 渲染系统提示词
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        # 添加解析器格式说明（VLM 模式可能不需要）
        format_instruction = self.parser.get_format_instruction()
        if format_instruction and not self.use_vlm:
            sys_prompt += f"\n\n{format_instruction}"
        
        # 渲染任务提示词
        task_params = self.get_task_prompt_params(pre_tool_results)
        task_prompt = ptg.render(self.task_prompt_template_name, **task_params)
        log.info(f"[build_messages]任务提示词: {task_prompt}")
        
        # 构建消息列表
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=task_prompt),
        ]
        
        log.info("提示词消息构建完成")
        return messages
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取任务提示词参数 - 子类可重写
        
        将前置工具结果转换为任务提示词模板所需的参数。
        子类可以重写此方法以自定义参数处理逻辑。
        
        Args:
            pre_tool_results (Dict[str, Any]): 前置工具执行结果
        
        Returns:
            Dict[str, Any]: 提示词模板参数字典
        
        Example:
            >>> def get_task_prompt_params(self, pre_tool_results):
            ...     return {
            ...         "user_input": pre_tool_results.get("input", ""),
            ...         "context": pre_tool_results.get("context", ""),
            ...     }
        """
        return pre_tool_results
    
    def build_generation_prompt(self, pre_tool_results: Dict[str, Any]) -> str:
        """
        构建生成提示词（用于 VLM 图像生成模式）
        
        将前置工具结果整合到 VLM 配置的 prompt 中。
        
        Args:
            pre_tool_results (Dict[str, Any]): 前置工具执行结果
        
        Returns:
            str: 生成提示词
        """
        return f"{self.vlm_config.get('prompt', '')}"

    # =========================================================================
    # E. LLM 创建与调用
    # =========================================================================
    
    def get_llm_caller(self, state: MainState) -> BaseLLMCaller:
        """
        根据配置返回对应的 LLM Caller
        
        根据 use_vlm 配置选择返回 VisionLLMCaller 或 TextLLMCaller。
        
        Args:
            state (MainState): 当前状态对象
        
        Returns:
            BaseLLMCaller: LLM 调用器实例
        
        Note:
            此方法目前未被广泛使用，主要使用 create_llm 方法。
        """
        if self.use_vlm:
            from dataflow_agent.llm_callers import VisionLLMCaller
            log.info(f"使用 VisionLLMCaller，模式: {self.vlm_config.get('mode', 'understanding')}")
            return VisionLLMCaller(
                state,
                vlm_config=self.vlm_config,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tool_mode=self.tool_mode,
                tool_manager=self.tool_manager,
                chat_api_url=self.chat_api_url
            )
        else:
            from dataflow_agent.llm_callers import TextLLMCaller
            return TextLLMCaller(
                state,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tool_mode=self.tool_mode,
                tool_manager=self.tool_manager,
                chat_api_url=self.chat_api_url
            )

    def create_llm(self, state: MainState, bind_post_tools: bool = False) -> ChatOpenAI:
        """
        创建 LLM 实例
        
        根据配置创建 ChatOpenAI 实例，可选择性地绑定后置工具。
        
        Args:
            state (MainState): 当前状态对象，包含 API 配置
            bind_post_tools (bool): 是否绑定后置工具，默认 False
        
        Returns:
            ChatOpenAI: 配置好的 LLM 实例
        
        Note:
            - 模型名称优先使用 self.model_name，否则使用 state.request.model
            - API URL 优先使用 self.chat_api_url，否则使用 state.request.chat_api_url
        """
        # 确定实际使用的模型和 URL
        actual_model = self.model_name or state.request.model
        actual_url = self.chat_api_url or state.request.chat_api_url
        
        log.info(f"[create_llm:]创建LLM实例，温度: {self.temperature}, "
                 f"最大token: {self.max_tokens}, 模型: {actual_model}, "
                 f"接口URL: {actual_url}, API Key: {state.request.api_key}")
        
        # 创建 LLM 实例
        llm = ChatOpenAI(
            openai_api_base=actual_url,
            openai_api_key=state.request.api_key,
            model_name=actual_model,
            temperature=self.temperature,
        )
        
        # 绑定后置工具（如果需要）
        if bind_post_tools and self.tool_manager:
            post_tools = self.get_post_tools()
            if post_tools:
                llm = llm.bind_tools(post_tools, tool_choice=self.tool_mode)
                log.info(f"[create_llm]:为LLM绑定了 {len(post_tools)} 个后置工具: "
                        f"{[t.name for t in post_tools]}")
        
        return llm
    
    async def process_with_llm_for_graph(self, messages: List[BaseMessage], state: MainState) -> BaseMessage:
        """
        图模式下的 LLM 调用
        
        在图执行模式下调用 LLM，会绑定后置工具以支持工具调用。
        
        Args:
            messages (List[BaseMessage]): 输入消息列表
            state (MainState): 当前状态对象
        
        Returns:
            BaseMessage: LLM 响应消息
        
        Raises:
            Exception: LLM 调用失败时抛出异常
        """
        llm = self.create_llm(state, bind_post_tools=True)
        try:
            response = await llm.ainvoke(messages)
            log.info(response)
            log.info(f"{self.role_name} 图模式LLM调用成功")
            return response
        except Exception as e:
            log.exception(f"{self.role_name} 图模式LLM调用失败: {e}")
            raise

    # =========================================================================
    # F. 工具管理
    # =========================================================================
    
    async def execute_pre_tools(self, state: MainState) -> Dict[str, Any]:
        """
        执行前置工具
        
        在 LLM 调用之前执行的工具，用于收集上下文信息。
        
        Args:
            state (MainState): 当前状态对象
        
        Returns:
            Dict[str, Any]: 前置工具执行结果
        
        Note:
            如果未提供 tool_manager，将返回默认值。
        """
        log.info(f"开始执行 {self.role_name} 的前置工具...")
        
        # 检查工具管理器
        if not self.tool_manager:
            log.info("未提供工具管理器，使用默认值")
            return self.get_default_pre_tool_results()
        
        # 执行前置工具
        results = await self.tool_manager.execute_pre_tools(self.role_name)
        
        # 设置默认值
        defaults = self.get_default_pre_tool_results()
        for key, default_value in defaults.items():
            if key not in results or results[key] is None:
                results[key] = default_value
                
        log.info(f"前置工具执行完成，获得: {list(results.keys())}")
        return results
    
    def get_post_tools(self) -> List[Tool]:
        """
        获取后置工具列表
        
        从工具管理器获取当前角色的后置工具，并去重。
        
        Returns:
            List[Tool]: 去重后的后置工具列表
        """
        if not self.tool_manager:
            return []
        
        tools = self.tool_manager.get_post_tools(self.role_name)
        
        # 去重
        uniq, seen = [], set()
        for t in tools:
            if t.name not in seen:
                uniq.append(t)
                seen.add(t.name)
        
        return uniq
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """
        获取默认前置工具结果 - 子类可重写
        
        当没有工具管理器或前置工具未返回某些字段时，使用此默认值。
        
        Returns:
            Dict[str, Any]: 默认的前置工具结果
        
        Example:
            >>> def get_default_pre_tool_results(self):
            ...     return {
            ...         "context": "",
            ...         "history": [],
            ...     }
        """
        return {}
    
    def has_tool_calls(self, message: BaseMessage) -> bool:
        """
        检查消息是否包含工具调用
        
        Args:
            message (BaseMessage): 要检查的消息
        
        Returns:
            bool: 如果消息包含工具调用返回 True，否则返回 False
        """
        return hasattr(message, 'tool_calls') and bool(getattr(message, 'tool_calls', None))

    # =========================================================================
    # G. 执行模式 - 简单模式
    # =========================================================================
    
    async def process_simple_mode(self, state: MainState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        简单模式处理 - 单次 LLM 调用
        
        最基础的执行模式，直接调用 LLM 并解析结果。
        
        Args:
            state (MainState): 当前状态对象
            pre_tool_results (Dict[str, Any]): 前置工具执行结果
        
        Returns:
            Dict[str, Any]: 解析后的 LLM 输出结果
        
        流程：
            1. 构建消息
            2. 合并历史消息（如果启用）
            3. 调用 LLM
            4. 更新消息历史
            5. 解析并返回结果
        """
        log.info(f"执行 {self.role_name} 简单模式...")
        
        # 构建消息
        messages = self.build_messages(state, pre_tool_results)
        
        # 消息历史管理
        if not self.ignore_history:
            history_messages = self.message_history.get_messages()
            if history_messages:
                messages = self.message_history.merge_histories(history_messages, messages)
                log.info(f"合并了 {len(history_messages)} 条历史消息")
        
        # 创建 LLM（不绑定工具）
        llm = self.create_llm(state, bind_post_tools=False)
        
        try:
            # 调用 LLM
            answer_msg = await llm.ainvoke(messages)
            answer_text = answer_msg.content
            log.info(f'LLM原始输出：{answer_text}')
            log.info("LLM调用成功，开始解析结果")
            
            # 更新消息历史
            if not self.ignore_history:
                self.message_history.add_messages([answer_msg])
                log.info("已更新消息历史")
                
        except Exception as e:
            log.exception("LLM调用失败: %s", e)
            return {"error": str(e)}
        
        return self.parse_result(answer_text)

    # =========================================================================
    # G. 执行模式 - ReAct 模式
    # =========================================================================
    
    async def process_react_mode(self, state: MainState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        ReAct 模式处理 - 带验证的循环调用
        
        循环调用 LLM 直到输出通过所有验证器，或达到最大重试次数。
        
        Args:
            state (MainState): 当前状态对象
            pre_tool_results (Dict[str, Any]): 前置工具执行结果
        
        Returns:
            Dict[str, Any]: 验证通过的结果，或包含错误信息的字典
        
        流程：
            1. 构建初始消息
            2. 循环调用 LLM
            3. 解析结果并运行验证器
            4. 如果验证通过，返回结果
            5. 如果验证失败，添加反馈消息并重试
            6. 达到最大重试次数后返回错误
        """
        log.info(f"执行 {self.role_name} ReAct模式 (最大重试: {self.react_max_retries})")
        
        # 构建初始消息
        messages = self.build_messages(state, pre_tool_results)
        
        # 消息历史管理
        if not self.ignore_history:
            history_messages = self.message_history.get_messages()
            if history_messages:
                messages = self.message_history.merge_histories(history_messages, messages)
                log.info(f"合并了 {len(history_messages)} 条历史消息")
        
        # 创建 LLM
        llm = self.create_llm(state, bind_post_tools=False)
        
        # 循环调用直到验证通过或达到最大重试次数
        for attempt in range(self.react_max_retries + 1):
            try:
                # 调用 LLM
                log.info(f"ReAct尝试 {attempt + 1}/{self.react_max_retries + 1}")
                answer_msg = await llm.ainvoke(messages)
                answer_text = answer_msg.content
                log.info(f'LLM原始输出：{answer_text[:200]}...' if len(answer_text) > 200 else f'LLM原始输出：{answer_text}')
                
                # 解析结果
                parsed_result = self.parse_result(answer_text)
                
                # 运行验证器
                all_passed, errors = self._run_validators(answer_text, parsed_result)
                
                if all_passed:
                    log.info(f"✓ {self.role_name} ReAct验证通过，共尝试 {attempt + 1} 次")
                    
                    # 更新消息历史
                    if not self.ignore_history:
                        self.message_history.add_messages([answer_msg])
                        log.info("已更新消息历史")
                    
                    return parsed_result
                
                # 验证未通过
                if attempt < self.react_max_retries:
                    # 构建反馈消息
                    feedback = self._build_validation_feedback(errors)
                    log.warning(f"[process_react_mode] : 验证未通过 (尝试 {attempt + 1}): {feedback}")
                    
                    # 添加 LLM 的回复和人类的反馈到消息列表
                    messages.append(AIMessage(content=answer_text))
                    messages.append(HumanMessage(content=feedback))
                else:
                    # 达到最大重试次数
                    log.error(f"[process_react_mode] : {self.role_name} ReAct达到最大重试次数，验证仍未通过")
                    return {
                        "error": "ReAct验证失败",
                        "attempts": attempt + 1,
                        "last_errors": errors,
                        "last_result": parsed_result
                    }
                    
            except Exception as e:
                log.exception(f"ReAct模式LLM调用失败 (尝试 {attempt + 1}): {e}")
                if attempt >= self.react_max_retries:
                    return {"error": f"LLM调用失败: {str(e)}"}
                # 继续重试
                continue
        
        # 理论上不会到这里
        return {"error": "ReAct处理异常终止"}

    # =========================================================================
    # G. 执行模式 - 并行模式
    # =========================================================================
    
    async def process_parallel_mode(self, state: MainState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        并行模式处理 - 并发执行多个 LLM 调用
        
        自动检测前置工具结果中的列表数据，对每个元素并行调用 LLM。
        
        Args:
            state (MainState): 当前状态对象
            pre_tool_results (Dict[str, Any]): 前置工具执行结果
        
        Returns:
            Dict[str, Any]: 包含所有并行结果的字典
                - parallel_results: 结果列表
                - total_processed: 处理总数
        
        数据检测优先级：
            1. pre_tool_results 本身是列表
            2. 包含 "parallel_items" 字段
            3. 任意值为列表的字段（取第一个非空列表）
        """
        log.info(f"执行 {self.role_name} 并行模式...")
        
        # ----- 智能检测并行数据 -----
        parallel_items = []
        
        # 情况1: pre_tool_results 本身是列表
        if isinstance(pre_tool_results, list):
            parallel_items = pre_tool_results
        
        # 情况2: 有明确的 parallel_items 字段
        elif "parallel_items" in pre_tool_results:
            parallel_items = pre_tool_results["parallel_items"]
        
        # 情况3: 检查任意值为列表的字段
        elif isinstance(pre_tool_results, dict):
            for key, value in pre_tool_results.items():
                if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    parallel_items = value
                    break
        
        log.critical(f"[process_parallel_mode 并行数据] : {pre_tool_results}")
        
        # 如果没有找到合适的并行数据，回退到简单模式
        if not parallel_items:
            log.warning("未找到合适的并行数据，回退到简单模式")
            return await self.process_simple_mode(state, pre_tool_results)
        
        log.info(f"找到 {len(parallel_items)} 条数据用于并行处理")
        
        # ----- 获取并发限制 -----
        concurrency_limit = 5  # 默认值
        if hasattr(self, '_execution_strategy') and hasattr(self._execution_strategy, 'config'):
            if hasattr(self._execution_strategy.config, 'concurrency_limit'):
                concurrency_limit = self._execution_strategy.config.concurrency_limit
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        # ----- 定义单个并行任务处理函数 -----
        async def process_item(item: dict) -> dict:
            """处理单个并行项"""
            async with semaphore:
                try:
                    # 为每个并行项创建独立的上下文
                    item_pre_tool_results = {}
                    
                    # 先保留原始前置工具结果中的非列表字段
                    if isinstance(pre_tool_results, dict):
                        for key, value in pre_tool_results.items():
                            if not isinstance(value, list):
                                item_pre_tool_results[key] = value
                    
                    # 然后用 item 的数据覆盖（item 优先级更高）
                    if isinstance(item, dict):
                        item_pre_tool_results.update(item)
                    
                    # 使用简单模式处理单个项
                    log.info(f"[process_item]开始处理并行项 {item_pre_tool_results}")
                    result = await self.process_simple_mode(state, item_pre_tool_results)
                    return result
                except Exception as e:
                    log.error(f"并行处理单个项失败: {e}")
                    return {"error": str(e)}
        
        # ----- 并行执行所有任务 -----
        tasks = [process_item(item) for item in parallel_items]
        results = await asyncio.gather(*tasks)
        
        log.info(f"并行模式执行完成，共处理 {len(results)} 个任务")
        
        return {
            "parallel_results": results,
            "total_processed": len(results)
        }

    # =========================================================================
    # G. 执行模式 - 图模式（ReAct 子图）
    # =========================================================================
    
    async def _execute_react_graph(self, state: MainState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        自动构建和执行 ReAct 子图
        
        使用 LangGraph 构建包含 assistant 和 tools 节点的子图，
        实现工具调用的自动循环。
        
        Args:
            state (MainState): 主状态对象
            pre_tool_results (Dict[str, Any]): 前置工具执行结果
        
        Returns:
            Dict[str, Any]: 子图执行结果
        
        子图结构：
            entry -> assistant -> [tools_condition] -> tools -> assistant -> ...
        """
        from langgraph.graph import StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition
        
        log.info(f"开始构建 {self.role_name} 的子图...")
        
        # 1. 获取后置工具
        post_tools = self.get_post_tools()
        if not post_tools:
            log.warning(f"{self.role_name} 没有后置工具，回退到简单模式")
            return await self.process_simple_mode(state, pre_tool_results)
                
        # 2. 使用 MainState 作为子图状态
        log.critical(f"state: {state.agent_results}")
        subgraph = StateGraph(type(state))
        
        # 3. 创建 assistant 节点函数
        assistant_func = self.create_assistant_node_func(state, pre_tool_results)
        
        # 4. 添加节点
        subgraph.add_node("assistant", assistant_func)
        subgraph.add_node("tools", ToolNode(post_tools))
        
        # 5. 添加边
        subgraph.add_conditional_edges("assistant", tools_condition)
        subgraph.add_edge("tools", "assistant")
        
        # 6. 设置入口点
        subgraph.set_entry_point("assistant")
        
        # 7. 编译并执行
        compiled_graph = subgraph.compile()
        log.info(f"{self.role_name} 子图编译完成")
        
        try:
            # 执行子图
            final_state = await compiled_graph.ainvoke(state)
            log.info(f"{self.role_name} 子图执行完成")
            
            # 8. 从 final_state 中提取结果
            result = final_state["agent_results"].get(self.role_name.lower(), {}).get("results", {})
            
            if not result:
                log.error("子图执行后未找到结果")
                return {"error": "子图执行异常：未找到结果"}
            
            log.info(f"{self.role_name} 子图结果解析完成")
            return result
            
        except Exception as e:
            log.exception(f"{self.role_name} 子图执行失败: {e}")
            return {"error": f"子图执行失败: {str(e)}"}
    
    def create_assistant_node_func(self, state: MainState, pre_tool_results: Dict[str, Any]):
        """
        创建 assistant 节点函数
        
        为 LangGraph 子图创建 assistant 节点的处理函数。
        
        Args:
            state (MainState): 主状态对象
            pre_tool_results (Dict[str, Any]): 前置工具执行结果
        
        Returns:
            Callable: 异步节点处理函数
        """
        async def assistant_node(graph_state):
            # 获取或构建消息
            messages = graph_state.get("messages", [])
            if not messages:
                messages = self.build_messages(state, pre_tool_results)
                log.info(f"构建 {self.role_name} 初始消息，包含前置工具结果")

            # 调用 LLM
            response = await self.process_with_llm_for_graph(messages, state)

            # 检查是否有工具调用
            if self.has_tool_calls(response):
                log.info(f"[create_assistant_node_func]: {self.role_name} LLM选择调用工具: ...")
                return {"messages": messages + [response]}
            else:
                # 没有工具调用，解析最终结果
                log.info(f"[create_assistant_node_func]: {self.role_name} LLM本次未调用工具，解析最终结果")
                result = self.parse_result(response.content)
                
                # 同步 agent_results
                state.agent_results[self.role_name.lower()] = {
                    "pre_tool_results": pre_tool_results,
                    "post_tools": [t.name for t in self.get_post_tools()],
                    "results": result
                }
                
                return {
                    "messages": messages + [response],
                    self.role_name.lower(): result,
                    "finished": True
                }
        
        return assistant_node

    # =========================================================================
    # H. ReAct 验证器
    # =========================================================================
    
    def get_react_validators(self) -> List[ValidatorFunc]:
        """
        获取 ReAct 模式的验证器列表 - 子类可重写
        
        验证器用于检查 LLM 输出是否符合预期格式和内容要求。
        
        Returns:
            List[ValidatorFunc]: 验证器函数列表
        
        Example:
            >>> def get_react_validators(self):
            ...     return [
            ...         self._default_json_validator,
            ...         self._check_required_fields,
            ...         self._validate_data_format,
            ...     ]
        """
        return [
            self._default_json_validator,
        ]
    
    @staticmethod
    def _default_json_validator(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        默认 JSON 格式验证器
        
        检查解析结果是否为有效的非空 JSON。
        
        Args:
            content (str): LLM 原始输出
            parsed_result (Dict[str, Any]): 解析后的结果
        
        Returns:
            Tuple[bool, Optional[str]]: (是否通过, 错误信息)
        """
        # 检查是否解析失败（只有 raw 字段）
        if "raw" in parsed_result and len(parsed_result) == 1:
            return False, (
                "你返回的内容不是有效的JSON格式。请确保返回纯JSON格式的数据，"
                "不要包含其他文字说明。正确的格式示例：\n"
                '{"key1": "value1", "key2": "value2"}'
            )
        
        # 检查是否为空字典
        if not parsed_result or (isinstance(parsed_result, dict) and not parsed_result):
            return False, "你返回的JSON为空，请提供完整的结果数据。"
        
        return True, None
    
    def _run_validators(self, content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        运行所有验证器
        
        Args:
            content (str): LLM 原始输出
            parsed_result (Dict[str, Any]): 解析后的结果
        
        Returns:
            Tuple[bool, List[str]]: (是否全部通过, 错误信息列表)
        """
        validators = self.get_react_validators()
        errors = []
        
        for i, validator in enumerate(validators):
            try:
                passed, error_msg = validator(content, parsed_result)
                if not passed:
                    validator_name = getattr(validator, '__name__', f'validator_{i}')
                    log.warning(f"验证器 {validator_name} 未通过: {error_msg}")
                    if error_msg:
                        errors.append(error_msg)
            except Exception as e:
                log.exception(f"验证器执行出错: {e}")
                errors.append(f"验证过程出错: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _build_validation_feedback(self, errors: List[str]) -> str:
        """
        构建验证失败的反馈消息
        
        Args:
            errors (List[str]): 错误信息列表
        
        Returns:
            str: 格式化的反馈消息
        """
        if not errors:
            return "输出格式有误，请按要求重新生成。"
        
        feedback_parts = ["你的输出存在以下问题，请修正后重新生成：\n"]
        for i, error in enumerate(errors, 1):
            feedback_parts.append(f"{i}. {error}")
        
        feedback_parts.append("\n请仔细检查并重新输出正确的结果。")
        return "\n".join(feedback_parts)

    # =========================================================================
    # I. Agent-as-Tool 功能
    # =========================================================================
    
    def get_tool_name(self) -> str:
        """
        获取作为工具时的名称 - 子类可重写
        
        Returns:
            str: 工具名称，格式为 "call_{role_name}_agent"
        """
        return f"call_{self.role_name.lower()}_agent"

    def get_tool_description(self) -> str:
        """
        获取作为工具时的描述 - 子类应重写提供更具体的描述
        
        Returns:
            str: 工具描述
        """
        return f"调用 {self.role_name} agent 来执行特定任务。该 agent 会根据输入参数执行相应的分析和处理。"

    def get_tool_args_schema(self) -> Type[BaseModel]:
        """
        获取作为工具时的参数模式 - 子类可重写
        
        Returns:
            Type[BaseModel]: Pydantic 模型类，定义工具参数结构
        """
        class DefaultAgentToolArgs(BaseModel):
            """默认 Agent 工具参数"""
            task_description: str = Field(
                description=f"传递给 {self.role_name} 的任务描述或指令"
            )
            additional_params: Optional[Dict[str, Any]] = Field(
                default=None,
                description="额外的参数，会被合并到前置工具结果中"
            )
        
        return DefaultAgentToolArgs

    def prepare_tool_execution_params(self, **tool_kwargs) -> Dict[str, Any]:
        """
        准备工具执行时的参数 - 子类可重写
        
        Args:
            **tool_kwargs: 工具调用时传入的参数
        
        Returns:
            Dict[str, Any]: 处理后的参数字典
        """
        params = {}
        
        # 合并 additional_params
        if 'additional_params' in tool_kwargs and tool_kwargs['additional_params']:
            params.update(tool_kwargs['additional_params'])
        
        # 添加其他参数
        for key, value in tool_kwargs.items():
            if key != 'additional_params':
                params[key] = value
        
        return params

    def extract_tool_result(self, state: MainState) -> Dict[str, Any]:
        """
        从状态中提取工具调用的结果 - 子类可重写
        
        Args:
            state (MainState): 执行后的状态对象
        
        Returns:
            Dict[str, Any]: 提取的结果
        """
        agent_result = state.agent_results.get(self.role_name, {})
        return agent_result.get('results', {})

    async def _execute_as_tool(self, state: MainState, **tool_kwargs) -> Dict[str, Any]:
        """
        作为工具执行的内部方法
        
        Args:
            state (MainState): 当前状态对象
            **tool_kwargs: 工具参数
        
        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            log.info(f"[Agent-as-Tool] 调用 {self.role_name}，参数: {tool_kwargs}")
            
            # 准备执行参数
            exec_params = self.prepare_tool_execution_params(**tool_kwargs)
            
            # 执行 Agent
            result_state = await self.execute(
                state, 
                use_agent=False,
                **exec_params
            )
            
            # 提取结果
            result = self.extract_tool_result(result_state)
            
            log.info(f"[Agent-as-Tool] {self.role_name} 执行完成")
            return result
            
        except Exception as e:
            log.exception(f"[Agent-as-Tool] {self.role_name} 执行失败: {e}")
            return {
                "error": str(e),
                "agent": self.role_name,
                "status": "failed"
            }

    def as_tool(self, state: MainState) -> Tool:
        """
        将 Agent 包装成可被调用的工具
        
        Args:
            state (MainState): 状态对象，将被传递给 Agent 执行
        
        Returns:
            Tool: LangChain Tool 实例
        
        Example:
            >>> writer_tool = writer_agent.as_tool(state)
            >>> result = await writer_tool.ainvoke({"task_description": "写一篇文章"})
        """
        async def agent_tool_func(**kwargs) -> Dict[str, Any]:
            return await self._execute_as_tool(state, **kwargs)
        
        def sync_agent_tool_func(**kwargs) -> Dict[str, Any]:
            return asyncio.run(agent_tool_func(**kwargs))
        
        return Tool(
            name=self.get_tool_name(),
            description=self.get_tool_description(),
            func=sync_agent_tool_func,
            coroutine=agent_tool_func,
            args_schema=self.get_tool_args_schema()
        )

    # =========================================================================
    # J. 状态管理与输出
    # =========================================================================
    
    def update_state_result(self, state: MainState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """
        更新状态结果 - 子类可重写
        
        将执行结果存储到状态对象中。
        
        Args:
            state (MainState): 状态对象
            result (Dict[str, Any]): 执行结果
            pre_tool_results (Dict[str, Any]): 前置工具结果
        """
        # 将结果存储到与角色名对应的属性中
        setattr(state, self.role_name.lower(), result)
        
        # 存储到 agent_results
        state.agent_results[self.role_name] = {
            "pre_tool_results": pre_tool_results,
            "post_tools": [t.name for t in self.get_post_tools()],
            "results": result
        }

    def store_outputs(self, data, file_name: str = None) -> str:
        """
        保存输出结果到文件
        
        Args:
            data: 要保存的数据
            file_name (str, optional): 文件名，默认使用时间戳
        
        Returns:
            str: 保存的文件路径
        """
        # 创建输出目录
        out_dir = Path(f"{PROJDIR}/outputs/{self.role_name.lower()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        if not file_name:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{ts}.pkl"
        
        file_path = out_dir / file_name
        
        # 保存数据
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        
        log.info(f"已保存->: {file_path}")
        return str(file_path)

    # =========================================================================
    # K. 主执行入口
    # =========================================================================
    
    async def execute(self, state: MainState, use_agent: bool = False, **kwargs) -> MainState:
        """
        统一执行入口 - Agent 的核心执行方法
        
        根据配置选择合适的执行模式，完成 Agent 的完整执行流程。
        
        Args:
            state (MainState): 当前状态对象
            use_agent (bool): 是否使用代理模式（图模式），默认 False
            **kwargs: 额外参数，会被合并到前置工具结果中
        
        Returns:
            MainState: 更新后的状态对象
        
        执行流程：
            1. 检查是否使用策略模式
            2. 检查是否使用 VLM 模式
            3. 执行前置工具
            4. 根据配置选择执行模式：
               - 图模式（use_agent=True 且有后置工具）
               - ReAct 模式（react_mode=True）
               - 简单模式（默认）
            5. 更新状态结果
        
        Example:
            >>> state = await agent.execute(state, use_agent=True)
            >>> result = state.agent_results["Writer"]["results"]
        """
        # 保存状态引用
        self.state = state
        
        # ----- 策略模式执行 -----
        if self._execution_strategy:
            log.info(f"使用策略模式执行: {self._execution_strategy.__class__.__name__}")
            try:
                pre_tool_results = await self.execute_pre_tools(state)
                result = await self._execution_strategy.execute(state, **kwargs)
                self.update_state_result(state, result, pre_tool_results)
                return state
            except Exception as e:
                log.exception(f"策略执行失败: {e}")
                error_result = {"error": str(e)}
                self.update_state_result(state, error_result, {})
                return state
            
        # ----- 常规执行流程 -----
        log.info(f"开始执行 {self.role_name} (ReAct模式: {self.react_mode}, 图模式: {use_agent})")

        # VLM 模式
        if getattr(self, "use_vlm", False):
            log.critical(f'[base agent]: 走多模态路径')
            result = await self._execute_vlm(state, **kwargs)
            self.update_state_result(state, result, {})
            log.info(f"{self.role_name} 多模态执行完成")
            return state
        
        try:
            # 1. 执行前置工具
            pre_tool_results = await self.execute_pre_tools(state)
            
            # 1.1 写入 temp_data
            try:
                if not hasattr(state, 'temp_data') or state.temp_data is None:
                    state.temp_data = {}
                state.temp_data['pre_tool_results'] = pre_tool_results
            except Exception:
                pass
            
            # 1.2 合并 kwargs 到前置工具结果
            pre_tool_results.update(kwargs)
            
            # 2. 获取后置工具
            post_tools = self.get_post_tools()
            
            # 3. 根据模式选择处理方式
            if use_agent and post_tools:
                # ----- 图模式 -----
                log.info(f"[子图新模式] 自动构建 {self.role_name} 的子图，"
                        f"后置工具: {[t.name for t in post_tools]}")
                result = await self._execute_react_graph(state, pre_tool_results)
                self.update_state_result(state, result, pre_tool_results)
                log.info(f"[子图新模式] {self.role_name} 子图模式执行完成")
                
                # 更新 temp_data
                if not hasattr(state, 'temp_data'):
                    state.temp_data = {}
                state.temp_data['pre_tool_results'] = pre_tool_results
                state.temp_data[f'{self.role_name}_instance'] = self
                
            elif self.react_mode:
                # ----- ReAct 模式 -----
                log.info("ReAct模式 - 带验证循环")
                result = await self.process_react_mode(state, pre_tool_results)
                self.update_state_result(state, result, pre_tool_results)
                log.info(f"{self.role_name} ReAct模式执行完成")
                
            else:
                # ----- 简单模式 -----
                if use_agent and not post_tools:
                    log.info("图模式无可用后置工具，回退到简单模式")
                result = await self.process_simple_mode(state, pre_tool_results)
                self.update_state_result(state, result, pre_tool_results)
                log.info(f"{self.role_name} 简单模式执行完成")
            
        except Exception as e:
            log.exception(f"{self.role_name} 执行失败: {e}")
            import traceback
            traceback.print_exc()
            error_result = {"error": str(e)}
            self.update_state_result(state, error_result, {})
            
        return state
    
    async def _execute_vlm(self, state: MainState, **kwargs) -> Dict[str, Any]:
        """
        Vision-LLM 专用执行流程
        
        与文本链路完全解耦，专门处理视觉语言模型的调用。
        
        Args:
            state (MainState): 当前状态对象
            **kwargs: 额外参数
        
        Returns:
            Dict[str, Any]: VLM 执行结果
        """
        # 1. 执行前置工具
        pre_tool_results = await self.execute_pre_tools(state)
    
        # 2. 构建消息
        mode = self.vlm_config.get("mode", "understanding")
        messages = self.build_messages(state, pre_tool_results)
    
        # 3. 调用 VisionLLMCaller
        from dataflow_agent.llm_callers import VisionLLMCaller
        vlm_caller = VisionLLMCaller(
            state,
            vlm_config=self.vlm_config,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tool_mode=self.tool_mode,
            tool_manager=self.tool_manager,
        )
        response = await vlm_caller.call(messages)
        log.info(f"{self.role_name} 多模态原始响应: {response}")
    
        # 4. 解析结果
        parsed = self.parse_result(response.content)
    
        # 5. 合并附加信息（如图像路径、base64 等）
        if hasattr(response, "additional_kwargs"):
            log.info(f"{self.role_name} 多模态附加信息: {response.additional_kwargs}")
            
            if isinstance(parsed, dict):
                parsed.update(response.additional_kwargs)
            elif isinstance(parsed, list):
                log.warning(f"{self.role_name} parsed 是列表类型，无法调用 update 方法，跳过附加参数合并")
                log.info(f"parsed 类型: {type(parsed).__name__}, 内容: {parsed}")
                log.info(f"additional_kwargs 内容: {response.additional_kwargs}")
            else:
                log.warning(f"{self.role_name} parsed 是 {type(parsed).__name__} 类型，无法调用 update 方法，跳过附加参数合并")
                log.info(f"parsed 内容: {parsed}")
                log.info(f"additional_kwargs 内容: {response.additional_kwargs}")
        
        return parsed
