from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
from dataflow_agent.logger import get_logger
import asyncio  # 添加导入

if TYPE_CHECKING:
    from dataflow_agent.agentroles.base_agent import BaseAgent
    from dataflow_agent.state import MainState

log = get_logger(__name__)


class ExecutionStrategy(ABC):
    """执行策略基类"""
    
    def __init__(self, agent: "BaseAgent", config: Any):
        self.agent = agent
        self.config = config
    
    @abstractmethod
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        """执行策略的核心方法"""
        pass


class SimpleStrategy(ExecutionStrategy):
    """简单模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[SimpleStrategy] 执行 {self.agent.role_name}")
        pre_tool_results = await self.agent.execute_pre_tools(state)
        result = await self.agent.process_simple_mode(state, pre_tool_results)
        return result


class ReactStrategy(ExecutionStrategy):
    """ReAct模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[ReactStrategy] 执行 {self.agent.role_name}，最大重试: {self.config.max_retries}")
        
        # 注入自定义验证器
        if self.config.validators:
            original_validators = self.agent.get_react_validators
            def custom_validators():
                return original_validators() + self.config.validators
            self.agent.get_react_validators = custom_validators
        
        pre_tool_results = await self.agent.execute_pre_tools(state)
        
        # 临时覆盖 react_max_retries
        original_retries = self.agent.react_max_retries
        self.agent.react_max_retries = self.config.max_retries
        
        result = await self.agent.process_react_mode(state, pre_tool_results)
        
        self.agent.react_max_retries = original_retries
        return result


class GraphStrategy(ExecutionStrategy):
    """图模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[GraphStrategy] 执行 {self.agent.role_name} 子图模式")
        pre_tool_results = await self.agent.execute_pre_tools(state)
        
        post_tools = self.agent.get_post_tools()
        if not post_tools:
            log.warning("无后置工具，回退到简单模式")
            return await self.agent.process_simple_mode(state, pre_tool_results)
        
        result = await self.agent._execute_react_graph(state, pre_tool_results)
        return result


class VLMStrategy(ExecutionStrategy):
    """视觉语言模型策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[VLMStrategy] 执行 {self.agent.role_name} VLM模式: {self.config.vlm_mode}")
        
        # 构建 VLM 配置字典
        vlm_config = {
            "mode": self.config.vlm_mode,
            "image_detail": self.config.image_detail,
            "max_image_size": self.config.max_image_size,
            **self.config.additional_params
        }
        
        # 临时注入配置
        # original_vlm_config = getattr(self.agent, 'vlm_config', {})

        self.agent.vlm_config.update(vlm_config)
        self.agent.model_name = self.config.model_name
        self.agent.temperature = self.config.temperature
        self.agent.max_tokens = self.config.max_tokens

        log.info(f"VLMStrategy 执行 {self.agent.role_name} VLM模式: {self.agent.vlm_config} + {kwargs},self.config={self.config}")
        result = await self.agent._execute_vlm(state, **kwargs)
        log.critical(f"VLMStrategy 执行 {self.agent.role_name} VLM模式结果: {result}")
        
        # self.agent.vlm_config = original_vlm_config
        return result


class ParallelStrategy(ExecutionStrategy):
    """并行模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[ParallelStrategy] 执行 {self.agent.role_name}，并行度限制: {self.config.concurrency_limit}")
        
        # 执行前置工具获取结果
        pre_tool_results = await self.agent.execute_pre_tools(state)
        
        # 检查是否有parallel_items，如果没有且pre_tool_results是列表，自动转换为parallel_items
        if "parallel_items" not in pre_tool_results and isinstance(pre_tool_results, list):
            pre_tool_results = {"parallel_items": pre_tool_results}
        
        # 执行并行模式
        return await self.agent.process_parallel_mode(state, pre_tool_results)


class StrategyFactory:
    """策略工厂"""
    
    _strategies = {
        "simple": SimpleStrategy,
        "react": ReactStrategy,
        "graph": GraphStrategy,
        "vlm": VLMStrategy,
        "parallel": ParallelStrategy,  # 注册并行策略
    }
    
    @classmethod
    def create(cls, mode: str, agent: "BaseAgent", config: Any) -> ExecutionStrategy:
        strategy_cls = cls._strategies.get(mode.lower())
        if not strategy_cls:
            raise ValueError(f"不支持的执行模式: {mode}，可选: {list(cls._strategies.keys())}")
        return strategy_cls(agent, config)
    
    @classmethod
    def register(cls, mode: str, strategy_cls: type):
        """注册自定义策略"""
        cls._strategies[mode.lower()] = strategy_cls