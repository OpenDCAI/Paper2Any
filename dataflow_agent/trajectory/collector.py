"""
轨迹收集器 - 实时捕获 Workflow 执行过程数据

通过 Hook 机制在执行过程中收集：
- Agent 执行信息
- LLM 调用记录
- 工具调用记录
- 状态变化
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from dataflow_agent.trajectory.models import (
    TrajectoryStep,
    LLMCallRecord,
    ToolCallRecord,
    MultimodalData,
    StepRole,
    ActionType,
)
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


@dataclass
class StepContext:
    """步骤上下文 - 用于跟踪当前正在执行的步骤"""
    step_index: int
    node_name: str
    role: str
    start_time: float
    input_context: Dict[str, Any] = field(default_factory=dict)
    llm_calls: List[LLMCallRecord] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    thought: Optional[str] = None
    action_type: Optional[str] = None
    action_payload: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    node_output: Optional[Dict[str, Any]] = None
    multimodal_input: Optional[MultimodalData] = None
    multimodal_output: Optional[MultimodalData] = None
    error: Optional[str] = None


class TrajectoryCollector:
    """
    轨迹收集器
    
    使用方式：
    1. 在 Workflow 开始时调用 start()
    2. 在每个节点执行前调用 on_node_start()
    3. 在 LLM/工具调用时调用相应的 on_xxx 方法
    4. 在每个节点执行后调用 on_node_end()
    5. 在 Workflow 结束时调用 finish() 获取所有步骤
    """
    
    def __init__(self):
        self.steps: List[TrajectoryStep] = []
        self.current_step: Optional[StepContext] = None
        self.step_counter: int = 0
        self.start_time: Optional[float] = None
        self.is_recording: bool = False
        
        # 初始输入
        self.initial_inputs: Dict[str, Any] = {}
        
        # 元数据
        self.metadata: Dict[str, Any] = {}
    
    def start(self, inputs: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """
        开始记录
        
        Args:
            inputs: 初始输入数据
            metadata: 额外元数据
        """
        self.steps = []
        self.current_step = None
        self.step_counter = 0
        self.start_time = time.time()
        self.is_recording = True
        self.initial_inputs = inputs or {}
        self.metadata = metadata or {}
        
        log.info(f"[TrajectoryCollector] 开始记录轨迹")
    
    def finish(self) -> List[TrajectoryStep]:
        """
        结束记录并返回所有步骤
        
        Returns:
            收集到的所有步骤
        """
        # 如果有未完成的步骤，先完成它
        if self.current_step:
            self._finalize_current_step()
        
        self.is_recording = False
        total_duration = (time.time() - self.start_time) * 1000 if self.start_time else 0
        
        log.info(f"[TrajectoryCollector] 记录完成，共 {len(self.steps)} 个步骤，"
                f"总耗时 {total_duration:.2f}ms")
        
        return self.steps
    
    def on_node_start(self, 
                      node_name: str, 
                      role: str = StepRole.SYSTEM_NODE.value,
                      input_context: Dict[str, Any] = None):
        """
        节点开始执行
        
        Args:
            node_name: 节点名称
            role: 角色类型
            input_context: 输入上下文
        """
        if not self.is_recording:
            return
        
        # 如果有未完成的步骤，先完成它
        if self.current_step:
            self._finalize_current_step()
        
        # 创建新的步骤上下文
        self.current_step = StepContext(
            step_index=self.step_counter,
            node_name=node_name,
            role=role,
            start_time=time.time(),
            input_context=input_context or {}
        )
        
        log.debug(f"[TrajectoryCollector] 节点开始: {node_name} (step {self.step_counter})")
    
    def on_node_end(self, 
                    output: Dict[str, Any] = None,
                    error: str = None):
        """
        节点执行结束
        
        Args:
            output: 节点输出
            error: 错误信息
        """
        if not self.is_recording or not self.current_step:
            return
        
        if output:
            self.current_step.node_output = output
        if error:
            self.current_step.error = error
        
        self._finalize_current_step()
        
        log.debug(f"[TrajectoryCollector] 节点结束: {self.current_step.node_name if self.current_step else 'unknown'}")
    
    def on_llm_call(self,
                    model: str,
                    messages: List[Dict[str, Any]],
                    response: str,
                    duration_ms: float = None,
                    token_usage: Dict[str, int] = None,
                    temperature: float = None):
        """
        记录 LLM 调用
        
        Args:
            model: 模型名称
            messages: 输入消息
            response: 响应内容
            duration_ms: 耗时（毫秒）
            token_usage: Token 使用量
            temperature: 温度参数
        """
        if not self.is_recording:
            return
        
        record = LLMCallRecord(
            model=model,
            messages_in=messages,
            response=response,
            timestamp=datetime.now().isoformat(),
            duration_ms=duration_ms,
            token_usage=token_usage,
            temperature=temperature
        )
        
        if self.current_step:
            self.current_step.llm_calls.append(record)
        else:
            # 如果没有当前步骤，创建一个临时步骤
            log.warning("[TrajectoryCollector] LLM 调用发生在节点外部")
            self.on_node_start("llm_call", StepRole.AGENT.value)
            self.current_step.llm_calls.append(record)
        
        log.debug(f"[TrajectoryCollector] 记录 LLM 调用: {model}")
    
    def on_tool_call(self,
                     tool_name: str,
                     tool_args: Dict[str, Any],
                     tool_result: Any,
                     duration_ms: float = None,
                     error: str = None):
        """
        记录工具调用
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            tool_result: 工具结果
            duration_ms: 耗时（毫秒）
            error: 错误信息
        """
        if not self.is_recording:
            return
        
        record = ToolCallRecord(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            timestamp=datetime.now().isoformat(),
            duration_ms=duration_ms,
            error=error
        )
        
        if self.current_step:
            self.current_step.tool_calls.append(record)
        else:
            log.warning("[TrajectoryCollector] 工具调用发生在节点外部")
            self.on_node_start("tool_call", StepRole.TOOL.value)
            self.current_step.tool_calls.append(record)
        
        log.debug(f"[TrajectoryCollector] 记录工具调用: {tool_name}")
    
    def on_thought(self, thought: str):
        """
        记录 Agent 的思考过程（ReAct 模式）
        
        Args:
            thought: 思考内容
        """
        if not self.is_recording or not self.current_step:
            return
        
        self.current_step.thought = thought
        log.debug(f"[TrajectoryCollector] 记录思考: {thought[:50]}...")
    
    def on_action(self, 
                  action_type: str,
                  action_payload: Dict[str, Any]):
        """
        记录 Agent 的动作（ReAct 模式）
        
        Args:
            action_type: 动作类型
            action_payload: 动作内容
        """
        if not self.is_recording or not self.current_step:
            return
        
        self.current_step.action_type = action_type
        self.current_step.action_payload = action_payload
        log.debug(f"[TrajectoryCollector] 记录动作: {action_type}")
    
    def on_observation(self, observation: str):
        """
        记录环境观察（ReAct 模式）
        
        Args:
            observation: 观察内容
        """
        if not self.is_recording or not self.current_step:
            return
        
        self.current_step.observation = observation
        log.debug(f"[TrajectoryCollector] 记录观察: {observation[:50]}...")
    
    def on_multimodal_input(self,
                            data_type: str,
                            path: str = None,
                            url: str = None,
                            metadata: Dict[str, Any] = None):
        """
        记录多模态输入
        
        Args:
            data_type: 数据类型（image/audio/video）
            path: 文件路径
            url: URL
            metadata: 元数据
        """
        if not self.is_recording or not self.current_step:
            return
        
        self.current_step.multimodal_input = MultimodalData(
            type=data_type,
            path=path,
            url=url,
            metadata=metadata or {}
        )
        log.debug(f"[TrajectoryCollector] 记录多模态输入: {data_type}")
    
    def on_multimodal_output(self,
                             data_type: str,
                             path: str = None,
                             url: str = None,
                             metadata: Dict[str, Any] = None):
        """
        记录多模态输出
        
        Args:
            data_type: 数据类型
            path: 文件路径
            url: URL
            metadata: 元数据
        """
        if not self.is_recording or not self.current_step:
            return
        
        self.current_step.multimodal_output = MultimodalData(
            type=data_type,
            path=path,
            url=url,
            metadata=metadata or {}
        )
        log.debug(f"[TrajectoryCollector] 记录多模态输出: {data_type}")
    
    def _finalize_current_step(self):
        """完成当前步骤并添加到步骤列表"""
        if not self.current_step:
            return
        
        # 计算耗时
        duration_ms = (time.time() - self.current_step.start_time) * 1000
        
        # 创建 TrajectoryStep
        step = TrajectoryStep(
            step_index=self.current_step.step_index,
            node_name=self.current_step.node_name,
            role=self.current_step.role,
            timestamp=datetime.now().isoformat(),
            input_context=self.current_step.input_context,
            thought=self.current_step.thought,
            action_type=self.current_step.action_type,
            action_payload=self.current_step.action_payload,
            observation=self.current_step.observation,
            node_output=self.current_step.node_output,
            llm_calls=self.current_step.llm_calls,
            tool_calls=self.current_step.tool_calls,
            multimodal_input=self.current_step.multimodal_input,
            multimodal_output=self.current_step.multimodal_output,
            error=self.current_step.error,
            duration_ms=duration_ms
        )
        
        self.steps.append(step)
        self.step_counter += 1
        self.current_step = None
    
    def get_current_step_index(self) -> int:
        """获取当前步骤索引"""
        return self.step_counter
    
    def get_steps_count(self) -> int:
        """获取已完成的步骤数量"""
        return len(self.steps)
    
    def get_initial_inputs(self) -> Dict[str, Any]:
        """获取初始输入"""
        return self.initial_inputs
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        return self.metadata


# ==================== 便捷函数 ====================

def create_collector() -> TrajectoryCollector:
    """创建轨迹收集器"""
    return TrajectoryCollector()
