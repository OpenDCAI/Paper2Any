"""
TRJ 数据模型定义

定义标准的轨迹数据结构，支持：
1. ReAct 模式：Context -> Thought -> Action -> Observation
2. Workflow 模式：State_In -> Node Processing -> State_Update
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class TrajectoryMode(str, Enum):
    """轨迹模式"""
    REACT = "react"
    WORKFLOW = "workflow"
    HYBRID = "hybrid"  # 混合模式


class StepRole(str, Enum):
    """步骤角色"""
    AGENT = "agent"
    ENVIRONMENT = "environment"
    SYSTEM_NODE = "system_node"
    TOOL = "tool"
    USER = "user"


class ActionType(str, Enum):
    """动作类型"""
    TOOL_CALL = "tool_call"
    RESPONSE = "response"
    STATE_UPDATE = "state_update"
    LLM_CALL = "llm_call"
    MULTIMODAL = "multimodal"


@dataclass
class ToolCallRecord:
    """工具调用记录"""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Any
    timestamp: str
    duration_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class LLMCallRecord:
    """LLM 调用记录"""
    model: str
    messages_in: List[Dict[str, Any]]  # 输入消息
    response: str  # 输出响应
    timestamp: str
    duration_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None  # {"prompt": x, "completion": y}
    temperature: Optional[float] = None


@dataclass
class MultimodalData:
    """多模态数据"""
    type: str  # "image" | "audio" | "video"
    path: Optional[str] = None  # 文件路径
    url: Optional[str] = None  # URL
    base64: Optional[str] = None  # Base64 编码（不推荐存储大数据）
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryStep:
    """
    单个执行步骤
    
    对于 ReAct 模式：
    - input_context: Agent 看到的上下文
    - thought: Agent 的思考过程
    - action_type: 动作类型（tool_call/response）
    - action_payload: 动作内容
    - observation: 环境反馈
    
    对于 Workflow 模式：
    - input_context: 节点输入状态
    - node_output: 节点输出/状态更新
    """
    step_index: int
    node_name: str
    role: str  # StepRole 的值
    timestamp: str
    
    # 输入上下文
    input_context: Dict[str, Any] = field(default_factory=dict)
    
    # ReAct 特有字段
    thought: Optional[str] = None
    action_type: Optional[str] = None  # ActionType 的值
    action_payload: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    
    # 通用输出
    node_output: Optional[Dict[str, Any]] = None
    
    # 详细记录
    llm_calls: List[LLMCallRecord] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    
    # 多模态数据
    multimodal_input: Optional[MultimodalData] = None
    multimodal_output: Optional[MultimodalData] = None
    
    # 错误信息
    error: Optional[str] = None
    
    # 执行时间
    duration_ms: Optional[float] = None
    
    # 额外元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "step_index": self.step_index,
            "node_name": self.node_name,
            "role": self.role,
            "timestamp": self.timestamp,
        }
        
        # 只添加非空字段
        if self.input_context:
            result["input_context"] = self.input_context
        if self.thought:
            result["thought"] = self.thought
        if self.action_type:
            result["action_type"] = self.action_type
        if self.action_payload:
            result["action_payload"] = self.action_payload
        if self.observation:
            result["observation"] = self.observation
        if self.node_output:
            result["node_output"] = self.node_output
        if self.llm_calls:
            result["llm_calls"] = [self._llm_call_to_dict(c) for c in self.llm_calls]
        if self.tool_calls:
            result["tool_calls"] = [self._tool_call_to_dict(c) for c in self.tool_calls]
        if self.multimodal_input:
            result["multimodal_input"] = self._multimodal_to_dict(self.multimodal_input)
        if self.multimodal_output:
            result["multimodal_output"] = self._multimodal_to_dict(self.multimodal_output)
        if self.error:
            result["error"] = self.error
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    @staticmethod
    def _llm_call_to_dict(call: LLMCallRecord) -> Dict[str, Any]:
        return {
            "model": call.model,
            "messages_in": call.messages_in,
            "response": call.response,
            "timestamp": call.timestamp,
            "duration_ms": call.duration_ms,
            "token_usage": call.token_usage,
            "temperature": call.temperature,
        }
    
    @staticmethod
    def _tool_call_to_dict(call: ToolCallRecord) -> Dict[str, Any]:
        return {
            "tool_name": call.tool_name,
            "tool_args": call.tool_args,
            "tool_result": call.tool_result,
            "timestamp": call.timestamp,
            "duration_ms": call.duration_ms,
            "error": call.error,
        }
    
    @staticmethod
    def _multimodal_to_dict(data: MultimodalData) -> Dict[str, Any]:
        result = {"type": data.type}
        if data.path:
            result["path"] = data.path
        if data.url:
            result["url"] = data.url
        if data.metadata:
            result["metadata"] = data.metadata
        # 不导出 base64 以减小文件大小
        return result


@dataclass
class TrajectoryFeedback:
    """用户反馈"""
    score: Optional[int] = None  # 1-5 评分
    comment: Optional[str] = None
    edited_response: Optional[str] = None  # 用户修改后的回答（用于 SFT）
    labels: List[str] = field(default_factory=list)  # 标签，如 ["good", "accurate"]
    timestamp: Optional[str] = None


@dataclass
class Trajectory:
    """
    完整的执行轨迹
    
    包含三个核心部分：
    1. Metadata: 元数据
    2. Steps: 执行步骤列表
    3. Outcome: 最终结果和反馈
    """
    # ===== 元数据 =====
    trace_id: str
    workflow_name: str
    timestamp: str
    status: str  # "success" | "failed" | "partial"
    mode: str  # TrajectoryMode 的值
    
    # 可选元数据
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    version: str = "1.0"
    
    # ===== 输入 =====
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # ===== 执行步骤 =====
    steps: List[TrajectoryStep] = field(default_factory=list)
    
    # ===== 输出 =====
    final_output: Any = None
    
    # ===== 反馈 =====
    feedback: Optional[TrajectoryFeedback] = None
    
    # ===== 统计信息 =====
    total_duration_ms: Optional[float] = None
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tokens: Optional[Dict[str, int]] = None
    
    # ===== 额外元数据 =====
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_trace_id() -> str:
        """生成唯一的 trace_id"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"trj_{timestamp}_{unique_id}"

    def add_step(self, step: TrajectoryStep):
        """添加执行步骤"""
        self.steps.append(step)
        # 更新统计
        self.total_llm_calls += len(step.llm_calls)
        self.total_tool_calls += len(step.tool_calls)

    def set_feedback(self, score: int = None, comment: str = None, 
                     edited_response: str = None, labels: List[str] = None):
        """设置用户反馈"""
        self.feedback = TrajectoryFeedback(
            score=score,
            comment=comment,
            edited_response=edited_response,
            labels=labels or [],
            timestamp=datetime.now().isoformat()
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 导出）"""
        result = {
            # 元数据
            "trace_id": self.trace_id,
            "workflow_name": self.workflow_name,
            "timestamp": self.timestamp,
            "status": self.status,
            "mode": self.mode,
            "version": self.version,
            
            # 输入
            "inputs": self.inputs,
            
            # 步骤
            "steps": [step.to_dict() for step in self.steps],
            
            # 输出
            "final_output": self.final_output,
            
            # 统计
            "statistics": {
                "total_steps": len(self.steps),
                "total_llm_calls": self.total_llm_calls,
                "total_tool_calls": self.total_tool_calls,
                "total_duration_ms": self.total_duration_ms,
                "total_tokens": self.total_tokens,
            }
        }
        
        # 可选字段
        if self.user_id:
            result["user_id"] = self.user_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.feedback:
            result["feedback"] = {
                "score": self.feedback.score,
                "comment": self.feedback.comment,
                "edited_response": self.feedback.edited_response,
                "labels": self.feedback.labels,
                "timestamp": self.feedback.timestamp,
            }
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result

    def to_sft_format(self) -> List[Dict[str, str]]:
        """
        转换为 SFT 训练格式（OpenAI messages 格式）
        
        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        messages = []
        
        for step in self.steps:
            if step.role == StepRole.AGENT.value:
                # Agent 的输出
                content = ""
                if step.thought:
                    content += f"<thought>{step.thought}</thought>\n"
                if step.action_type == ActionType.TOOL_CALL.value and step.action_payload:
                    tool_name = step.action_payload.get("tool_name", "")
                    tool_args = step.action_payload.get("tool_args", "")
                    content += f"<call>{tool_name}({tool_args})</call>"
                elif step.node_output:
                    content += str(step.node_output)
                
                if content:
                    messages.append({"role": "assistant", "content": content})
                    
            elif step.role in [StepRole.ENVIRONMENT.value, StepRole.TOOL.value]:
                # 工具/环境的输出
                if step.observation:
                    messages.append({"role": "tool", "content": step.observation})
                    
            elif step.role == StepRole.USER.value:
                # 用户输入
                if step.input_context:
                    content = step.input_context.get("query", str(step.input_context))
                    messages.append({"role": "user", "content": content})
        
        return messages

    def to_dpo_format(self) -> Dict[str, Any]:
        """
        转换为 DPO 训练格式
        
        Returns:
            {"prompt": "...", "chosen": [...], "rejected": [...]}
        """
        # 提取 prompt
        prompt = self.inputs.get("query", self.inputs.get("target", ""))
        
        # 当前轨迹作为 chosen 或 rejected
        trajectory_steps = self.to_sft_format()
        
        return {
            "prompt": prompt,
            "trajectory": trajectory_steps,
            "score": self.feedback.score if self.feedback else None,
            "status": self.status,
        }
