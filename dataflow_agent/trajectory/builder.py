"""
轨迹构建器 - 将收集的原始数据转换为标准 TRJ 格式

支持：
1. 从 State 和 Collector 构建完整轨迹
2. 自动检测 ReAct/Workflow 模式
3. 提取关键信息和统计数据
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from dataflow_agent.trajectory.models import (
    Trajectory,
    TrajectoryStep,
    TrajectoryMode,
    StepRole,
)
from dataflow_agent.trajectory.collector import TrajectoryCollector
from dataflow_agent.state import MainState
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


class TrajectoryBuilder:
    """
    轨迹构建器
    
    将 TrajectoryCollector 收集的原始步骤数据和 State 对象
    转换为标准的 Trajectory 对象
    """
    
    def __init__(self):
        pass
    
    def build_from_state(self,
                        state: MainState,
                        collector: TrajectoryCollector,
                        workflow_name: str,
                        user_id: str = None,
                        session_id: str = None) -> Trajectory:
        """
        从 State 和 Collector 构建完整轨迹
        
        Args:
            state: Workflow 执行后的最终状态
            collector: 轨迹收集器
            workflow_name: Workflow 名称
            user_id: 用户 ID
            session_id: 会话 ID
            
        Returns:
            完整的 Trajectory 对象
        """
        log.info(f"[TrajectoryBuilder] 开始构建轨迹: {workflow_name}")
        
        # 1. 生成 trace_id
        trace_id = Trajectory.generate_trace_id()
        
        # 2. 获取步骤
        steps = collector.finish()
        
        # 3. 检测模式
        mode = self._detect_mode(state, steps)
        
        # 4. 提取输入
        inputs = self._extract_inputs(state, collector)
        
        # 5. 提取输出
        final_output = self._extract_final_output(state)
        
        # 6. 判断状态
        status = self._determine_status(state, steps)
        
        # 7. 计算统计信息
        total_duration_ms = self._calculate_total_duration(steps)
        total_tokens = self._calculate_total_tokens(steps)
        
        # 8. 构建 Trajectory
        trajectory = Trajectory(
            trace_id=trace_id,
            workflow_name=workflow_name,
            timestamp=datetime.now().isoformat(),
            status=status,
            mode=mode,
            user_id=user_id,
            session_id=session_id or getattr(state.request, 'session_id', None),
            inputs=inputs,
            steps=steps,
            final_output=final_output,
            total_duration_ms=total_duration_ms,
            total_tokens=total_tokens,
            metadata=collector.get_metadata()
        )
        
        # 更新统计
        trajectory.total_llm_calls = sum(len(step.llm_calls) for step in steps)
        trajectory.total_tool_calls = sum(len(step.tool_calls) for step in steps)
        
        log.info(f"[TrajectoryBuilder] 轨迹构建完成: {trace_id}, "
                f"模式={mode}, 步骤数={len(steps)}, 状态={status}")
        
        return trajectory
    
    def build_from_steps(self,
                        steps: List[TrajectoryStep],
                        workflow_name: str,
                        inputs: Dict[str, Any] = None,
                        final_output: Any = None,
                        **kwargs) -> Trajectory:
        """
        直接从步骤列表构建轨迹（不依赖 State）
        
        Args:
            steps: 步骤列表
            workflow_name: Workflow 名称
            inputs: 输入数据
            final_output: 最终输出
            **kwargs: 其他参数
            
        Returns:
            Trajectory 对象
        """
        trace_id = Trajectory.generate_trace_id()
        mode = self._detect_mode_from_steps(steps)
        status = "success" if not any(step.error for step in steps) else "failed"
        
        trajectory = Trajectory(
            trace_id=trace_id,
            workflow_name=workflow_name,
            timestamp=datetime.now().isoformat(),
            status=status,
            mode=mode,
            inputs=inputs or {},
            steps=steps,
            final_output=final_output,
            **kwargs
        )
        
        # 更新统计
        trajectory.total_llm_calls = sum(len(step.llm_calls) for step in steps)
        trajectory.total_tool_calls = sum(len(step.tool_calls) for step in steps)
        trajectory.total_duration_ms = self._calculate_total_duration(steps)
        trajectory.total_tokens = self._calculate_total_tokens(steps)
        
        return trajectory
    
    def _detect_mode(self, state: MainState, steps: List[TrajectoryStep]) -> str:
        """
        检测轨迹模式
        
        通过分析 State 和 Steps 判断是 ReAct 还是 Workflow 模式
        """
        # 检查是否有 thought 字段（ReAct 特征）
        has_thoughts = any(step.thought for step in steps)
        
        # 检查是否有 observation 字段（ReAct 特征）
        has_observations = any(step.observation for step in steps)
        
        # 检查消息历史（ReAct 通常有更多的对话轮次）
        messages = getattr(state, 'messages', [])
        has_many_messages = len(messages) > 5
        
        # 检查是否有明确的 agent 角色步骤
        has_agent_steps = any(step.role == StepRole.AGENT.value for step in steps)
        
        if has_thoughts or has_observations:
            return TrajectoryMode.REACT.value
        elif has_agent_steps and has_many_messages:
            return TrajectoryMode.HYBRID.value
        else:
            return TrajectoryMode.WORKFLOW.value
    
    def _detect_mode_from_steps(self, steps: List[TrajectoryStep]) -> str:
        """仅从步骤检测模式"""
        has_thoughts = any(step.thought for step in steps)
        has_observations = any(step.observation for step in steps)
        
        if has_thoughts or has_observations:
            return TrajectoryMode.REACT.value
        else:
            return TrajectoryMode.WORKFLOW.value
    
    def _extract_inputs(self, state: MainState, collector: TrajectoryCollector) -> Dict[str, Any]:
        """
        提取输入数据
        
        优先级：
        1. Collector 记录的初始输入
        2. State.request 中的字段
        3. State 的其他相关字段
        """
        inputs = {}
        
        # 从 collector 获取
        collector_inputs = collector.get_initial_inputs()
        if collector_inputs:
            inputs.update(collector_inputs)
        
        # 从 state.request 提取
        if hasattr(state, 'request'):
            request = state.request
            
            # 提取常见字段
            if hasattr(request, 'target') and request.target:
                inputs['query'] = request.target
            
            if hasattr(request, 'model'):
                inputs['model'] = request.model
            
            if hasattr(request, 'language'):
                inputs['language'] = request.language
            
            # 提取其他可能的输入字段
            for field in ['json_file', 'python_file_path', 'keywords', 'style']:
                if hasattr(request, field):
                    value = getattr(request, field)
                    if value:
                        inputs[field] = value
        
        return inputs
    
    def _extract_final_output(self, state: MainState) -> Any:
        """
        提取最终输出
        
        从 State 的不同字段中提取最终结果
        """
        # 尝试从 agent_results 获取
        if hasattr(state, 'agent_results') and state.agent_results:
            # 获取最后一个 agent 的结果
            last_agent_result = None
            for agent_name, result in state.agent_results.items():
                if isinstance(result, dict) and 'results' in result:
                    last_agent_result = result['results']
            
            if last_agent_result:
                return last_agent_result
        
        # 尝试从特定字段获取
        for field in ['final_output', 'execution_result', 'pipeline_structure_code', 
                     'recommendation', 'icon_prompt', 'research_summary']:
            if hasattr(state, field):
                value = getattr(state, field)
                if value:
                    return value
        
        # 如果都没有，返回整个 state 的字典表示（简化版）
        return {"status": "completed"}
    
    def _determine_status(self, state: MainState, steps: List[TrajectoryStep]) -> str:
        """
        判断执行状态
        
        Returns:
            "success" | "failed" | "partial"
        """
        # 检查是否有错误步骤
        has_errors = any(step.error for step in steps)
        
        # 检查 execution_result
        if hasattr(state, 'execution_result'):
            exec_result = state.execution_result
            if isinstance(exec_result, dict):
                if exec_result.get('success') is False:
                    return "failed"
                elif exec_result.get('success') is True:
                    return "success"
        
        # 根据错误情况判断
        if has_errors:
            # 如果所有步骤都有错误，则失败
            if all(step.error for step in steps):
                return "failed"
            else:
                return "partial"
        
        return "success"
    
    def _calculate_total_duration(self, steps: List[TrajectoryStep]) -> Optional[float]:
        """计算总耗时"""
        if not steps:
            return None
        
        total = sum(step.duration_ms for step in steps if step.duration_ms)
        return total if total > 0 else None
    
    def _calculate_total_tokens(self, steps: List[TrajectoryStep]) -> Optional[Dict[str, int]]:
        """计算总 token 使用量"""
        total_prompt = 0
        total_completion = 0
        
        for step in steps:
            for llm_call in step.llm_calls:
                if llm_call.token_usage:
                    total_prompt += llm_call.token_usage.get('prompt', 0)
                    total_completion += llm_call.token_usage.get('completion', 0)
        
        if total_prompt > 0 or total_completion > 0:
            return {
                'prompt': total_prompt,
                'completion': total_completion,
                'total': total_prompt + total_completion
            }
        
        return None


# ==================== 便捷函数 ====================

def create_builder() -> TrajectoryBuilder:
    """创建轨迹构建器"""
    return TrajectoryBuilder()
