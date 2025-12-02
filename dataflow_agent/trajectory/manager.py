"""
轨迹管理器 - 统一的轨迹管理入口

提供简单易用的 API 来：
1. 开始/停止轨迹记录
2. 自动构建和导出轨迹
3. 批量管理轨迹
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from dataflow_agent.trajectory.models import Trajectory
from dataflow_agent.trajectory.collector import TrajectoryCollector
from dataflow_agent.trajectory.builder import TrajectoryBuilder
from dataflow_agent.trajectory.exporter import TrajectoryExporter
from dataflow_agent.state import MainState
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


class TrajectoryManager:
    """
    轨迹管理器 - 统一入口
    
    使用示例：
    ```python
    # 1. 创建管理器
    trj_manager = TrajectoryManager()
    
    # 2. 开始记录
    trj_manager.start_recording(inputs={"query": "..."})
    
    # 3. 在 workflow 执行过程中，collector 会自动记录
    # （需要在 workflow 中集成 collector 的 hook）
    
    # 4. 停止记录并生成轨迹
    trajectory = trj_manager.stop_recording(
        state=final_state,
        workflow_name="my_workflow"
    )
    
    # 5. 导出
    filepath = trj_manager.export(trajectory, format="json")
    ```
    """
    
    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: 导出目录
        """
        self.collector = TrajectoryCollector()
        self.builder = TrajectoryBuilder()
        self.exporter = TrajectoryExporter(output_dir)
        
        self.is_recording = False
        self.current_trajectory: Optional[Trajectory] = None
        
        log.info("[TrajectoryManager] 初始化完成")
    
    def start_recording(self, 
                       inputs: Dict[str, Any] = None,
                       metadata: Dict[str, Any] = None):
        """
        开始记录轨迹
        
        Args:
            inputs: 初始输入数据
            metadata: 额外元数据
        """
        self.collector.start(inputs=inputs, metadata=metadata)
        self.is_recording = True
        log.info("[TrajectoryManager] 开始记录轨迹")
    
    def stop_recording(self,
                      state: MainState,
                      workflow_name: str,
                      user_id: str = None,
                      session_id: str = None) -> Trajectory:
        """
        停止记录并生成轨迹
        
        Args:
            state: Workflow 执行后的最终状态
            workflow_name: Workflow 名称
            user_id: 用户 ID
            session_id: 会话 ID
            
        Returns:
            生成的 Trajectory 对象
        """
        if not self.is_recording:
            log.warning("[TrajectoryManager] 未在记录状态，无法停止")
            return None
        
        # 构建轨迹
        trajectory = self.builder.build_from_state(
            state=state,
            collector=self.collector,
            workflow_name=workflow_name,
            user_id=user_id,
            session_id=session_id
        )
        
        self.is_recording = False
        self.current_trajectory = trajectory
        
        log.info(f"[TrajectoryManager] 轨迹记录完成: {trajectory.trace_id}")
        return trajectory
    
    def export(self,
              trajectory: Trajectory = None,
              format: str = "json",
              filepath: str = None,
              **kwargs) -> str:
        """
        导出轨迹
        
        Args:
            trajectory: 要导出的轨迹，如果为 None 则使用当前轨迹
            format: 导出格式（json/jsonl/sft/dpo）
            filepath: 文件路径
            **kwargs: 其他参数
            
        Returns:
            保存的文件路径
        """
        if trajectory is None:
            trajectory = self.current_trajectory
        
        if trajectory is None:
            log.error("[TrajectoryManager] 没有可导出的轨迹")
            return None
        
        if format == "json":
            return self.exporter.export_to_json(trajectory, filepath, **kwargs)
        elif format == "jsonl":
            return self.exporter.export_to_jsonl([trajectory], filepath, **kwargs)
        elif format == "sft":
            return self.exporter.export_to_jsonl([trajectory], filepath, mode="sft")
        elif format == "dpo":
            return self.exporter.export_to_jsonl([trajectory], filepath, mode="dpo")
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def export_batch(self,
                    trajectories: List[Trajectory],
                    format: str = "jsonl",
                    filepath: str = None,
                    **kwargs) -> str:
        """
        批量导出轨迹
        
        Args:
            trajectories: 轨迹列表
            format: 导出格式
            filepath: 文件路径
            **kwargs: 其他参数
            
        Returns:
            保存的文件路径
        """
        if format == "jsonl":
            return self.exporter.export_to_jsonl(trajectories, filepath, **kwargs)
        elif format == "sft":
            return self.exporter.export_sft_dataset(trajectories, filepath, **kwargs)
        else:
            raise ValueError(f"Batch export not supported for format: {format}")
    
    def get_collector(self) -> TrajectoryCollector:
        """获取收集器实例（用于手动集成）"""
        return self.collector
    
    def get_current_trajectory(self) -> Optional[Trajectory]:
        """获取当前轨迹"""
        return self.current_trajectory
    
    def add_feedback(self,
                    trajectory: Trajectory = None,
                    score: int = None,
                    comment: str = None,
                    edited_response: str = None,
                    labels: List[str] = None):
        """
        添加用户反馈
        
        Args:
            trajectory: 轨迹对象，如果为 None 则使用当前轨迹
            score: 评分 1-5
            comment: 评论
            edited_response: 用户修改后的回答
            labels: 标签列表
        """
        if trajectory is None:
            trajectory = self.current_trajectory
        
        if trajectory is None:
            log.error("[TrajectoryManager] 没有可添加反馈的轨迹")
            return
        
        trajectory.set_feedback(
            score=score,
            comment=comment,
            edited_response=edited_response,
            labels=labels
        )
        
        log.info(f"[TrajectoryManager] 已添加反馈到轨迹: {trajectory.trace_id}")


# ==================== 全局单例 ====================

_global_manager: Optional[TrajectoryManager] = None


def get_trajectory_manager(output_dir: str = None) -> TrajectoryManager:
    """
    获取全局轨迹管理器（单例模式）
    
    Args:
        output_dir: 输出目录
        
    Returns:
        TrajectoryManager 实例
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = TrajectoryManager(output_dir)
    
    return _global_manager


def reset_trajectory_manager():
    """重置全局轨迹管理器"""
    global _global_manager
    _global_manager = None


# ==================== 便捷函数 ====================

def quick_record(workflow_func,
                workflow_name: str,
                inputs: Dict[str, Any] = None,
                export_format: str = "json",
                **workflow_kwargs):
    """
    快速记录 workflow 执行轨迹的装饰器/函数
    
    使用示例：
    ```python
    # 作为装饰器
    @quick_record(workflow_name="my_workflow")
    async def my_workflow(state):
        # workflow 逻辑
        return final_state
    
    # 或作为函数
    final_state = await quick_record(
        my_workflow,
        workflow_name="my_workflow",
        inputs={"query": "..."},
        state=initial_state
    )
    ```
    """
    import asyncio
    from functools import wraps
    
    # 如果是装饰器用法
    if callable(workflow_func):
        @wraps(workflow_func)
        async def wrapper(*args, **kwargs):
            manager = get_trajectory_manager()
            
            # 开始记录
            manager.start_recording(inputs=inputs)
            
            try:
                # 执行 workflow
                if asyncio.iscoroutinefunction(workflow_func):
                    result = await workflow_func(*args, **kwargs)
                else:
                    result = workflow_func(*args, **kwargs)
                
                # 停止记录
                trajectory = manager.stop_recording(
                    state=result,
                    workflow_name=workflow_name
                )
                
                # 导出
                filepath = manager.export(trajectory, format=export_format)
                log.info(f"[quick_record] 轨迹已导出: {filepath}")
                
                return result
                
            except Exception as e:
                log.exception(f"[quick_record] Workflow 执行失败: {e}")
                raise
        
        return wrapper
    
    # 如果是函数调用用法
    else:
        raise ValueError("quick_record 应该作为装饰器使用")
