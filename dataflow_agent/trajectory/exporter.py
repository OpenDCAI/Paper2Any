"""
轨迹导出器 - 将 TRJ 导出为不同格式

支持：
1. JSON 格式导出（单个/批量）
2. JSONL 格式导出（用于训练数据）
3. SFT/DPO 格式转换
4. 数据库存储（可选）
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from dataflow_agent.trajectory.models import Trajectory
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)

PROJDIR = get_project_root()
DEFAULT_OUTPUT_DIR = PROJDIR / "outputs" / "trajectories"


class TrajectoryExporter:
    """
    轨迹导出器
    
    提供多种导出格式和存储方式
    """
    
    def __init__(self, output_dir: Union[str, Path] = None):
        """
        Args:
            output_dir: 输出目录，默认为 outputs/trajectories
        """
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[TrajectoryExporter] 输出目录: {self.output_dir}")
    
    def export_to_json(self,
                      trajectory: Trajectory,
                      filepath: str = None,
                      pretty: bool = True) -> str:
        """
        导出为 JSON 文件
        
        Args:
            trajectory: 轨迹对象
            filepath: 文件路径，如果为 None 则自动生成
            pretty: 是否格式化输出
            
        Returns:
            保存的文件路径
        """
        if filepath is None:
            filename = f"{trajectory.trace_id}.json"
            filepath = self.output_dir / filename
        else:
            filepath = Path(filepath)
        
        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        data = trajectory.to_dict()
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        log.info(f"[TrajectoryExporter] 已导出 JSON: {filepath}")
        return str(filepath)
    
    def export_to_jsonl(self,
                       trajectories: List[Trajectory],
                       filepath: str = None,
                       mode: str = "raw") -> str:
        """
        批量导出为 JSONL 文件（每行一个 JSON 对象）
        
        Args:
            trajectories: 轨迹列表
            filepath: 文件路径
            mode: 导出模式
                - "raw": 完整的 TRJ 数据
                - "sft": SFT 训练格式
                - "dpo": DPO 训练格式
                
        Returns:
            保存的文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectories_{mode}_{timestamp}.jsonl"
            filepath = self.output_dir / filename
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for trj in trajectories:
                if mode == "raw":
                    data = trj.to_dict()
                elif mode == "sft":
                    data = {
                        "trace_id": trj.trace_id,
                        "messages": trj.to_sft_format(),
                        "metadata": {
                            "workflow": trj.workflow_name,
                            "status": trj.status,
                        }
                    }
                elif mode == "dpo":
                    data = trj.to_dpo_format()
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        log.info(f"[TrajectoryExporter] 已导出 JSONL ({mode}): {filepath}, "
                f"共 {len(trajectories)} 条")
        return str(filepath)
    
    def export_sft_dataset(self,
                          trajectories: List[Trajectory],
                          filepath: str = None,
                          filter_success: bool = True) -> str:
        """
        导出 SFT 训练数据集
        
        Args:
            trajectories: 轨迹列表
            filepath: 文件路径
            filter_success: 是否只保留成功的轨迹
            
        Returns:
            保存的文件路径
        """
        # 过滤
        if filter_success:
            trajectories = [t for t in trajectories if t.status == "success"]
            log.info(f"[TrajectoryExporter] 过滤后保留 {len(trajectories)} 条成功轨迹")
        
        return self.export_to_jsonl(trajectories, filepath, mode="sft")
    
    def export_dpo_dataset(self,
                          chosen_trajectories: List[Trajectory],
                          rejected_trajectories: List[Trajectory],
                          filepath: str = None) -> str:
        """
        导出 DPO 训练数据集（成对数据）
        
        Args:
            chosen_trajectories: 正例轨迹（成功的）
            rejected_trajectories: 负例轨迹（失败的）
            filepath: 文件路径
            
        Returns:
            保存的文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dpo_pairs_{timestamp}.jsonl"
            filepath = self.output_dir / filename
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 按 prompt 分组
        prompt_groups = {}
        
        for trj in chosen_trajectories:
            prompt = trj.inputs.get("query", "")
            if prompt not in prompt_groups:
                prompt_groups[prompt] = {"chosen": [], "rejected": []}
            prompt_groups[prompt]["chosen"].append(trj)
        
        for trj in rejected_trajectories:
            prompt = trj.inputs.get("query", "")
            if prompt not in prompt_groups:
                prompt_groups[prompt] = {"chosen": [], "rejected": []}
            prompt_groups[prompt]["rejected"].append(trj)
        
        # 生成成对数据
        pairs_count = 0
        with open(filepath, 'w', encoding='utf-8') as f:
            for prompt, group in prompt_groups.items():
                chosen_list = group.get("chosen", [])
                rejected_list = group.get("rejected", [])
                
                # 每个 chosen 和每个 rejected 配对
                for chosen in chosen_list:
                    for rejected in rejected_list:
                        pair = {
                            "prompt": prompt,
                            "chosen": chosen.to_sft_format(),
                            "rejected": rejected.to_sft_format(),
                            "metadata": {
                                "chosen_trace_id": chosen.trace_id,
                                "rejected_trace_id": rejected.trace_id,
                                "chosen_score": chosen.feedback.score if chosen.feedback else None,
                                "rejected_score": rejected.feedback.score if rejected.feedback else None,
                            }
                        }
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                        pairs_count += 1
        
        log.info(f"[TrajectoryExporter] 已导出 DPO 数据集: {filepath}, "
                f"共 {pairs_count} 对")
        return str(filepath)
    
    def export_statistics(self,
                         trajectories: List[Trajectory],
                         filepath: str = None) -> str:
        """
        导出统计信息
        
        Args:
            trajectories: 轨迹列表
            filepath: 文件路径
            
        Returns:
            保存的文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistics_{timestamp}.json"
            filepath = self.output_dir / filename
        else:
            filepath = Path(filepath)
        
        # 计算统计信息
        stats = {
            "total_trajectories": len(trajectories),
            "by_status": {},
            "by_mode": {},
            "by_workflow": {},
            "total_steps": 0,
            "total_llm_calls": 0,
            "total_tool_calls": 0,
            "total_duration_ms": 0,
            "avg_steps_per_trajectory": 0,
            "success_rate": 0,
        }
        
        for trj in trajectories:
            # 按状态统计
            status = trj.status
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            # 按模式统计
            mode = trj.mode
            stats["by_mode"][mode] = stats["by_mode"].get(mode, 0) + 1
            
            # 按 workflow 统计
            workflow = trj.workflow_name
            stats["by_workflow"][workflow] = stats["by_workflow"].get(workflow, 0) + 1
            
            # 累计统计
            stats["total_steps"] += len(trj.steps)
            stats["total_llm_calls"] += trj.total_llm_calls
            stats["total_tool_calls"] += trj.total_tool_calls
            if trj.total_duration_ms:
                stats["total_duration_ms"] += trj.total_duration_ms
        
        # 计算平均值
        if len(trajectories) > 0:
            stats["avg_steps_per_trajectory"] = stats["total_steps"] / len(trajectories)
            success_count = stats["by_status"].get("success", 0)
            stats["success_rate"] = success_count / len(trajectories)
        
        # 保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        log.info(f"[TrajectoryExporter] 已导出统计信息: {filepath}")
        return str(filepath)
    
    def load_from_json(self, filepath: str) -> Trajectory:
        """
        从 JSON 文件加载轨迹
        
        Args:
            filepath: 文件路径
            
        Returns:
            Trajectory 对象
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 这里需要实现从字典重建 Trajectory 的逻辑
        # 简化版本：直接返回字典
        log.info(f"[TrajectoryExporter] 已加载轨迹: {filepath}")
        return data
    
    def load_from_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """
        从 JSONL 文件加载轨迹列表
        
        Args:
            filepath: 文件路径
            
        Returns:
            轨迹列表
        """
        trajectories = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    trajectories.append(data)
        
        log.info(f"[TrajectoryExporter] 已加载 {len(trajectories)} 条轨迹: {filepath}")
        return trajectories


# ==================== 便捷函数 ====================

def create_exporter(output_dir: str = None) -> TrajectoryExporter:
    """创建轨迹导出器"""
    return TrajectoryExporter(output_dir)


def quick_export(trajectory: Trajectory, 
                format: str = "json",
                output_dir: str = None) -> str:
    """
    快速导出单个轨迹
    
    Args:
        trajectory: 轨迹对象
        format: 导出格式（json/jsonl）
        output_dir: 输出目录
        
    Returns:
        保存的文件路径
    """
    exporter = create_exporter(output_dir)
    
    if format == "json":
        return exporter.export_to_json(trajectory)
    elif format == "jsonl":
        return exporter.export_to_jsonl([trajectory])
    else:
        raise ValueError(f"Unknown format: {format}")
