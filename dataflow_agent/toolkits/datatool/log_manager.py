# dataflow_agent/toolkits/datatool/log_manager.py

from __future__ import annotations
import os
import json
import re
from typing import Any, Dict
from datetime import datetime
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


def log_agent_input_output(agent_name: str, inputs: Dict[str, Any], outputs: Any = None, logger: Any = None):
    """
    记录Agent的输入和输出
    
    Args:
        agent_name: Agent名称
        inputs: 输入参数字典
        outputs: 输出结果
        logger: LogManager实例（可选）
    """
    # 记录到标准日志
    log.info(f"[Agent Input] {agent_name}: {json.dumps(inputs, indent=2, ensure_ascii=False, default=str)}")
    if outputs is not None:
        # 对于大型输出，只记录摘要
        if isinstance(outputs, (dict, list)) and len(str(outputs)) > 1000:
            output_summary = f"<{type(outputs).__name__} with {len(outputs) if isinstance(outputs, (list, dict)) else 'N/A'} items>"
            log.info(f"[Agent Output] {agent_name}: {output_summary}")
        else:
            log.info(f"[Agent Output] {agent_name}: {json.dumps(outputs, indent=2, ensure_ascii=False, default=str)}")
    
    # 如果提供了LogManager，也记录到文件
    if logger:
        logger.log_data(f"{agent_name}_input", inputs, is_json=True)
        if outputs is not None:
            logger.log_data(f"{agent_name}_output", outputs, is_json=True)


class LogManager:
    """负责为每次运行创建日志目录并保存每一步的数据。"""
    
    def __init__(self, base_dir="logs"):
        self.run_dir = ""
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir

    def new_run(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        log.info(f"日志将保存在: {self.run_dir}")

    def log_data(self, step_name: str, data: Any, is_json: bool = False):
        if not self.run_dir:
            log.warning("LogManager尚未初始化，无法记录日志。")
            return
        safe_step_name = re.sub(r'[\\/*?:"<>|]', "", step_name)
        safe_step_name = re.sub(r'\s+', '_', safe_step_name).strip("_") or "log"
        extension = ".json" if is_json else ".txt"
        filename = os.path.join(self.run_dir, f"{safe_step_name}{extension}")
        content = json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            log.error(f"写入日志文件 {filename} 时出错: {e}")


