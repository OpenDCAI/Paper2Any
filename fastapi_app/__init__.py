from __future__ import annotations

"""
FastAPI backend for DataFlow Agent.

该包提供一组 HTTP API，用于以服务化方式调用 dataflow_agent.workflow.* 中的各类工作流。
典型使用方式：

    # 从项目根目录启动（推荐）
    cd /path/to/Paper2Any
    uvicorn fastapi_app.main:app --reload --port 8051

路由划分约定：
- /workflows/*   ：工作流发现与（后续）通用运行接口
- /operator/*    ：算子编写相关接口（基于 wf_pipeline_write）
- /pipeline/*    ：流水线推荐/导出相关接口（基于 wf_pipeline_recommend_* 等）
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path even when launched from fastapi_app/
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .main import app, create_app  # noqa: F401
