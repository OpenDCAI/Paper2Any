from __future__ import annotations

"""
FastAPI backend for DataFlow Agent.

该包提供一组 HTTP API，用于以服务化方式调用 dataflow_agent.workflow.* 中的各类工作流。
典型使用方式：

    uvicorn fastapi_app.main:app --reload

路由划分约定：
- /workflows/*   ：工作流发现与（后续）通用运行接口
- /operator/*    ：算子编写相关接口（基于 wf_pipeline_write）
- /pipeline/*    ：流水线推荐/导出相关接口（基于 wf_pipeline_recommend_* 等）
"""

# Lazy import to avoid loading heavy dependencies at module level
# Use: from fastapi_app import app  (will load main.py on first access)
def __getattr__(name):
    if name in ("app", "create_app"):
        from .main import app, create_app
        return app if name == "app" else create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

