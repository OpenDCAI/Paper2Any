from __future__ import annotations

"""
Router package for FastAPI backend.

- operator_write : 基于 wf_pipeline_write 的算子编写相关接口
- pipeline_rec   : 基于 wf_pipeline_recommend_extract_json 的流水线推荐相关接口
- workflows      : 工作流发现/元信息接口（基于 dataflow_agent.workflow.registry）
"""

# 让 `from fastapi_app.routers import operator_write, pipeline_rec, workflows` 可用
from . import operator_write, pipeline_rec, workflows

__all__ = ["operator_write", "pipeline_rec", "workflows"]
