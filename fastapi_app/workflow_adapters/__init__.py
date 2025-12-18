from __future__ import annotations

"""
Workflow adapters package.

拆分自原来的 `fastapi_app.workflow_adapters` 大文件，用于封装
dataflow_agent.workflow.* 的调用逻辑。

对外暂时保持与旧版相同的导出接口，方便逐步迁移：
- run_operator_write_pipeline_api
- run_paper_to_video_api
- run_paper2figure_wf_api
"""

from .paper2video import run_paper_to_video_api
from .paper2figure import run_paper2figure_wf_api

__all__ = [
    "run_paper_to_video_api",
    "run_paper2figure_wf_api",
]
