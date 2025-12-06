from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi_app.routers import operator_write, pipeline_rec, workflows, paper2video


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例。

    这里只做基础框架搭建：
    - CORS 配置
    - 路由挂载
    """
    app = FastAPI(
        title="DataFlow Agent FastAPI Backend",
        version="0.1.0",
        description="HTTP API wrapper for dataflow_agent.workflow.* pipelines",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由挂载
    app.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
    app.include_router(operator_write.router, prefix="/operator", tags=["operator_write"])
    app.include_router(pipeline_rec.router, prefix="/pipeline", tags=["pipeline_recommend"])
    app.include_router(paper2video.router, prefix="/paper2video", tags=["paper2video"])
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


# 供 uvicorn 使用：uvicorn fastapi_app.main:app --reload
app = create_app()
