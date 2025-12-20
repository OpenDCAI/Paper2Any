from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from fastapi_app.routers import paper2video
from fastapi_app.routers import paper2any, paper2ppt
from fastapi_app.routers import pdf2ppt
from dataflow_agent.utils import get_project_root


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例。

    这里只做基础框架搭建：
    - CORS 配置
    - 路由挂载
    - 静态文件服务
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
    app.include_router(paper2video.router, prefix="/paper2video", tags=["paper2video"])
    # Paper2Graph / Paper2PPT 假接口，对接前端 /api/*
    app.include_router(paper2any.router, prefix="/api", tags=["paper2any"])
    # Paper2PPT full pipeline JSON 接口
    app.include_router(paper2ppt.router, prefix="/paper2ppt", tags=["paper2ppt"])
    # pdf2ppt_with_sam workflow 接口：仅上传 PDF，返回 PPTX 文件
    app.include_router(pdf2ppt.router, prefix="/api", tags=["pdf2ppt"])

    # 挂载静态文件目录（用于提供生成的 PPTX/SVG/PNG 文件）
    project_root = get_project_root()
    outputs_dir = project_root / "outputs"
    
    # 确保目录存在
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Mounting /outputs to {outputs_dir}")
    
    app.mount(
        "/outputs",
        StaticFiles(directory=str(outputs_dir)),
        name="outputs",
    )

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


# 供 uvicorn 使用：uvicorn fastapi_app.main:app --reload --port 9999
app = create_app()
