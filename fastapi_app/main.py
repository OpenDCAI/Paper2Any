from __future__ import annotations

import os
import tempfile
from pathlib import Path

# 启动时加载 fastapi_app/.env 到环境变量，使 os.getenv("COSYVOICE_KEY") 等能读到
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file)
except ImportError:
    pass


def _configure_runtime_tempdir() -> None:
    project_root = Path(__file__).resolve().parent.parent
    runtime_tmp = Path(
        os.getenv("PAPER2ANY_RUNTIME_TMPDIR", str(project_root / "outputs" / "system" / "tmp"))
    ).expanduser().resolve()
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(runtime_tmp)
    tempfile.tempdir = str(runtime_tmp)


_configure_runtime_tempdir()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi_app.config import settings
from fastapi_app.routers import account
from fastapi_app.routers import paper2video
from fastapi_app.routers import paper2any, paper2citation, paper2figure, paper2ppt, paper2poster
from fastapi_app.routers import pdf2ppt, image2ppt, kb, kb_workflows, kb_embedding, files
from fastapi_app.routers import image2drawio
from fastapi_app.routers import mindmap
from fastapi_app.routers import paper2drawio
from fastapi_app.routers import paper2rebuttal
from fastapi_app.middleware.api_key import APIKeyMiddleware
from dataflow_agent.utils import get_project_root
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


def _parse_cors_allow_origins(raw_value: str) -> list[str]:
    raw = (raw_value or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


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

    allow_origins = _parse_cors_allow_origins(settings.CORS_ALLOW_ORIGINS)
    allow_credentials = bool(allow_origins) and "*" not in allow_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX or None,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API key verification for /api/* routes
    app.add_middleware(APIKeyMiddleware)

    # 路由挂载
    # Paper2Graph / System
    app.include_router(paper2any.router, prefix="/api/v1", tags=["paper2any"])
    app.include_router(paper2figure.router, prefix="/api/v1", tags=["paper2figure"])
    app.include_router(account.router, prefix="/api/v1", tags=["account"])
    # Paper2PPT
    app.include_router(paper2ppt.router, prefix="/api/v1", tags=["paper2ppt"])
    # Paper2Citation
    app.include_router(paper2citation.router, prefix="/api/v1", tags=["paper2citation"])
    # paper2video
    app.include_router(paper2video.router, prefix="/api/v1", tags=["paper2video"])
    # Paper2Poster
    app.include_router(paper2poster.router, prefix="/api/v1", tags=["paper2poster"])
    # PDF2PPT
    app.include_router(pdf2ppt.router, prefix="/api/v1", tags=["pdf2ppt"])
    # Image2PPT
    app.include_router(image2ppt.router, prefix="/api/v1", tags=["image2ppt"])
    # Image2DrawIO
    app.include_router(image2drawio.router, prefix="/api/v1", tags=["image2drawio"])
    # MindMap
    app.include_router(mindmap.router, prefix="/api/v1", tags=["mindmap"])
    # 知识库接口
    app.include_router(kb.router, prefix="/api/v1", tags=["Knowledge Base"])
    app.include_router(kb_workflows.router, prefix="/api/v1", tags=["Knowledge Base Workflows"])
    app.include_router(kb_embedding.router, prefix="/api/v1", tags=["Knowledge Base Embedding"])
    # 文件管理接口
    app.include_router(files.router, prefix="/api/v1", tags=["Files"])
    # Paper2Drawio
    app.include_router(paper2drawio.router, prefix="/api/v1", tags=["paper2drawio"])
    # Paper2Rebuttal
    app.include_router(paper2rebuttal.router, prefix="/api/v1", tags=["paper2rebuttal"])

    # 挂载静态文件目录（用于提供生成的 PPTX/SVG/PNG 文件）
    project_root = get_project_root()
    outputs_dir = project_root / "outputs"
    
    # 确保目录存在
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"[INFO] Mounting /outputs to {outputs_dir}")
    
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
