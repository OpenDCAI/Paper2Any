"""
Minimal FastAPI app for testing auth and rate limiting.

This app excludes workflow endpoints to avoid heavy ML dependencies.
Use this for lightweight integration testing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create minimal app
app = FastAPI(
    title="DataFlow-Agent API (Lite)",
    description="Minimal API for testing auth and rate limiting",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "lite"}


# Import only lightweight routers (bypass routers/__init__.py to avoid heavy imports)
# These imports don't pull in workflow_adapters or dataflow_agent
import importlib.util
import sys
from pathlib import Path

def _import_router_directly(name: str):
    """Import a router module directly without going through the package."""
    router_path = Path(__file__).parent / "routers" / f"{name}.py"
    if not router_path.exists():
        raise ImportError(f"Router not found: {router_path}")

    spec = importlib.util.spec_from_file_location(f"fastapi_app.routers.{name}", router_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"fastapi_app.routers.{name}"] = module
    spec.loader.exec_module(module)
    return module

# Import lightweight routers
user_module = _import_router_directly("user")
auth_module = _import_router_directly("auth")

app.include_router(user_module.router, prefix="/api", tags=["user"])
app.include_router(auth_module.router, prefix="/api", tags=["auth"])  # auth router already has /auth prefix


def create_app():
    """Factory function for creating the lite app."""
    return app
