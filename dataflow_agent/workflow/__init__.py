# dataflow_agent/workflow/__init__.py

import importlib
from pathlib import Path

from .paper2video_subprocess import run_paper2video_via_subprocess
from .registry import RuntimeRegistry

_pkg_path = Path(__file__).resolve().parent
_imported_workflow_modules: set[str] = set()


def _workflow_module_names() -> list[str]:
    return sorted(
        f"{__name__}.{py.stem}"
        for py in _pkg_path.glob("wf_*.py")
    )


def _import_workflow_modules_until(name: str | None = None) -> None:
    """
    Lazy-load workflow definition modules.

    Importing every workflow during FastAPI startup pulls in heavyweight
    dependencies such as torchvision/transformers and makes the backend look
    dead for a long time. Only import enough modules to satisfy the requested
    workflow registration, and fall back to importing everything only when the
    caller explicitly asks for the full list.
    """
    if name is not None and name in RuntimeRegistry._workflows:
        return

    for mod_name in _workflow_module_names():
        if mod_name in _imported_workflow_modules:
            continue
        importlib.import_module(mod_name)
        _imported_workflow_modules.add(mod_name)
        if name is not None and name in RuntimeRegistry._workflows:
            return

# ---- 2. 工作流的统一接口 ---------------------------------------------
def get_workflow(name: str):
    """
    根据工作流名称获取 create_pipeline_graph 工厂方法。

    Args:
        name (str): 工作流名称（注册名）

    Returns:
        Callable: 用于构建该工作流图的工厂函数
    """
    _import_workflow_modules_until(name)
    return RuntimeRegistry.get(name)


async def run_workflow(name: str, state):
    if name == "paper2video":
        return await run_paper2video_via_subprocess(name, state)

    factory = get_workflow(name)
    graph_builder = factory()
    graph = graph_builder.build()
    return await graph.ainvoke(state)


# ---- 3. 工作流注册信息公开接口 -------------------------------------------
# 提供所有已注册工作流的列表，便于外部查询与 introspection
def list_workflows():
    _import_workflow_modules_until()
    return RuntimeRegistry.all()
