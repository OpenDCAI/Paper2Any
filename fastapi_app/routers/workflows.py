from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from dataflow_agent.workflow import get_workflow, list_workflows

router = APIRouter()


@router.get(
    "/",
    summary="列出所有已注册的工作流名称",
    description=(
        "从 dataflow_agent.workflow.registry 中读取当前已注册的所有工作流。"
        "返回一个名称列表，可用于前端动态展示或后续选择具体工作流。"
    ),
)
def list_all_workflows() -> Dict[str, List[str]]:
    wf_dict = list_workflows()
    return {"workflows": sorted(wf_dict.keys())}


@router.get(
    "/{name}",
    summary="查看指定工作流的基本信息",
    description=(
        "根据工作流名称加载其 factory，并尝试实例化 GenericGraphBuilder，"
        "返回入口节点、state_model 名称以及节点列表等基础元信息。"
        "仅做元信息查询，不会真正执行工作流。"
    ),
)
def get_workflow_info(name: str) -> Dict[str, Any]:
    try:
        factory = get_workflow(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    info: Dict[str, Any] = {
        "name": name,
        "factory_repr": repr(factory),
    }

    # 尝试获取更多 builder 信息（如果 factory 符合 GenericGraphBuilder 约定）
    try:
        builder = factory()
        entry_point = getattr(builder, "entry_point", None)
        state_model = getattr(builder, "state_model", None)
        nodes = getattr(builder, "nodes", None)

        if entry_point is not None:
            info["entry_point"] = entry_point
        if state_model is not None:
            info["state_model"] = getattr(state_model, "__name__", str(state_model))
        if isinstance(nodes, dict):
            info["nodes"] = sorted(nodes.keys())
    except Exception:
        # 仅做元信息探测，任何异常都忽略，避免影响基础可用性
        pass

    return info
