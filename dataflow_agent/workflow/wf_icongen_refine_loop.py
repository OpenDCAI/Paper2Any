from __future__ import annotations
import os
import time
from typing import Optional, Dict, Any

from dataflow_agent.state import MainState
from dataflow_agent.workflow.registry import register
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger

from dataflow_agent.toolkits.imtool.req_img import generate_or_edit_and_save_image_async
from dataflow_agent.toolkits.imtool.bg_tool import local_tool_for_bg_remove
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)


# ---------- helpers ----------
def _ensure(state: MainState) -> MainState:
    """
    保证 state 具备必要结构；_vars 可能不存在（LangGraph/DF的State），统一在此创建。
    """
    if not getattr(state, "agent_results", None):
        state.agent_results = {}
    if getattr(state, "_vars", None) is None:
        setattr(state, "_vars", {})
    # 给测试/调用端使用的最终图片路径(每轮都会刷新)
    if "final_img" not in state._vars:
        state._vars["final_img"] = None
    return state


def _get_final_img_from_results(res: Dict[str, Any]) -> Optional[str]:
    if "round2_img" in res:
        return res["round2_img"]["path"]
    if "round1_img" in res:
        return res["round1_img"]["path"]
    return None


def _ts_name(stem: str, ext: str = ".png") -> str:
    return f"./{stem}_{int(time.time()*1000)%10_000_000}{ext}"


# ---------- workflow ----------
@register("icongen_refine_loop")
def create_icongen_refine_loop_graph() -> GenericGraphBuilder:
    """
    一次完整执行 = （可能的）首轮生图 -> （可选）编辑生图 -> 抠图
    - 首轮：request 需要 keywords/style
    - 编辑：request 需要 prev_image + edit_prompt
    - 多轮通过“多次调用 graph.ainvoke(state)”由上层驱动
    """
    builder = GenericGraphBuilder(
        state_model=MainState,
        entry_point="build_prompt"
    )

    # ---- 节点：build_prompt（仅首轮有意义） ----
    async def build_prompt(state: MainState):
        state = _ensure(state)
        req = state.request or {}

        # 有 prev_image 说明是编辑轮，跳过文生图 prompt 构造
        if req.get("prev_image"):
            log.info("[build_prompt] editing round, skip prompt build")
            return state

        kw = (req.get("keywords") or "").strip()
        style = (req.get("style") or "").strip()

        if not kw:
            # 没关键词，后续会走编辑轮逻辑；这里留个提示并跳过
            log.warning("[build_prompt] no keywords found for first round, will rely on edit path if provided.")
            return state

        prompt = f"flat minimal vector icon, {kw}, style {style}, center, no background, no text"
        # 关键：prompt 写入持久区，避免下一次 ainvoke 丢失
        state.agent_results["prompt"] = prompt
        # 同时也写入 _vars 方便本次编排链路内使用
        state._vars["prompt"] = prompt
        log.info(f"[build_prompt] prompt = {prompt}")
        return state

    # ---- 节点：gen（文生图 或 图生图） ----
    async def gen(state: MainState):
        state = _ensure(state)
        req = state.request or {}

        api_url = req.get("chat_api_url")
        api_key = os.getenv("DF_API_KEY")

        prev_image = req.get("prev_image")
        edit_prompt = (req.get("edit_prompt") or "").strip()

        # 编辑轮：必须要 prev_image + edit_prompt
        if prev_image and edit_prompt:
            out_path = _ts_name("icon_round_edit")
            log.info(f"[gen] edit image: in={prev_image}, prompt={edit_prompt}")
            b64 = await generate_or_edit_and_save_image_async(
                prompt=edit_prompt,
                image_path=prev_image,
                save_path=out_path,
                api_url=api_url,
                api_key=api_key,
                model="gemini-2.5-flash-image-preview",
                use_edit=True,
            )
            state.agent_results["round2_img"] = {"path": out_path, "b64": b64}
            state._vars["final_img"] = out_path
            return state

        # 首轮：使用 prompt 文生图
        prompt = state.agent_results.get("prompt") or state._vars.get("prompt")
        if not prompt:
            # 明确提示首轮缺 prompt 的情况
            log.warning("[gen] prompt missing, skip first-gen (likely editing-only round without prev_image).")
            return state

        out_path = _ts_name("icon_round_base")
        log.info(f"[gen] first image: prompt={prompt}")
        b64 = await generate_or_edit_and_save_image_async(
            prompt=prompt,
            save_path=out_path,
            api_url=api_url,
            api_key=api_key,
            model=state.request.model,
        )
        state.agent_results["round1_img"] = {"path": out_path, "b64": b64}
        state._vars["final_img"] = out_path
        return state

    # ---- 节点：bg（抠图，可选，如果上一阶段无图就跳过） ----
    async def bg(state: MainState):
        state = _ensure(state)
        # 以 round2 > round1 的优先级选择输入图
        src = _get_final_img_from_results(state.agent_results)
        if not src:
            log.warning("[bg] no image produced in gen(), skip background removal.")
            return state

        try:
            out_path = local_tool_for_bg_remove({
                "image_path": src,
                "model_path": f"{get_project_root()}/dataflow_agent/toolkits/imtool/models/RMBG-2.0/onnx/model.onnx",         # 允许 bg_tool 内部走默认
                "output_dir": "./"
            })
        except Exception as e:
            log.error(f"[bg] background removal failed: {e}")
            return state

        # bg_tool 可能返回 None，保护一下
        if not out_path:
            log.warning("[bg] bg tool returned None, keep original as final.")
            return state

        state.agent_results["bg_removed"] = {"path": out_path}
        state._vars["final_img"] = out_path
        log.info(f"[bg] background removed -> {out_path}")
        return state

    async def END(state: MainState):
        return _ensure(state)

    # 装配
    builder.add_nodes({
        "build_prompt": build_prompt,
        "gen": gen,
        "bg": bg,
        "END": END,
    })

    builder.add_edges([
        ("build_prompt", "gen"),
        ("gen", "bg"),
        ("bg", "END"),
    ])

    return builder
