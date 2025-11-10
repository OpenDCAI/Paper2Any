from __future__ import annotations
import asyncio
import pytest

# ------------ 依赖 -------------
from dataflow_agent.state import IconGenRequest, IconGenState  
from dataflow_agent.workflow import run_workflow               
# --------------------------------


# ============ 核心异步流程 ============
async def run_icon_pipeline() -> IconGenState:
    # 1) 构造请求
    req = IconGenRequest(
        language="en",
        keywords="cat, coffee, rocket",
        style="flat, pastel, outline",
    )

    # 2) 初始化状态
    state = IconGenState(request=req, messages=[])

    # 3) 通过注册中心执行工作流
    final_state: IconGenState = await run_workflow("icongen", state)
    return final_state


# ============ pytest 入口 ============
@pytest.mark.asyncio
async def test_icon_pipeline():
    final_state = await run_icon_pipeline()

    # -- 简单断言：至少生成了一条 prompt --
    assert final_state.icon_prompts, "icon_prompts 为空！"

    # -- 调试输出，可按需保留 --
    print("\n=== icon_prompts ===")
    for p in final_state.icon_prompts:
        print(p)

    if final_state.svg_results:
        print("\n=== svg_results (前 120 字符预览) ===")
        for svg in final_state.svg_results:
            print(svg[:120] + " ...")


# ============ 直接 python 执行 ============
if __name__ == "__main__":
    asyncio.run(run_icon_pipeline())
