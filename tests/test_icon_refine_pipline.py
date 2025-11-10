import os, asyncio
from dataflow_agent.state import MainState
from dataflow_agent.workflow.wf_icongen_refine_loop import create_icongen_refine_loop_graph

os.environ["DF_API_KEY"] = os.getenv("DF_API_KEY") or "dummy"

async def run_round(graph, state, keywords=None, edit_prompt=None, prev_image=None):
    # 保留 API 设置
    old_api = state.request.get("chat_api_url")
    state.request = {"chat_api_url": old_api}

    # 首轮
    if keywords:
        state.request["keywords"] = keywords
        state.request["style"] = "neon glow"

    # 编辑轮
    if edit_prompt:
        state.request["edit_prompt"] = edit_prompt
        state.request["prev_image"] = prev_image

    print(f"call workflow, request={state.request}")

    # 运行：返回 dict
    out = await graph.ainvoke(state)

    # 把返回的 dict 覆盖回原 state
    if isinstance(out, dict):
        if not hasattr(state, "_vars"):
            state._vars = {}
        if not hasattr(state, "agent_results"):
            state.agent_results = {}

        # 主要存储区域
        request  = out.get("request", {})
        _vars    = out.get("_vars", {})
        results  = out.get("agent_results", {})

        state.request.update(request)
        state._vars.update(_vars)
        state.agent_results.update(results)

    # 取图片
    img = (
        state._vars.get("final_img")
        or (state.agent_results.get("bg_removed") or {}).get("path")
        or (state.agent_results.get("round2_img") or {}).get("path")
        or (state.agent_results.get("round1_img") or {}).get("path")
    )

    print(f"result image: {img}")
    return state, img

async def main():
    graph = create_icongen_refine_loop_graph().build()
    state = MainState(request={"chat_api_url": "http://123.129.219.111:3000/v1"})

    print("\nRound 1 → base")
    state, img1 = await run_round(graph, state, keywords="cyberpunk cat")

    print("\nRound 2 → gears")
    state, img2 = await run_round(graph, state, edit_prompt="add neon gears", prev_image=img1)

    print("\nRound 3 → whiskers")
    state, img3 = await run_round(graph, state, edit_prompt="add circuit whiskers", prev_image=img2)

    print("\nRound 4 → glasses")
    state, img4 = await run_round(graph, state, edit_prompt="add hologram glasses", prev_image=img3)

    print("\nFINAL:", img4)


if __name__ == "__main__":
    asyncio.run(main())
