import os
import asyncio
import gradio as gr

from dataflow_agent.state import MainState
from dataflow_agent.workflow.wf_icongen_refine_loop import create_icongen_refine_loop_graph
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

graph = create_icongen_refine_loop_graph().build()
global_state = MainState(request={"chat_api_url": "http://123.129.219.111:3000/v1"})
last_image = None

os.environ["DF_API_KEY"] = "sk-h6nyfmxUx70YUMBQNwaayrmXda62L7rxvytMNshxACWVzJXe"

os.environ.setdefault("DF_API_KEY", os.getenv("DF_API_KEY", "dummy"))


def _merge_state(state, out):
    if isinstance(out, dict):
        state.request.update(out.get("request", {}) or {})
        state._vars = {**getattr(state, "_vars", {}), **(out.get("_vars") or {})}
        state.agent_results = {**getattr(state, "agent_results", {}), **(out.get("agent_results") or {})}
        return state
    return state


def _get_img(state):
    return (
        (getattr(state, "_vars", {}) or {}).get("final_img")
        or ((getattr(state, "agent_results", {}) or {}).get("bg_removed") or {}).get("path")
        or ((getattr(state, "agent_results", {}) or {}).get("round2_img") or {}).get("path")
        or ((getattr(state, "agent_results", {}) or {}).get("round1_img") or {}).get("path")
    )


async def run_once(keywords=None, style=None, edit_prompt=None, prev_img=None):
    global global_state
    old_api = global_state.request.get("chat_api_url")
    global_state.request = {"chat_api_url": old_api}
    if keywords: global_state.request["keywords"] = keywords
    if style: global_state.request["style"] = style
    if prev_img: global_state.request["prev_image"] = prev_img
    if edit_prompt: global_state.request["edit_prompt"] = edit_prompt

    out = await graph.ainvoke(global_state)
    global_state = _merge_state(global_state, out)
    return _get_img(global_state)


def create_icongen_refine():
    """页面入口函数"""
    with gr.Blocks(title="IconGen Pro — Infinite Refine (No History)") as page:
        gr.Markdown("## 🎨 IconGen Pro — 无限图标迭代")

        with gr.Row():
            keywords = gr.Textbox(label="🎯 关键词首次生成", placeholder="例如：小兔子 / 机器人")
            style = gr.Textbox(label="✨ 风格", placeholder="例如：极简 / 扁平 / 赛博朋克")

        gen_btn = gr.Button("🚀 生成初始图")
        image_output = gr.Image(label="输出图像", type="filepath")

        edit_prompt = gr.Textbox(label="✏️ 编辑提示词", placeholder="例如：add neon glow / change color scheme")
        refine_btn = gr.Button("🎨 继续 refine")

        def ui_generate(keywords, style):
            global last_image
            log.info(f"[IconGen] generate | {keywords=} {style=}")
            img = asyncio.run(run_once(keywords=keywords, style=style))
            last_image = img
            return img

        def ui_refine(edit_prompt):
            global last_image
            if not last_image:
                return None
            log.info(f"[IconGen] refine | {edit_prompt=}")
            img = asyncio.run(run_once(edit_prompt=edit_prompt, prev_img=last_image))
            last_image = img
            return img

        gen_btn.click(ui_generate, [keywords, style], [image_output])
        refine_btn.click(ui_refine, [edit_prompt], [image_output])

    return page