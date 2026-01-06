from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Tuple

import gradio as gr

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
from gradio_app.utils.space_paths import new_run_dir

log = get_logger(__name__)


def _list_generated_pages(run_dir: str) -> list[str]:
    if not run_dir:
        return []
    img_dir = Path(run_dir) / "ppt_pages"
    if not img_dir.exists():
        return []
    return [str(p.resolve()) for p in sorted(img_dir.glob("page_*.png"))]


def _pdf_path(run_dir: str) -> str:
    if not run_dir:
        return ""
    p = Path(run_dir) / "paper2ppt.pdf"
    return str(p.resolve()) if p.exists() else ""


def _safe_json_loads(s: str) -> list[dict]:
    try:
        obj = json.loads(s or "[]")
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"pagecontent JSON 解析失败: {e}") from e
    if not isinstance(obj, list):
        raise gr.Error("pagecontent 必须是 JSON list")
    out: list[dict] = []
    for i, it in enumerate(obj):
        if not isinstance(it, dict):
            raise gr.Error(f"pagecontent[{i}] 必须是 object(dict)")
        out.append(it)
    return out


async def _run_pagecontent_from_pptx(
    pptx_path: str,
    language: str,
    chat_api_url: str,
    api_key: str,
    llm_model: str,
    page_count: int,
    style: str,
    aspect_ratio: str,
) -> Tuple[str, str]:
    if not pptx_path:
        raise gr.Error("请上传 .pptx 文件。")
    if not (chat_api_url or "").strip():
        raise gr.Error("请填写 chat_api_url。")
    if not (api_key or "").strip():
        raise gr.Error("请填写 api_key。")

    os.environ["DF_API_KEY"] = api_key

    run_dir = new_run_dir("ppt2polish")
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    src = Path(pptx_path)
    dst = (input_dir / "input.pptx").resolve()
    shutil.copy2(src, dst)

    req = Paper2FigureRequest(
        language=language,
        chat_api_url=chat_api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=llm_model,
        input_type="PPT",
        style=style or "",
        page_count=int(page_count),
    )
    state = Paper2FigureState(request=req, messages=[], agent_results={})
    state.result_path = str(run_dir)
    state.aspect_ratio = aspect_ratio or "16:9"
    state.paper_file = str(dst)

    from dataflow_agent.workflow import run_workflow
    final_state: Paper2FigureState = await run_workflow("paper2page_content", state)
    pagecontent = getattr(final_state, "pagecontent", []) or []
    return str(run_dir), json.dumps(pagecontent, ensure_ascii=False, indent=2)


async def _run_polish_all(
    run_dir: str,
    pagecontent_json: str,
    language: str,
    chat_api_url: str,
    api_key: str,
    llm_model: str,
    gen_fig_model: str,
    style: str,
    aspect_ratio: str,
) -> Tuple[list[str], str]:
    if not run_dir:
        raise gr.Error("请先生成 pagecontent。")
    if not (gen_fig_model or "").strip():
        raise gr.Error("请填写 gen_fig_model。")
    os.environ["DF_API_KEY"] = api_key

    pagecontent = _safe_json_loads(pagecontent_json)

    req = Paper2FigureRequest(
        language=language,
        chat_api_url=chat_api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=llm_model,
        gen_fig_model=gen_fig_model,
        style=style or "",
    )
    state = Paper2FigureState(request=req, messages=[], agent_results={})
    state.result_path = run_dir
    state.aspect_ratio = aspect_ratio or "16:9"
    state.pagecontent = pagecontent
    state.gen_down = False
    state.edit_page_num = -1
    state.edit_page_prompt = ""

    from dataflow_agent.workflow import run_workflow
    await run_workflow("paper2ppt_parallel", state)
    return _list_generated_pages(run_dir), _pdf_path(run_dir)


async def _run_edit_one(
    run_dir: str,
    page_id: int,
    edit_prompt: str,
    language: str,
    chat_api_url: str,
    api_key: str,
    llm_model: str,
    gen_fig_model: str,
    style: str,
    aspect_ratio: str,
) -> list[str]:
    if not run_dir:
        raise gr.Error("请先生成一次 polish（产生 ppt_pages/page_*.png）。")
    if page_id is None or int(page_id) < 0:
        raise gr.Error("请选择 page_id。")
    if not (edit_prompt or "").strip():
        raise gr.Error("请输入 edit_prompt。")
    os.environ["DF_API_KEY"] = api_key

    existing = _list_generated_pages(run_dir)
    if not existing:
        raise gr.Error("未找到已生成的页面图片（ppt_pages/page_*.png）。")
    if int(page_id) >= len(existing):
        raise gr.Error(f"page_id 超出范围：0~{len(existing) - 1}")

    req = Paper2FigureRequest(
        language=language,
        chat_api_url=chat_api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=llm_model,
        gen_fig_model=gen_fig_model,
        style=style or "",
        all_edited_down=False,
    )
    state = Paper2FigureState(request=req, messages=[], agent_results={})
    state.result_path = run_dir
    state.aspect_ratio = aspect_ratio or "16:9"
    state.gen_down = True
    state.generated_pages = existing
    state.edit_page_num = int(page_id)
    state.edit_page_prompt = str(edit_prompt)

    from dataflow_agent.workflow import run_workflow
    await run_workflow("paper2ppt_parallel", state)
    return _list_generated_pages(run_dir)


async def _run_export(run_dir: str, language: str, chat_api_url: str, api_key: str, llm_model: str, gen_fig_model: str) -> str:
    if not run_dir:
        raise gr.Error("请先生成一次 polish（产生 ppt_pages/page_*.png）。")
    os.environ["DF_API_KEY"] = api_key

    req = Paper2FigureRequest(
        language=language,
        chat_api_url=chat_api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=llm_model,
        gen_fig_model=gen_fig_model,
        all_edited_down=True,
    )
    state = Paper2FigureState(request=req, messages=[], agent_results={})
    state.result_path = run_dir
    state.gen_down = True
    from dataflow_agent.workflow import run_workflow
    await run_workflow("paper2ppt_parallel", state)
    return _pdf_path(run_dir)


def create_ppt2polish() -> gr.Blocks:
    with gr.Blocks(title="PPT Polish (HF Space)") as page:
        gr.Markdown("## ✨ PPT Polish")

        with gr.Accordion("🛠️ API 配置", open=True):
            with gr.Row():
                llm_model = gr.Textbox(label="🤖 文本模型 (model)", value="gpt-5.1")
                gen_fig_model = gr.Textbox(label="🖼️ 图像模型 (gen_fig_model)", value="gemini-3-pro-image-preview")
            with gr.Row():
                chat_api_url = gr.Textbox(label="🌐 chat_api_url", value=os.getenv("DF_API_URL", ""))
                api_key = gr.Textbox(label="🔑 api_key", value="", type="password")
            with gr.Row():
                language = gr.Dropdown(label="语言 (language)", choices=["zh", "en"], value="zh")
                aspect_ratio = gr.Dropdown(label="比例 (aspect_ratio)", choices=["16:9", "4:3", "1:1", "9:16"], value="16:9")
            with gr.Row():
                page_count = gr.Slider(label="最多处理页数 (page_count)", minimum=1, maximum=60, step=1, value=20)
                style = gr.Textbox(label="风格 (style)", value="")

        with gr.Accordion("📥 上传 PPTX", open=True):
            pptx = gr.File(label="上传 .pptx", file_types=[".pptx"], type="filepath", height=120)

        with gr.Row():
            btn_pagecontent = gr.Button("1) 解析并生成 pagecontent")
            btn_polish = gr.Button("2) 全量美化 (Polish)")

        run_dir_out = gr.Textbox(label="result_path (持久化目录)", interactive=False)
        pagecontent_json = gr.Textbox(label="pagecontent (可编辑 JSON)", lines=14)

        with gr.Accordion("🖼️ 预览 / 二次编辑", open=True):
            gallery = gr.Gallery(label="美化后预览 (ppt_pages/page_*.png)", columns=2, height=420)
            pdf_file = gr.File(label="下载 PDF (paper2ppt.pdf)", type="filepath")

            with gr.Row():
                page_id = gr.Number(label="page_id (0-based)", value=0, precision=0)
                edit_prompt = gr.Textbox(label="edit_prompt", placeholder="例如：更现代、留白更多、突出标题…")
            with gr.Row():
                btn_edit = gr.Button("3) 单页二次编辑")
                btn_export = gr.Button("4) 重新导出 PDF")

        btn_pagecontent.click(
            _run_pagecontent_from_pptx,
            inputs=[pptx, language, chat_api_url, api_key, llm_model, page_count, style, aspect_ratio],
            outputs=[run_dir_out, pagecontent_json],
        )

        btn_polish.click(
            _run_polish_all,
            inputs=[run_dir_out, pagecontent_json, language, chat_api_url, api_key, llm_model, gen_fig_model, style, aspect_ratio],
            outputs=[gallery, pdf_file],
        )

        btn_edit.click(
            _run_edit_one,
            inputs=[run_dir_out, page_id, edit_prompt, language, chat_api_url, api_key, llm_model, gen_fig_model, style, aspect_ratio],
            outputs=[gallery],
        )

        btn_export.click(
            _run_export,
            inputs=[run_dir_out, language, chat_api_url, api_key, llm_model, gen_fig_model],
            outputs=[pdf_file],
        )

    return page
