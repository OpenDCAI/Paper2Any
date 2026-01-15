from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
from gradio_app.utils.space_paths import new_run_dir

log = get_logger(__name__)


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


def _list_generated_pages(run_dir: str) -> list[str]:
    if not run_dir:
        return []
    img_dir = Path(run_dir) / "ppt_pages"
    if not img_dir.exists():
        return []
    return [str(p.resolve()) for p in sorted(img_dir.glob("page_*.png"))]


def _pdf_path(run_dir: str) -> Optional[str]:
    if not run_dir:
        return None
    p = Path(run_dir) / "paper2ppt.pdf"
    return str(p.resolve()) if p.exists() else None


def _page_info_md(page_id: int, total: int) -> str:
    if total <= 0:
        return "暂无预览（还没有生成页面图片）。"
    page_id = max(0, min(int(page_id), total - 1))
    return f"当前页：第 **{page_id + 1} / {total}** 页（`page_id={page_id}`）"


def _make_pages_zip(run_dir: str) -> Optional[str]:
    if not run_dir:
        return None
    img_dir = Path(run_dir) / "ppt_pages"
    if not img_dir.exists():
        return None

    pages = sorted(img_dir.glob("page_*.png"))
    if not pages:
        return None

    zip_path = (Path(run_dir) / "ppt_pages.zip").resolve()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pages:
            zf.write(p, arcname=p.name)
    return str(zip_path)


def _norm_index(idx: Any) -> int:
    if isinstance(idx, (tuple, list)) and idx:
        return int(idx[0])
    return int(idx)


def _sync_after_pages_updated(
    pages: list[str],
    page_id: int,
    run_dir: str,
) -> Tuple[gr.Slider, Optional[str], str, gr.Button, gr.Button, gr.DownloadButton]:
    total = len(pages or [])
    if total <= 0:
        return (
            gr.update(minimum=0, maximum=0, value=0, step=1, interactive=False),
            None,
            _page_info_md(0, 0),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=None, interactive=False),
        )

    pid = max(0, min(int(page_id or 0), total - 1))
    zip_path = _make_pages_zip(run_dir)
    return (
        gr.update(minimum=0, maximum=total - 1, value=pid, step=1, interactive=True),
        pages[pid],
        _page_info_md(pid, total),
        gr.update(interactive=(pid > 0)),
        gr.update(interactive=(pid < total - 1)),
        gr.update(value=zip_path, interactive=bool(zip_path)),
    )


def _on_page_id_change(pages: list[str], page_id: int) -> Tuple[Optional[str], str, gr.Button, gr.Button]:
    total = len(pages or [])
    if total <= 0:
        return None, _page_info_md(0, 0), gr.update(interactive=False), gr.update(interactive=False)

    pid = max(0, min(int(page_id or 0), total - 1))
    return pages[pid], _page_info_md(pid, total), gr.update(interactive=(pid > 0)), gr.update(interactive=(pid < total - 1))


def _on_prev_next(pages: list[str], page_id: int, delta: int) -> Tuple[gr.Slider, Optional[str], str, gr.Button, gr.Button]:
    total = len(pages or [])
    if total <= 0:
        return (
            gr.update(value=0),
            None,
            _page_info_md(0, 0),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    pid = max(0, min(int(page_id or 0) + int(delta), total - 1))
    return (
        gr.update(value=pid),
        pages[pid],
        _page_info_md(pid, total),
        gr.update(interactive=(pid > 0)),
        gr.update(interactive=(pid < total - 1)),
    )


async def _run_pagecontent(
    input_mode: str,
    text: str,
    language: str,
    chat_api_url: str,
    api_key: str,
    llm_model: str,
    page_count: int,
    style: str,
    aspect_ratio: str,
) -> Tuple[str, str]:
    mode = (input_mode or "").strip().upper()
    if mode not in {"TEXT", "TOPIC"}:
        raise gr.Error("当前 Space 版本暂不支持 PDF；请选择 TEXT 或 TOPIC。")
    if not (text or "").strip():
        raise gr.Error("请输入内容。")
    if not (chat_api_url or "").strip():
        raise gr.Error("请填写 chat_api_url。")
    if not (api_key or "").strip():
        raise gr.Error("请填写 api_key。")

    os.environ["DF_API_KEY"] = api_key

    run_dir = new_run_dir("paper2ppt")
    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    (run_dir / "input" / f"input_{mode.lower()}.txt").write_text(text, encoding="utf-8")

    req = Paper2FigureRequest(
        language=language,
        chat_api_url=chat_api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=llm_model,
        target=text,
        input_type=mode,
        style=style or "",
        page_count=int(page_count),
    )
    state = Paper2FigureState(request=req, messages=[], agent_results={})
    state.result_path = str(run_dir)
    state.aspect_ratio = aspect_ratio or "16:9"
    state.text_content = text

    from dataflow_agent.workflow import run_workflow
    final_state = await run_workflow("paper2page_content", state)
    # langgraph will return `dict` when the state schema is a dataclass.
    if isinstance(final_state, dict):
        pagecontent = final_state.get("pagecontent") or []
    else:
        pagecontent = getattr(final_state, "pagecontent", []) or []
    return str(run_dir), json.dumps(pagecontent, ensure_ascii=False, indent=2)


async def _run_generate(
    run_dir: str,
    pagecontent_json: str,
    language: str,
    chat_api_url: str,
    api_key: str,
    llm_model: str,
    gen_fig_model: str,
    style: str,
    aspect_ratio: str,
) -> Tuple[list[str], Any, list[str]]:
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

    pages = _list_generated_pages(run_dir)
    pdf = _pdf_path(run_dir)
    return pages, gr.update(value=pdf, interactive=bool(pdf)), pages


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
) -> Tuple[list[str], list[str]]:
    if not run_dir:
        raise gr.Error("请先生成一次 PPT（产生 ppt_pages/page_*.png）。")
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
    pages = _list_generated_pages(run_dir)
    return pages, pages


async def _run_export(run_dir: str, language: str, chat_api_url: str, api_key: str, llm_model: str, gen_fig_model: str) -> Any:
    if not run_dir:
        raise gr.Error("请先生成一次 PPT（产生 ppt_pages/page_*.png）。")
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
    pdf = _pdf_path(run_dir)
    return gr.update(value=pdf, interactive=bool(pdf))


def create_paper2ppt() -> gr.Blocks:
    with gr.Blocks(title="Paper2PPT (HF Space)") as page:
        gr.Markdown("## 🎬 Paper2PPT")

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
                page_count = gr.Slider(label="页数 (page_count)", minimum=1, maximum=30, step=1, value=8)
                style = gr.Textbox(label="风格 (style)", value="")

        with gr.Accordion("📥 输入", open=True):
            input_mode = gr.Radio(label="输入类型", choices=["TEXT", "TOPIC"], value="TEXT")
            text = gr.Textbox(label="输入内容", lines=10, placeholder="粘贴论文摘要/要点，或输入 topic。")

        with gr.Row():
            btn_pagecontent = gr.Button("1) 生成 pagecontent")
            btn_generate = gr.Button("2) 生成 PPT (PDF)")

        run_dir_out = gr.Textbox(label="result_path (持久化目录)", interactive=False)
        pagecontent_json = gr.Textbox(label="pagecontent (可编辑 JSON)", lines=14)

        with gr.Accordion("🖼️ 预览 / 美化", open=True):
            with gr.Row():
                pdf_download = gr.DownloadButton("下载 PDF", value=None, file_name="paper2ppt.pdf", interactive=False)
                pages_zip_download = gr.DownloadButton("下载页面 ZIP", value=None, file_name="ppt_pages.zip", interactive=False)

            gr.Markdown("提示：点击缩略图选择页面；或用“上一页/下一页”浏览。")
            page_info = gr.Markdown(_page_info_md(0, 0))

            pages_state = gr.State([])

            with gr.Row():
                preview_image = gr.Image(label="当前页预览", type="filepath", height=520)
                gallery = gr.Gallery(label="生成页缩略图 (ppt_pages/page_*.png)", columns=2, height=520)

            with gr.Row():
                btn_prev = gr.Button("上一页", interactive=False)
                page_id = gr.Slider(label="选择页 (page_id)", minimum=0, maximum=0, step=1, value=0, interactive=False)
                btn_next = gr.Button("下一页", interactive=False)

            with gr.Row():
                edit_prompt = gr.Textbox(label="单页编辑提示 (edit_prompt)", placeholder="例如：更简洁、加入公司配色、提升对比度…")
            with gr.Row():
                btn_edit = gr.Button("3) 单页二次编辑")
                btn_export = gr.Button("4) 重新导出 PDF")

        pc_evt = btn_pagecontent.click(
            _run_pagecontent,
            inputs=[input_mode, text, language, chat_api_url, api_key, llm_model, page_count, style, aspect_ratio],
            outputs=[run_dir_out, pagecontent_json],
        )
        pc_reset_evt = pc_evt.then(
            lambda: ([], gr.update(value=None, interactive=False), []),
            inputs=[],
            outputs=[gallery, pdf_download, pages_state],
        )
        pc_reset_evt.then(
            _sync_after_pages_updated,
            inputs=[pages_state, page_id, run_dir_out],
            outputs=[page_id, preview_image, page_info, btn_prev, btn_next, pages_zip_download],
        )

        gen_evt = btn_generate.click(
            _run_generate,
            inputs=[run_dir_out, pagecontent_json, language, chat_api_url, api_key, llm_model, gen_fig_model, style, aspect_ratio],
            outputs=[gallery, pdf_download, pages_state],
        )
        gen_evt.then(
            _sync_after_pages_updated,
            inputs=[pages_state, page_id, run_dir_out],
            outputs=[page_id, preview_image, page_info, btn_prev, btn_next, pages_zip_download],
        )

        edit_evt = btn_edit.click(
            _run_edit_one,
            inputs=[run_dir_out, page_id, edit_prompt, language, chat_api_url, api_key, llm_model, gen_fig_model, style, aspect_ratio],
            outputs=[gallery, pages_state],
        )
        edit_evt.then(
            _sync_after_pages_updated,
            inputs=[pages_state, page_id, run_dir_out],
            outputs=[page_id, preview_image, page_info, btn_prev, btn_next, pages_zip_download],
        )

        btn_export.click(
            _run_export,
            inputs=[run_dir_out, language, chat_api_url, api_key, llm_model, gen_fig_model],
            outputs=[pdf_download],
        )

        page_id.change(
            _on_page_id_change,
            inputs=[pages_state, page_id],
            outputs=[preview_image, page_info, btn_prev, btn_next],
        )
        btn_prev.click(
            lambda pages, pid: _on_prev_next(pages, pid, -1),
            inputs=[pages_state, page_id],
            outputs=[page_id, preview_image, page_info, btn_prev, btn_next],
        )
        btn_next.click(
            lambda pages, pid: _on_prev_next(pages, pid, +1),
            inputs=[pages_state, page_id],
            outputs=[page_id, preview_image, page_info, btn_prev, btn_next],
        )

        def _on_gallery_select(pages: list[str], run_dir: str, evt: gr.SelectData):  # type: ignore[name-defined]
            idx = _norm_index(getattr(evt, "index", 0))
            total = len(pages or [])
            if total <= 0:
                return (
                    gr.update(value=0, interactive=False),
                    None,
                    _page_info_md(0, 0),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )
            pid = max(0, min(int(idx), total - 1))
            return (
                gr.update(value=pid),
                pages[pid],
                _page_info_md(pid, total),
                gr.update(interactive=(pid > 0)),
                gr.update(interactive=(pid < total - 1)),
            )

        gallery.select(
            _on_gallery_select,
            inputs=[pages_state, run_dir_out],
            outputs=[page_id, preview_image, page_info, btn_prev, btn_next],
        )

    return page
