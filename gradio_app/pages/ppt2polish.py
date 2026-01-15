from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
try:
    from gradio_app.utils.space_paths import new_run_dir
except ModuleNotFoundError:
    from utils.space_paths import new_run_dir  # type: ignore

log = get_logger(__name__)


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


def _norm_index(idx) -> int:
    if isinstance(idx, (tuple, list)) and idx:
        return int(idx[0])
    return int(idx)


def _page_title_from_item(item: dict, idx: int) -> str:
    for k in ["title", "slide_title", "page_title", "heading"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"第 {idx + 1} 页"


def _page_layout_from_item(item: dict) -> str:
    for k in ["layout_description", "content", "text", "description", "body"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list):
            s = "\n".join(str(x).strip() for x in v if str(x).strip())
            if s.strip():
                return s.strip()
    return ""


def _page_bullets_from_item(item: dict) -> str:
    for k in ["bullets", "bullet_points", "points", "key_points"]:
        v = item.get(k)
        if isinstance(v, list):
            return "\n".join(str(x).strip() for x in v if str(x).strip())
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _page_asset_ref_from_item(item: dict) -> str:
    for k in ["asset_ref", "asset", "assetRef"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _build_pagecontent_outline_md(pages: list[dict]) -> str:
    if not pages:
        return "暂无 pagecontent。请先点击「1) 解析并生成 pagecontent」。"
    lines = ["### pagecontent 概览"]
    for i, it in enumerate(pages):
        title = _page_title_from_item(it, i)
        snippet = _page_layout_from_item(it)
        snippet = (snippet[:120] + "…") if len(snippet) > 120 else snippet
        if snippet:
            lines.append(f"- **{i + 1}. {title}**：{snippet}")
        else:
            lines.append(f"- **{i + 1}. {title}**")
    return "\n".join(lines)


def _pc_nav_md(idx: int, total: int) -> str:
    if total <= 0:
        return "当前编辑页：—"
    idx = max(0, min(int(idx), total - 1))
    return f"当前编辑页：第 **{idx + 1} / {total}** 页"


def _pc_render_controls(
    pages: list[dict],
    page_idx: int,
) -> Tuple[int, str, str, str, str, str, gr.Button, gr.Button, gr.Number, gr.Button]:
    total = len(pages or [])
    if total <= 0:
        return (
            0,
            "",
            "",
            "",
            "",
            _pc_nav_md(0, 0),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=None, interactive=False, minimum=None, maximum=None, step=1, precision=0),
            gr.update(interactive=False),
        )

    idx = max(0, min(int(page_idx or 0), total - 1))
    it = pages[idx] or {}
    return (
        idx,
        _page_title_from_item(it, idx),
        _page_layout_from_item(it),
        _page_bullets_from_item(it),
        _page_asset_ref_from_item(it),
        _pc_nav_md(idx, total),
        gr.update(interactive=(idx > 0)),
        gr.update(interactive=(idx < total - 1)),
        gr.update(value=idx + 1, interactive=True, minimum=1, maximum=total, step=1, precision=0),
        gr.update(interactive=True),
    )


def _pc_prev_next(
    pages: list[dict],
    page_idx: int,
    delta: int,
) -> Tuple[int, str, str, str, str, str, gr.Button, gr.Button, gr.Number, gr.Button]:
    total = len(pages or [])
    if total <= 0:
        return _pc_render_controls([], 0)
    idx = max(0, min(int(page_idx or 0) + int(delta), total - 1))
    return _pc_render_controls(pages, idx)


def _pc_go_to(
    pages: list[dict],
    page_no_1based: float,
) -> Tuple[int, str, str, str, str, str, gr.Button, gr.Button, gr.Number, gr.Button]:
    total = len(pages or [])
    if total <= 0:
        return _pc_render_controls([], 0)
    if page_no_1based is None:
        raise gr.Error("请输入要跳转的页码（从 1 开始）。")
    idx = int(page_no_1based) - 1
    idx = max(0, min(idx, total - 1))
    return _pc_render_controls(pages, idx)


def _sync_pagecontent_ui(
    pagecontent_json: str,
) -> Tuple[list[dict], int, str, str, str, str, str, str, gr.Button, gr.Button, gr.Number, gr.Button]:
    pages = _safe_json_loads(pagecontent_json)
    outline = _build_pagecontent_outline_md(pages)
    idx, title, layout, bullets, asset_ref, nav, prev_u, next_u, go_num_u, go_btn_u = _pc_render_controls(pages, 0)
    return pages, idx, title, layout, bullets, asset_ref, outline, nav, prev_u, next_u, go_num_u, go_btn_u


def _parse_bullets(text: str) -> list[str]:
    out: list[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(("-", "*")):
            s = s[1:].strip()
        if s:
            out.append(s)
    return out


def _apply_page_edit(
    pages: list[dict],
    page_idx: int,
    title: str,
    layout_description: str,
    bullets_text: str,
    asset_ref: str,
) -> Tuple[list[dict], str, str]:
    if not pages:
        raise gr.Error("pagecontent 为空：请先生成 pagecontent。")
    idx = max(0, min(int(page_idx or 0), len(pages) - 1))
    it = dict(pages[idx] or {})

    title = (title or "").strip()
    if title:
        it["title"] = title
    else:
        it.pop("title", None)

    layout_description = (layout_description or "").strip()
    if layout_description:
        it["layout_description"] = layout_description
    else:
        it.pop("layout_description", None)

    bullets = _parse_bullets(bullets_text)
    if bullets:
        it["bullets"] = bullets
    else:
        it.pop("bullets", None)

    asset_ref = (asset_ref or "").strip()
    for k in ["asset_ref", "asset", "assetRef"]:
        it.pop(k, None)
    if asset_ref:
        it["asset_ref"] = asset_ref

    pages = list(pages)
    pages[idx] = it

    json_out = json.dumps(pages, ensure_ascii=False, indent=2)
    return pages, json_out, _build_pagecontent_outline_md(pages)


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
    final_state = await run_workflow("paper2page_content", state)
    # langgraph will return `dict` when the state schema is a dataclass.
    if isinstance(final_state, dict):
        pagecontent = final_state.get("pagecontent") or []
    else:
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
) -> Tuple[list[str], object, list[str]]:
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
    pages = _list_generated_pages(run_dir)
    return pages, pages


async def _run_export(run_dir: str, language: str, chat_api_url: str, api_key: str, llm_model: str, gen_fig_model: str) -> object:
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
    pdf = _pdf_path(run_dir)
    return gr.update(value=pdf, interactive=bool(pdf))


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
        pagecontent_state = gr.State([])
        pc_edit_idx_state = gr.State(0)

        with gr.Accordion("📝 pagecontent（按页编辑）", open=True):
            pagecontent_outline = gr.Markdown(_build_pagecontent_outline_md([]))
            pc_nav = gr.Markdown(_pc_nav_md(0, 0))
            with gr.Row():
                btn_pc_prev = gr.Button("上一页", interactive=False)
                btn_pc_next = gr.Button("下一页", interactive=False)
                pc_go_num = gr.Number(label="跳转到页（从 1 开始）", value=None, precision=0, minimum=1, step=1, interactive=False)
                btn_pc_go = gr.Button("跳转", interactive=False)
            page_title = gr.Textbox(label="标题 (title)")
            page_layout = gr.Textbox(label="页面内容 (layout_description)", lines=10, placeholder="用自然语言描述这一页要表达的内容/结构。")
            page_bullets = gr.Textbox(label="要点 (bullets，每行一条)", lines=6, placeholder="- 要点1\n- 要点2")
            page_asset_ref = gr.Textbox(label="素材引用 (asset_ref，可选)", placeholder="例如：/data/outputs/.../fig1.png 或 Table 2")
            with gr.Row():
                btn_apply_page = gr.Button("应用本页修改")
                btn_reload_pagecontent = gr.Button("从 JSON 重新加载")

        with gr.Accordion("高级：pagecontent JSON", open=False):
            pagecontent_json = gr.Textbox(label="pagecontent (JSON)", lines=14)

        with gr.Accordion("🖼️ 预览 / 二次编辑", open=True):
            with gr.Row():
                pdf_download = gr.DownloadButton("下载 PDF", value=None, interactive=False)
                pages_zip_download = gr.DownloadButton("下载页面 ZIP", value=None, interactive=False)

            gr.Markdown("提示：点击缩略图选择页面；或用“上一页/下一页”浏览。")
            page_info = gr.Markdown(_page_info_md(0, 0))

            pages_state = gr.State([])

            with gr.Row():
                preview_image = gr.Image(label="当前页预览", type="filepath", height=520)
                gallery = gr.Gallery(label="美化后缩略图 (ppt_pages/page_*.png)", columns=2, height=520)

            with gr.Row():
                btn_prev = gr.Button("上一页", interactive=False)
                page_id = gr.Slider(label="选择页 (page_id)", minimum=0, maximum=0, step=1, value=0, interactive=False)
                btn_next = gr.Button("下一页", interactive=False)

            with gr.Row():
                edit_prompt = gr.Textbox(label="edit_prompt", placeholder="例如：更现代、留白更多、突出标题…")
            with gr.Row():
                btn_edit = gr.Button("3) 单页二次编辑")
                btn_export = gr.Button("4) 重新导出 PDF")

        pc_evt = btn_pagecontent.click(
            _run_pagecontent_from_pptx,
            inputs=[pptx, language, chat_api_url, api_key, llm_model, page_count, style, aspect_ratio],
            outputs=[run_dir_out, pagecontent_json],
        )
        pc_ui_evt = pc_evt.then(
            _sync_pagecontent_ui,
            inputs=[pagecontent_json],
            outputs=[
                pagecontent_state,
                pc_edit_idx_state,
                page_title,
                page_layout,
                page_bullets,
                page_asset_ref,
                pagecontent_outline,
                pc_nav,
                btn_pc_prev,
                btn_pc_next,
                pc_go_num,
                btn_pc_go,
            ],
        )
        pc_reset_evt = pc_ui_evt.then(
            lambda: ([], gr.update(value=None, interactive=False), []),
            inputs=[],
            outputs=[gallery, pdf_download, pages_state],
        )
        pc_reset_evt.then(
            _sync_after_pages_updated,
            inputs=[pages_state, page_id, run_dir_out],
            outputs=[page_id, preview_image, page_info, btn_prev, btn_next, pages_zip_download],
        )

        btn_reload_pagecontent.click(
            _sync_pagecontent_ui,
            inputs=[pagecontent_json],
            outputs=[
                pagecontent_state,
                pc_edit_idx_state,
                page_title,
                page_layout,
                page_bullets,
                page_asset_ref,
                pagecontent_outline,
                pc_nav,
                btn_pc_prev,
                btn_pc_next,
                pc_go_num,
                btn_pc_go,
            ],
        )
        btn_apply_page.click(
            _apply_page_edit,
            inputs=[pagecontent_state, pc_edit_idx_state, page_title, page_layout, page_bullets, page_asset_ref],
            outputs=[pagecontent_state, pagecontent_json, pagecontent_outline],
        ).then(
            _pc_render_controls,
            inputs=[pagecontent_state, pc_edit_idx_state],
            outputs=[
                pc_edit_idx_state,
                page_title,
                page_layout,
                page_bullets,
                page_asset_ref,
                pc_nav,
                btn_pc_prev,
                btn_pc_next,
                pc_go_num,
                btn_pc_go,
            ],
        )
        btn_pc_prev.click(
            lambda pages, idx: _pc_prev_next(pages, idx, -1),
            inputs=[pagecontent_state, pc_edit_idx_state],
            outputs=[
                pc_edit_idx_state,
                page_title,
                page_layout,
                page_bullets,
                page_asset_ref,
                pc_nav,
                btn_pc_prev,
                btn_pc_next,
                pc_go_num,
                btn_pc_go,
            ],
        )
        btn_pc_next.click(
            lambda pages, idx: _pc_prev_next(pages, idx, +1),
            inputs=[pagecontent_state, pc_edit_idx_state],
            outputs=[
                pc_edit_idx_state,
                page_title,
                page_layout,
                page_bullets,
                page_asset_ref,
                pc_nav,
                btn_pc_prev,
                btn_pc_next,
                pc_go_num,
                btn_pc_go,
            ],
        )
        btn_pc_go.click(
            _pc_go_to,
            inputs=[pagecontent_state, pc_go_num],
            outputs=[
                pc_edit_idx_state,
                page_title,
                page_layout,
                page_bullets,
                page_asset_ref,
                pc_nav,
                btn_pc_prev,
                btn_pc_next,
                pc_go_num,
                btn_pc_go,
            ],
        )

        polish_evt = btn_polish.click(
            _run_polish_all,
            inputs=[run_dir_out, pagecontent_json, language, chat_api_url, api_key, llm_model, gen_fig_model, style, aspect_ratio],
            outputs=[gallery, pdf_download, pages_state],
        )
        polish_evt.then(
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

        def _on_gallery_select(pages: list[str], evt: gr.SelectData):  # type: ignore[name-defined]
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
            inputs=[pages_state],
            outputs=[page_id, preview_image, page_info, btn_prev, btn_next],
        )

    return page
