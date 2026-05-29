from __future__ import annotations

import time
from pathlib import Path

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureState
from dataflow_agent.toolkits.ppt_native import (
    export_svg_deck_to_pptx,
    render_pagecontent_to_svg_deck,
)
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow.registry import register

log = get_logger(__name__)


def _ensure_result_path(state: Paper2FigureState) -> str:
    raw = getattr(state, "result_path", None)
    if raw:
        Path(raw).mkdir(parents=True, exist_ok=True)
        return str(raw)

    base_dir = (get_project_root() / "outputs" / "paper2ppt_native" / str(int(time.time()))).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    state.result_path = str(base_dir)
    return state.result_path


@register("paper2ppt_svg_native")
def create_paper2ppt_svg_native_graph() -> GenericGraphBuilder:  # noqa: N802
    """Generate a native editable PPTX from existing paper2ppt pagecontent.

    This is the first native branch:
    pagecontent -> constrained SVG files -> PPT Master's DrawingML converter.
    The existing image-based workflow remains unchanged and can be used as
    fallback while this path matures.
    """
    builder = GenericGraphBuilder(state_model=Paper2FigureState, entry_point="_start_")

    def _start_(state: Paper2FigureState) -> Paper2FigureState:
        result_path = Path(_ensure_result_path(state))
        state.pagecontent = state.pagecontent or []
        if not state.pagecontent:
            raise ValueError("[paper2ppt_svg_native] pagecontent is required")
        (result_path / "svg_output").mkdir(parents=True, exist_ok=True)
        return state

    async def render_svg_pages(state: Paper2FigureState) -> Paper2FigureState:
        result_root = Path(_ensure_result_path(state))
        svg_dir = result_root / "svg_output"
        req = getattr(state, "request", None)
        api_key = (
            getattr(req, "chat_api_key", "")
            or getattr(req, "api_key", "")
            or ""
        )
        render_result = await render_pagecontent_to_svg_deck(
            state.pagecontent or [],
            svg_dir,
            result_root=result_root,
            style=getattr(req, "style", "") or getattr(state, "style", ""),
            language=getattr(req, "language", "zh"),
            chat_api_url=getattr(req, "chat_api_url", "") or "",
            api_key=api_key,
            model=getattr(req, "model", "") or "",
        )
        report_by_index = {
            int(report.get("page_idx", -1)): report
            for report in render_result.page_reports
        }

        updated_pagecontent = []
        for idx, item in enumerate(state.pagecontent or []):
            next_item = dict(item) if isinstance(item, dict) else {"title": str(item)}
            page_report = report_by_index.get(idx, {})
            next_item.update(
                {
                    "page_idx": idx,
                    "generated_svg_path": str(render_result.svg_files[idx]),
                    "mode": page_report.get("mode", "native_svg"),
                    "native_svg_report": page_report,
                }
            )
            updated_pagecontent.append(next_item)
        state.pagecontent = updated_pagecontent
        log.info(
            "[paper2ppt_svg_native] rendered %s SVG pages into %s",
            len(render_result.svg_files),
            svg_dir,
        )
        return state

    def export_native_pptx(state: Paper2FigureState) -> Paper2FigureState:
        result_root = Path(_ensure_result_path(state))
        svg_files = [
            Path(item["generated_svg_path"])
            for item in state.pagecontent or []
            if isinstance(item, dict) and item.get("generated_svg_path")
        ]
        if not svg_files:
            raise ValueError("[paper2ppt_svg_native] no SVG pages to export")

        pptx_path = result_root / "paper2ppt_native_editable.pptx"
        export_svg_deck_to_pptx(svg_files, pptx_path, verbose=False)
        state.ppt_pptx_path = str(pptx_path)
        state.ppt_pdf_path = ""
        log.info("[paper2ppt_svg_native] exported native PPTX: %s", pptx_path)
        return state

    nodes = {
        "_start_": _start_,
        "render_svg_pages": render_svg_pages,
        "export_native_pptx": export_native_pptx,
        "_end_": lambda state: state,
    }
    edges = [
        ("_start_", "render_svg_pages"),
        ("render_svg_pages", "export_native_pptx"),
        ("export_native_pptx", "_end_"),
    ]
    builder.add_nodes(nodes).add_edges(edges)
    return builder
