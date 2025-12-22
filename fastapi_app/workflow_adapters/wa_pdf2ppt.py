from __future__ import annotations

"""
pdf2ppt_with_sam 工作流封装。

- 输入：一个 PDF 文件路径
- 调用 dataflow_agent.workflow.run_workflow("pdf2ppt_with_sam", state)
- 输出：生成的 PPT 路径

当前直接复用 Paper2FigureState / Paper2FigureRequest，
逻辑与 tests/test_pdf2ppt.py 中保持一致。
"""

from pathlib import Path
from typing import Any

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow import run_workflow
import time

from fastapi_app.schemas import Paper2PPTRequest, Paper2PPTResponse

log = get_logger(__name__)


def _ensure_result_path_for_pdf2ppt(invite_code: str | None) -> Path:
    """
    为 pdf2ppt_with_sam workflow 统一一个根输出目录：
    outputs/{invite_code or 'default'}/pdf2ppt_with_sam/<timestamp>/
    """
    project_root = get_project_root()
    ts = int(time.time())
    code = invite_code or "default"
    base_dir = (project_root / "outputs" / code / "pdf2ppt_with_sam" / str(ts)).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


async def run_pdf2ppt_wf_api(req: Paper2PPTRequest) -> Paper2PPTResponse:
    """
    对 pdf2ppt_with_sam workflow 的封装。

    入参:
        - req.input_type: 目前预期为 "PDF"
        - req.input_content: PDF 文件路径（上传后保存到本地的路径）
        - req.invite_code: 用于区分输出目录（可选）

    行为:
        - 归一化 PDF 路径
        - 计算本次调用统一的 result_path
        - 构造 Paper2FigureState / Paper2FigureRequest
        - 把 pdf 路径挂到 state.pdf_file 上，并设置 state.result_path
        - 调用 run_workflow("pdf2ppt_with_sam", state)
        - 从 final_state.ppt_path 提取生成的 PPT 路径
        - 从 final_state.result_path 提取最终输出根目录（如无则回退到本次计算的 result_path）
        - 封装为 Paper2PPTResponse 返回
    """
    project_root = get_project_root()

    # 允许在 req.input_content 中传相对路径，这里统一转绝对路径
    raw_pdf_path = Path(req.input_content or "")
    if not raw_pdf_path.is_absolute():
        pdf_path = (project_root / raw_pdf_path).resolve()
    else:
        pdf_path = raw_pdf_path.resolve()

    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"[pdf2ppt] PDF file not found: {pdf_path}")

    # 统一输出根目录，按 invite_code + 时间戳 区分
    result_root = _ensure_result_path_for_pdf2ppt(getattr(req, "invite_code", None))

    # 构造 state/request，传入前端配置的参数
    p2f_req = Paper2FigureRequest(
        chat_api_url=req.chat_api_url,
        api_key=req.api_key,
        model=req.model,
        gen_fig_model=req.gen_fig_model,
        language=req.language,
        style=req.style,
        page_count=req.page_count,
    )
    state = Paper2FigureState(
        messages=[],
        request=p2f_req,
    )
    state.pdf_file = str(pdf_path)
    state.result_path = str(result_root)
    state.use_ai_edit = req.use_ai_edit
    log.critical(f"[pdf2ppt 是否使用AI： ] state.use_ai_edit = {state.use_ai_edit}")

    log.info(
        f"[pdf2ppt] start workflow 'pdf2ppt_with_sam_ocr_mineru' "
        f"with pdf_file={state.pdf_file}, result_path={state.result_path}"
    )

    final_state: Paper2FigureState = await run_workflow("pdf2ppt_with_sam_ocr_mineru", state)

    ppt_path_value = final_state["ppt_path"]
    if not ppt_path_value:
        raise RuntimeError("[pdf2ppt] workflow did not set `ppt_path` on state")

    ppt_path = Path(str(ppt_path_value))
    if not ppt_path.is_absolute():
        ppt_path = (project_root / ppt_path).resolve()

    if not ppt_path.exists() or not ppt_path.is_file():
        raise FileNotFoundError(f"[pdf2ppt] generated PPT file not found: {ppt_path}")

    # 最终的 result_path 以 workflow 内设置为准，若不存在则回退到 result_root
    final_result_path = getattr(final_state, "result_path", str(result_root))
    final_result_dir = Path(str(final_result_path))
    if not final_result_dir.is_absolute():
        final_result_dir = (project_root / final_result_dir).resolve()

    log.info(f"[pdf2ppt] generated PPT path: {ppt_path}, result_path: {final_result_dir}")

    # Paper2PPTResponse 里同时支持 pdf/pptx 路径字段，这里只填 pptx 路径
    return Paper2PPTResponse(
        success=True,
        ppt_pdf_path="",
        ppt_pptx_path=str(ppt_path),
        pagecontent=[],
        result_path=str(final_result_dir),
        all_output_files=[str(ppt_path)],
    )
