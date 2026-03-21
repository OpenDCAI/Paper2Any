"""
Paper2Drawio Service 层
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import UploadFile, Request

from dataflow_agent.state import Paper2DrawioState, Paper2DrawioRequest
from dataflow_agent.toolkits.drawio_tools import wrap_xml, extract_cells
from dataflow_agent.toolkits.multimodaltool.mineru_tool import run_mineru_pdf_extract_http
from dataflow_agent.logger import get_logger
from fastapi_app.config.settings import settings

log = get_logger(__name__)

BASE_OUTPUT_DIR = Path("outputs").resolve()
task_semaphore = asyncio.Semaphore(2)


class Paper2DrawioService:
    """Paper2Drawio 业务服务"""

    @staticmethod
    def _parse_model_candidates(model_value: str | None) -> List[str]:
        raw = (model_value or settings.PAPER2DRAWIO_DEFAULT_MODEL or "").strip()
        seen: set[str] = set()
        ordered: List[str] = []
        for item in raw.split(","):
            model = item.strip()
            if not model or model in seen:
                continue
            seen.add(model)
            ordered.append(model)
        if ordered:
            return ordered
        return [settings.PAPER2DRAWIO_DEFAULT_MODEL]

    @staticmethod
    def _model_run_dir(parent: Path, model_name: str, index: int) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._")
        if not safe:
            safe = f"model_{index + 1}"
        run_dir = parent / "candidates" / f"{index + 1:02d}_{safe}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _create_run_dir(self, prefix: str, email: Optional[str]) -> Path:
        """创建运行目录"""
        ts = int(time.time())
        run_dir = BASE_OUTPUT_DIR / prefix / str(ts)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "input").mkdir(exist_ok=True)
        return run_dir

    async def _run_workflow_once(
        self,
        *,
        request_chat_api_url: str,
        request_api_key: str,
        candidate_model: str,
        enable_vlm_validation: bool,
        vlm_model: Optional[str],
        vlm_validation_max_retries: Optional[int],
        workflow_input_type: str,
        workflow_paper_file: str,
        workflow_text_content: str,
        diagram_type: str,
        diagram_style: str,
        language: str,
        use_sam3_workflow: bool,
        result_dir: Path,
    ) -> Dict[str, Any]:
        state = Paper2DrawioState(
            request=Paper2DrawioRequest(
                language=language,
                chat_api_url=request_chat_api_url,
                api_key=request_api_key,
                chat_api_key=request_api_key,
                model=candidate_model,
                enable_vlm_validation=bool(enable_vlm_validation),
                vlm_model=vlm_model or settings.PAPER2DRAWIO_VLM_MODEL,
                vlm_validation_max_retries=vlm_validation_max_retries or 3,
                input_type=workflow_input_type,
                diagram_type=diagram_type,
                diagram_style=diagram_style,
            ),
            paper_file=workflow_paper_file,
            text_content=workflow_text_content,
            result_path=str(result_dir),
        )

        from dataflow_agent.workflow.registry import RuntimeRegistry

        try:
            async with task_semaphore:
                workflow_name = "paper2drawio_visual" if use_sam3_workflow else "paper2drawio_semantic"
                log.info(
                    f"[paper2drawio] selected workflow={workflow_name}, input_type={workflow_input_type}, model={candidate_model}"
                )
                factory = RuntimeRegistry.get(workflow_name)
                builder = factory()
                graph = builder.build()
                final_state = await graph.ainvoke(state)

            raw_xml = final_state.get("drawio_xml", "") if isinstance(final_state, dict) else (final_state.drawio_xml or "")
            output_path = final_state.get("output_xml_path", "") if isinstance(final_state, dict) else (final_state.output_xml_path or "")
            xml_content = wrap_xml(raw_xml) if raw_xml else ""

            return {
                "success": bool(xml_content),
                "xml_content": xml_content,
                "file_path": output_path,
                "error": None if xml_content else f"Model {candidate_model} failed to generate diagram",
                "used_model": candidate_model,
            }
        except asyncio.CancelledError:
            log.info(f"[paper2drawio] cancelled candidate model={candidate_model}")
            raise
        except Exception as e:
            log.error(f"生成图表失败(model={candidate_model}): {e}")
            return {
                "success": False,
                "xml_content": "",
                "file_path": "",
                "error": str(e),
                "used_model": candidate_model,
            }

    async def generate_diagram(
        self,
        request: Request,
        chat_api_url: str,
        api_key: str,
        model: str,
        enable_vlm_validation: bool,
        vlm_model: Optional[str],
        vlm_validation_max_retries: Optional[int],
        input_type: str,
        diagram_type: str,
        diagram_style: str,
        language: str,
        email: Optional[str],
        file: Optional[UploadFile],
        text_content: Optional[str],
    ) -> Dict[str, Any]:
        """生成图表"""
        run_dir = self._create_run_dir("paper2drawio", email)
        input_dir = run_dir / "input"

        # 处理输入
        paper_file = ""
        if input_type == "PDF" and file:
            pdf_path = input_dir / (file.filename or "input.pdf")
            content = await file.read()
            pdf_path.write_bytes(content)
            paper_file = str(pdf_path)

        text_input = (text_content or "").strip()
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        text_is_image_path = Path(text_input).suffix.lower() in image_exts if text_input else False
        use_sam3_workflow = text_is_image_path

        workflow_input_type = input_type
        workflow_paper_file = paper_file
        workflow_text_content = text_content or ""

        if input_type == "PDF":
            if not paper_file:
                return {
                    "success": False,
                    "xml_content": "",
                    "file_path": "",
                    "error": "Missing PDF file",
                }
            try:
                markdown_text, _ = await run_mineru_pdf_extract_http(
                    paper_file,
                    str(run_dir),
                )
            except Exception as e:
                log.error(f"[paper2drawio] MinerU extraction failed: {e}")
                return {
                    "success": False,
                    "xml_content": "",
                    "file_path": "",
                    "error": f"MinerU extraction failed: {e}",
                }

            if not markdown_text.strip():
                return {
                    "success": False,
                    "xml_content": "",
                    "file_path": "",
                    "error": "MinerU extraction returned empty markdown",
                }

            workflow_input_type = "TEXT"
            workflow_paper_file = ""
            workflow_text_content = markdown_text

        # SAM3 流程使用平台内置 OCR 服务配置；普通流程沿用用户入参
        request_chat_api_url = chat_api_url
        request_api_key = api_key
        if use_sam3_workflow:
            request_chat_api_url = settings.PAPER2DRAWIO_OCR_API_URL
            request_api_key = settings.PAPER2DRAWIO_OCR_API_KEY

        candidate_models = self._parse_model_candidates(model)
        if len(candidate_models) <= 1:
            result = await self._run_workflow_once(
                request_chat_api_url=request_chat_api_url,
                request_api_key=request_api_key,
                candidate_model=candidate_models[0],
                enable_vlm_validation=enable_vlm_validation,
                vlm_model=vlm_model,
                vlm_validation_max_retries=vlm_validation_max_retries,
                workflow_input_type=workflow_input_type,
                workflow_paper_file=workflow_paper_file,
                workflow_text_content=workflow_text_content,
                diagram_type=diagram_type,
                diagram_style=diagram_style,
                language=language,
                use_sam3_workflow=use_sam3_workflow,
                result_dir=run_dir,
            )
            if result["success"]:
                return result
            return {
                "success": False,
                "xml_content": "",
                "file_path": "",
                "error": result.get("error") or "Failed to generate diagram",
            }

        log.info(f"[paper2drawio] race mode enabled, candidates={candidate_models}")
        tasks = [
            asyncio.create_task(
                self._run_workflow_once(
                    request_chat_api_url=request_chat_api_url,
                    request_api_key=request_api_key,
                    candidate_model=candidate_model,
                    enable_vlm_validation=enable_vlm_validation,
                    vlm_model=vlm_model,
                    vlm_validation_max_retries=vlm_validation_max_retries,
                    workflow_input_type=workflow_input_type,
                    workflow_paper_file=workflow_paper_file,
                    workflow_text_content=workflow_text_content,
                    diagram_type=diagram_type,
                    diagram_style=diagram_style,
                    language=language,
                    use_sam3_workflow=use_sam3_workflow,
                    result_dir=self._model_run_dir(run_dir, candidate_model, idx),
                )
            )
            for idx, candidate_model in enumerate(candidate_models)
        ]

        failures: List[str] = []
        try:
            for future in asyncio.as_completed(tasks):
                result = await future
                if result.get("success"):
                    winner = result.get("used_model", "")
                    log.info(f"[paper2drawio] race winner model={winner}")
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    return result

                used_model = result.get("used_model") or "unknown"
                error = result.get("error") or "unknown error"
                failures.append(f"{used_model}: {error}")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "success": False,
            "xml_content": "",
            "file_path": "",
            "error": "All candidate models failed: " + " | ".join(failures[:6]),
        }

    async def chat_edit(
        self,
        request: Request,
        current_xml: str,
        message: str,
        chat_history: List[Dict[str, str]],
        chat_api_url: str,
        api_key: str,
        model: str,
    ) -> Dict[str, Any]:
        """对话式编辑"""
        current_cells = (
            extract_cells(current_xml)
            if ("<mxfile" in current_xml or "<diagram" in current_xml)
            else current_xml
        )
        state = Paper2DrawioState(
            request=Paper2DrawioRequest(
                chat_api_url=chat_api_url,
                api_key=api_key,
                model=model,
                input_type="TEXT",
                edit_instruction=message,
                chat_history=chat_history,
            ),
            drawio_xml=current_cells,
            text_content=message,
        )

        from dataflow_agent.workflow.registry import RuntimeRegistry

        try:
            async with task_semaphore:
                factory = RuntimeRegistry.get("paper2drawio_semantic")
                builder = factory()
                graph = builder.build()
                final_state = await graph.ainvoke(state)

            raw_xml = (
                final_state.get("drawio_xml", "")
                if isinstance(final_state, dict)
                else (final_state.drawio_xml or "")
            )
            xml_content = wrap_xml(raw_xml) if raw_xml else ""
            return {
                "success": bool(xml_content),
                "xml_content": xml_content,
                "message": "Diagram updated" if xml_content else "",
                "error": None if xml_content else "Failed to update diagram",
            }
        except Exception as e:
            log.error(f"编辑图表失败: {e}")
            return {
                "success": False,
                "xml_content": current_xml,
                "message": "",
                "error": str(e),
            }

    async def export_diagram(
        self,
        request: Request,
        xml_content: str,
        format: str,
        filename: str,
    ) -> Dict[str, Any]:
        """导出图表"""
        run_dir = self._create_run_dir("paper2drawio_export", None)

        if format == "drawio":
            output_path = run_dir / f"{filename}.drawio"
            full_xml = (
                xml_content if "<mxfile" in xml_content else wrap_xml(xml_content)
            )
            output_path.write_text(full_xml, encoding="utf-8")
        else:
            output_path = run_dir / f"{filename}.{format}"
            full_xml = (
                xml_content if "<mxfile" in xml_content else wrap_xml(xml_content)
            )
            output_path.write_text(full_xml, encoding="utf-8")

        return {
            "success": True,
            "file_path": str(output_path),
        }
