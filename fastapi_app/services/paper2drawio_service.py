"""
Paper2Drawio Service 层
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import UploadFile, Request

from dataflow_agent.state import Paper2DrawioState, Paper2DrawioRequest
from dataflow_agent.toolkits.drawio_tools import wrap_xml, extract_cells
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

BASE_OUTPUT_DIR = Path("outputs").resolve()
task_semaphore = asyncio.Semaphore(2)


class Paper2DrawioService:
    """Paper2Drawio 业务服务"""

    def _create_run_dir(self, prefix: str, email: Optional[str]) -> Path:
        """创建运行目录"""
        ts = int(time.time())
        run_dir = BASE_OUTPUT_DIR / prefix / str(ts)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "input").mkdir(exist_ok=True)
        return run_dir

    async def generate_diagram(
        self,
        request: Request,
        chat_api_url: str,
        api_key: str,
        model: str,
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

        # 构造 State
        state = Paper2DrawioState(
            request=Paper2DrawioRequest(
                language=language,
                chat_api_url=chat_api_url,
                api_key=api_key,
                model=model,
                input_type=input_type,
                diagram_type=diagram_type,
                diagram_style=diagram_style,
            ),
            paper_file=paper_file,
            text_content=text_content or "",
            result_path=str(run_dir),
        )

        # 执行 workflow
        from dataflow_agent.workflow.registry import RuntimeRegistry

        try:
            async with task_semaphore:
                factory = RuntimeRegistry.get("paper2drawio")
                builder = factory()
                graph = builder.build()
                final_state = await graph.ainvoke(state)

            raw_xml = final_state.get("drawio_xml", "") if isinstance(final_state, dict) else (final_state.drawio_xml or "")
            output_path = final_state.get("output_xml_path", "") if isinstance(final_state, dict) else (final_state.output_xml_path or "")

            # 包装 XML 为完整的 draw.io 格式
            xml_content = wrap_xml(raw_xml) if raw_xml else ""

            return {
                "success": bool(xml_content),
                "xml_content": xml_content,
                "file_path": output_path,
                "error": None if xml_content else "Failed to generate diagram",
            }
        except Exception as e:
            log.error(f"生成图表失败: {e}")
            return {
                "success": False,
                "xml_content": "",
                "file_path": "",
                "error": str(e),
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
                factory = RuntimeRegistry.get("paper2drawio")
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
