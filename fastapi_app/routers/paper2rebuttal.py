from __future__ import annotations

import os
import re
import uuid
import shutil
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, Body
from fastapi.responses import StreamingResponse
from fastapi_app.schemas import ErrorResponse
from fastapi_app.services.rebuttal_service import (
    rebuttal_service,
    init_llm_client,
    get_llm_client,
    ProcessStatus,
    ReviewCheckAgent,
)
from dataflow_agent.toolkits.rebuttal import pdf_to_md, _read_text
from fastapi_app.dependencies import get_optional_user, AuthUser
from fastapi_app.utils import _to_outputs_url
from dataflow_agent.utils import get_project_root
from dataflow_agent.logger import get_logger

router = APIRouter(tags=["paper2rebuttal"])
PROJECT_ROOT = get_project_root()
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
log = get_logger(__name__)


def _resolve_user_dir(user: Optional[AuthUser], email: Optional[str]) -> str:
    if user:
        return user.email or user.id
    if email:
        return email
    return "default"


def _get_session_dir(session_id: str) -> str:
    session = rebuttal_service.get_session(session_id)
    if not session:
        session = rebuttal_service.restore_session_from_disk(session_id)
    if session and session.session_dir:
        return session.session_dir
    return str(PROJECT_ROOT / "rebuttal_sessions" / session_id)


def _write_session_meta(session_dir: str, user_dir: str, email: Optional[str]) -> None:
    try:
        meta_path = os.path.join(session_dir, "session_meta.json")
        meta = {
            "user_dir": user_dir,
            "email": email or "",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _export_rebuttal_outputs(session_id: str, user_dir: str) -> Path:
    session_dir = _get_session_dir(session_id)
    output_dir = OUTPUTS_ROOT / user_dir / "paper2rebuttal" / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        (os.path.join(session_dir, "paper.pdf"), "paper.pdf"),
        (os.path.join(session_dir, "paper.md"), "paper.md"),
        (os.path.join(session_dir, "review.txt"), "review.txt"),
        (os.path.join(session_dir, "session_meta.json"), "session_meta.json"),
        (os.path.join(session_dir, "logs", "summary.md"), "summary.md"),
        (os.path.join(session_dir, "logs", "session_summary.json"), "session_summary.json"),
        (os.path.join(session_dir, "logs", "token_usage.json"), "token_usage.json"),
    ]

    for src, name in candidates:
        if os.path.exists(src) and os.path.isfile(src):
            dst = output_dir / name
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

    final_txt = os.path.join(session_dir, "logs", "final_rebuttal.txt")
    if os.path.exists(final_txt) and os.path.isfile(final_txt):
        try:
            with open(final_txt, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            with open(output_dir / "final_rebuttal.md", "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass

    return output_dir


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_file_content(file_obj) -> tuple[Optional[str], Optional[bytes]]:
    """Read file content from UploadFile"""
    if file_obj is None:
        return None, None
    
    # Read file content
    content = file_obj.file.read()
    return file_obj.filename, content


def decode_review_bytes(data: bytes) -> str:
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
    for enc in encodings:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode('utf-8', errors='replace')


def parse_review_into_items(review_text: str) -> List[Dict[str, str]]:
    """将评审原文解析为 review-1, review-2 ... 列表，便于前端展示"""
    text = (review_text or "").strip()
    if not text:
        return []
    items: List[Dict[str, str]] = []
    # 匹配多种常见编号格式: Review 1 / Q1. / [q1] / 1. / (1) / 1)
    pattern = re.compile(
        r'(?:^|\n)\s*(?:Review\s*#?\s*)?(\d+)[\.\)\]:\s]*|'
        r'(?:^|\n)\s*\[?\s*q\s*(\d+)\s*\]?\s*[\.\)\]:\s]*|'
        r'(?:^|\n)\s*\(\s*(\d+)\s*\)\s*',
        re.IGNORECASE
    )
    last_end = 0
    for m in pattern.finditer(text):
        num = next((g for g in m.groups() if g is not None), None)
        if num is None:
            continue
        num = int(num)
        start = m.start()
        if start > last_end:
            prev_content = text[last_end:start].strip()
            if prev_content and not items:
                items.append({"id": "review-0", "content": prev_content})
        content_start = m.end()
        next_m = pattern.search(text, content_start)
        content_end = next_m.start() if next_m else len(text)
        content = text[content_start:content_end].strip()
        items.append({"id": f"review-{num}", "content": content})
        last_end = content_end
    if not items and text:
        items.append({"id": "review-1", "content": text})
    return items


def save_uploaded_files(
    pdf_file: UploadFile,
    session_id: str,
    review_file: Optional[UploadFile] = None,
    review_text: Optional[str] = None,
) -> tuple[str, str, str]:
    """保存论文 PDF 与评审内容。评审可为：上传文件（PDF/txt/md）或直接文本。"""
    from dataflow_agent.utils import get_project_root
    project_root = get_project_root()
    session_dir = os.path.join(project_root, "rebuttal_sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    pdf_save_path = os.path.join(session_dir, "paper.pdf")
    review_save_path = os.path.join(session_dir, "review.txt")
    
    pdf_filename, pdf_data = read_file_content(pdf_file)
    if pdf_data is None:
        raise ValueError("PDF file upload failed")
    with open(pdf_save_path, "wb") as f:
        f.write(pdf_data)
    
    if review_text is not None and review_text.strip():
        review_final = review_text.strip()
    elif review_file is not None:
        review_filename, review_data = read_file_content(review_file)
        if review_data is None:
            raise ValueError("Review file upload failed")
        fn = (review_filename or "").lower()
        if fn.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(review_data)
                tmp_path = tmp.name
            try:
                out_dir = os.path.dirname(tmp_path)
                md_path = pdf_to_md(tmp_path, out_dir, parser="mineru")
                if not (md_path and os.path.isfile(md_path)):
                    md_path = pdf_to_md(tmp_path, out_dir, parser="docling")
                if md_path and os.path.isfile(md_path):
                    with open(md_path, "r", encoding="utf-8", errors="replace") as f:
                        review_final = f.read()
                else:
                    review_final = decode_review_bytes(review_data)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        else:
            review_final = decode_review_bytes(review_data)
    else:
        raise ValueError("请提供评审文件或评审文本（review_file 或 review_text）")
    
    with open(review_save_path, "w", encoding="utf-8") as f:
        f.write(review_final)
    return pdf_save_path, review_save_path, review_final


async def progress_stream_generator(session_id: str):
    """Generator function for streaming progress updates"""
    try:
        # Send initial message
        yield f"data: {json.dumps({'type': 'progress', 'message': '🚀 开始分析...', 'step': 'init'})}\n\n"
        
        # Wait a bit for session to be created
        await asyncio.sleep(0.5)
        
        # Monitor log collector for updates
        last_log_count = 0
        max_iterations = 36000  # Safety limit (3 hours max: 36000 * 0.3s = 10800s)
        iteration = 0
        last_status = None
        
        while iteration < max_iterations:
            # Get session to access log collector
            session = rebuttal_service.get_session(session_id)
            if not session:
                # Try to restore from disk
                session = rebuttal_service.restore_session_from_disk(session_id)
                if not session:
                    await asyncio.sleep(1)
                    iteration += 1
                    continue
            
            # Check log collector for new messages
            if session.log_collector:
                current_logs = session.log_collector.get_all()
                log_lines = current_logs.split('\n') if current_logs else []
                
                # Send new log entries
                if len(log_lines) > last_log_count:
                    for i in range(last_log_count, len(log_lines)):
                        log_line = log_lines[i].strip()
                        if log_line:
                            # Extract meaningful message
                            message = log_line
                            if '[' in log_line and ']' in log_line:
                                # Try to extract the message part after timestamp
                                parts = log_line.split(']', 1)
                                if len(parts) > 1:
                                    message = parts[1].strip()
                            
                            # Map to user-friendly messages
                            friendly_message = map_progress_message(message)
                            
                            yield f"data: {json.dumps({'type': 'progress', 'message': friendly_message})}\n\n"
                    
                    last_log_count = len(log_lines)
            
            # Check status changes
            current_status = session.overall_status
            if current_status != last_status:
                if current_status == ProcessStatus.WAITING_FEEDBACK:
                    yield f"data: {json.dumps({'type': 'complete', 'message': '✅ 分析完成！所有问题已处理完毕'})}\n\n"
                    break
                elif current_status == ProcessStatus.COMPLETED:
                    yield f"data: {json.dumps({'type': 'complete', 'message': '✅ 所有任务已完成！'})}\n\n"
                    break
                elif current_status == ProcessStatus.ERROR:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'❌ 处理出错: {session.progress_message}'})}\n\n"
                    break
                last_status = current_status
            
            # Also check if all questions are processed (fallback check)
            # Use same logic as GET /session: check if ALL questions have strategy_review_output (strategy)
            if session.questions and len(session.questions) > 0:
                all_processed = all(
                    (q.strategy_review_output or "").strip() for q in session.questions
                )
                if all_processed and current_status != ProcessStatus.WAITING_FEEDBACK:
                    # Set status if not already set
                    session.overall_status = ProcessStatus.WAITING_FEEDBACK
                    yield f"data: {json.dumps({'type': 'complete', 'message': '✅ 分析完成！所有问题已处理完毕'})}\n\n"
                    break
            
            await asyncio.sleep(0.3)  # Check every 300ms
            iteration += 1
        
        if iteration >= max_iterations:
            yield f"data: {json.dumps({'type': 'timeout', 'message': '⏱️ 处理超过3小时，请检查后台日志或稍后刷新页面'})}\n\n"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': f'❌ 错误: {str(e)}'})}\n\n"


def map_progress_message(raw_message: str) -> str:
    """Map raw log messages to user-friendly progress messages"""
    message_lower = raw_message.lower()
    
    # Agent messages
    if 'agent-paper_summary' in message_lower or 'generating paper summary' in message_lower or 'paper summary' in message_lower:
        return '📄 正在生成论文摘要...'
    elif 'agent-issue_extract' in message_lower and 'extract' in message_lower:
        return '🔍 正在提取评审问题...'
    elif 'agent-issue_check' in message_lower and ('check' in message_lower or 'validat' in message_lower):
        return '✓ 正在验证问题提取结果...'
    elif 'agent-paper_search' in message_lower or 'determining search strategy' in message_lower or 'search strategy' in message_lower:
        return '🔎 正在分析问题并确定搜索策略...'
    elif 'searching' in message_lower or ('search' in message_lower and 'query' in message_lower):
        return '📚 正在搜索相关论文...'
    elif 'agent-reference_filter' in message_lower or 'selecting relevant papers' in message_lower or 'filter' in message_lower:
        return '📑 正在筛选相关论文...'
    elif 'agent-reference_analyze' in message_lower or 'analyzing reference' in message_lower or 'reference paper' in message_lower:
        return '📖 正在分析参考文献内容...'
    elif 'agent-strategy_gen' in message_lower or 'generating rebuttal strategy' in message_lower:
        return '💡 正在生成反驳策略...'
    elif 'agent-strategy_review' in message_lower or 'optimizing rebuttal strategy' in message_lower or 'check and optimize' in message_lower:
        return '✨ 正在优化反驳策略...'
    elif 'agent-rebuttal_draft' in message_lower or 'generating rebuttal draft' in message_lower:
        return '📝 正在生成反驳信草稿...'
    elif 'agent-rebuttal_final' in message_lower or 'proofreading' in message_lower or 'final version' in message_lower:
        return '🔍 正在校对并生成最终版本...'
    
    # Process messages
    elif 'converting pdf' in message_lower or 'pdf to markdown' in message_lower or 'pdf conversion' in message_lower:
        return '🔄 正在转换PDF为Markdown格式...'
    elif 'parsing question' in message_lower or 'extract questions' in message_lower:
        return '📋 正在解析问题列表...'
    elif 'processing question' in message_lower or 'process question' in message_lower:
        return '⚙️ 正在处理问题...'
    elif 'downloading' in message_lower and 'paper' in message_lower:
        return '⬇️ 正在下载论文...'
    elif 'analyzing' in message_lower and 'paper' in message_lower and 'reference' not in message_lower:
        return '🔬 正在分析论文内容...'
    elif 'complete' in message_lower or 'finished' in message_lower or 'waiting for feedback' in message_lower:
        return '✅ 处理完成！'
    elif 'error' in message_lower or 'failed' in message_lower:
        return f'❌ 错误: {raw_message}'
    elif 'found' in message_lower and 'papers' in message_lower:
        return '📚 已找到相关论文'
    elif 'selected' in message_lower and 'papers' in message_lower:
        return '✓ 已筛选出相关论文'
    
    # Default: return cleaned message
    cleaned = raw_message.replace('[Progress]', '').replace('[Q', '问题').replace('[Parallel]', '').strip()
    # Remove common prefixes
    for prefix in ['[INFO]', '[DEBUG]', '[SUCCESS]', '[WARNING]', '[ERROR]']:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned if cleaned else raw_message


@router.post(
    "/paper2rebuttal/parse-review",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def parse_review(
    review_file: Optional[UploadFile] = File(None),
    review_text: Optional[str] = Form(None),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    model: Optional[str] = Form("kimi-k2.5"),
):
    """解析评审内容：支持上传 PDF/txt/md 或直接传入文本。
    所有形式的输入都会先得到原始文本（PDF 用 docling 转文本），再统一做形式化：
    若提供了 chat_api_url 与 api_key，则用 LLM 提取 review-1, review-2...；否则用规则解析（按 Review 1 / Q1 等分段）。
    返回形式化的 review 列表供用户 check。"""
    try:
        if review_file is not None:
            filename, data = read_file_content(review_file)
            if data is None:
                raise ValueError("评审文件读取失败")
            fn = (filename or "").lower()
            if fn.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    out_dir = os.path.dirname(tmp_path)
                    md_path = pdf_to_md(tmp_path, out_dir, parser="mineru")
                    if not (md_path and os.path.isfile(md_path)):
                        md_path = pdf_to_md(tmp_path, out_dir, parser="docling")
                    if md_path and os.path.isfile(md_path):
                        with open(md_path, "r", encoding="utf-8", errors="replace") as f:
                            review_text_parsed = f.read()
                    else:
                        review_text_parsed = decode_review_bytes(data)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                review_text_parsed = decode_review_bytes(data)
        elif review_text and review_text.strip():
            review_text_parsed = review_text.strip()
        else:
            raise ValueError("请上传评审文件或粘贴评审文本")

        # 所有形式的输入：只要提供了 API 就用 ReviewCheckAgent (LLM) 形式化，否则用规则解析
        use_llm = bool(chat_api_url and api_key)

        reviews: List[Dict[str, str]] = []
        if use_llm and chat_api_url and api_key:
            init_llm_client(api_key=api_key.strip(), chat_api_url=chat_api_url.strip(), model=model or "kimi-k2.5")
            try:
                # Use ReviewCheckAgent for review extraction
                review_agent = ReviewCheckAgent(review_text_parsed, temperature=0.2, log_dir=None)
                reviews = review_agent.run()
            except Exception as llm_e:
                import traceback
                traceback.print_exc()
                log.warning(f"[parse_review] ReviewCheckAgent failed: {llm_e}, falling back to rule-based parsing")
                reviews = []
        if not reviews:
            reviews = parse_review_into_items(review_text_parsed)
        return {"review_text": review_text_parsed, "reviews": reviews}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/paper2rebuttal/start",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def start_analysis(
    request: Request,
    pdf_file: UploadFile = File(...),
    review_file: Optional[UploadFile] = File(None),
    review_text: Optional[str] = Form(None),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    model: str = Form("deepseek-v3.1"),
    email: Optional[str] = Form(None),
    user: Optional[AuthUser] = Depends(get_optional_user),
):
    """Start rebuttal analysis. 评审可为：上传文件（PDF/txt/md）或直接文本 review_text。"""
    import threading
    
    if review_file is None and not (review_text and review_text.strip()):
        raise HTTPException(status_code=400, detail="请提供评审文件或评审文本（review_file 或 review_text）")
    
    try:
        # Initialize LLM client
        init_llm_client(api_key=api_key.strip(), chat_api_url=chat_api_url.strip(), model=model)
        
        # Create session
        session_id = str(uuid.uuid4())[:8]
        pdf_path, review_path, _ = save_uploaded_files(
            pdf_file, session_id, review_file=review_file, review_text=review_text
        )
        session = rebuttal_service.create_session(session_id, pdf_path, review_path)
        user_dir = _resolve_user_dir(user, email)
        _write_session_meta(session.session_dir, user_dir, email)

        # Generate review-check logs (ReviewCheckAgent) for export zip
        try:
            logs_dir = os.path.join(session.session_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            review_raw = _read_text(review_path)
            review_agent = ReviewCheckAgent(review_raw, temperature=0.2, log_dir=logs_dir)
            reviews = review_agent.run()
            with open(os.path.join(logs_dir, "agent-review_check_reviews.json"), "w", encoding="utf-8") as f:
                json.dump(reviews, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"[start_analysis] ReviewCheckAgent review-check log failed: {e}")
        
        # Run analysis in background thread
        def run_analysis():
            try:
                # Run initial analysis
                rebuttal_service.run_initial_analysis(session_id)
                
                # Get session to determine number of questions
                session_obj = rebuttal_service.get_session(session_id)
                num_questions = len(session_obj.questions) if session_obj and session_obj.questions else 3
                # Use all questions as workers for maximum parallelism
                max_workers = num_questions
                log.info(f"[LOG] Processing {num_questions} questions with {max_workers} parallel workers")
                
                # Process all questions in parallel
                rebuttal_service.process_all_questions_parallel(session_id, max_workers=max_workers)
                
                # Ensure status is set after completion
                session = rebuttal_service.get_session(session_id)
                if session:
                    session.overall_status = ProcessStatus.WAITING_FEEDBACK
                    log.info("[LOG] Analysis completed, status set to WAITING_FEEDBACK")
                try:
                    _export_rebuttal_outputs(session_id, user_dir)
                except Exception:
                    pass
            except Exception as e:
                log.exception(f"[ERROR] Background analysis failed: {e}")
                # Set error status
                session = rebuttal_service.get_session(session_id)
                if session:
                    session.overall_status = ProcessStatus.ERROR
                    session.progress_message = f"Error: {str(e)}"
        
        # Start background thread
        analysis_thread = threading.Thread(target=run_analysis, daemon=True)
        analysis_thread.start()
        
        # Return immediately with session_id
        return {
            "status": "processing",
            "session_id": session_id,
            "message": "分析已开始，请通过进度端点获取实时更新",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper2rebuttal/progress/{session_id}")
async def stream_progress(session_id: str):
    """Stream progress updates for a session using Server-Sent Events"""
    return StreamingResponse(
        progress_stream_generator(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get(
    "/paper2rebuttal/session/{session_id}",
    response_model=Dict[str, Any],
    responses={404: {"model": ErrorResponse}},
)
async def get_session(session_id: str):
    """Get session information"""
    session = rebuttal_service.get_session(session_id)
    if not session:
        # Try to restore from disk
        session = rebuttal_service.restore_session_from_disk(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Only consider "all processed" when every question has strategy (strategy_review_output)
    all_questions_processed = bool(
        session.questions
        and all((q.strategy_review_output or "").strip() for q in session.questions)
    )
    return {
        "session_id": session_id,
        "questions": [
            {
                "question_id": q.question_id,
                "question_text": q.question_text,
                "strategy": q.strategy_review_output or "",
                "strategy_text": q.strategy_text or "",
                "todo_list": q.todo_list or [],
                "draft_response": q.draft_response or "",
                "revision_count": q.revision_count,
                "is_satisfied": q.is_satisfied,
                "feedback_history": q.feedback_history,
                "searched_papers": q.searched_papers or [],
                "selected_papers": q.selected_papers or [],
                "analyzed_papers": q.analyzed_papers or [],
                "history": q.history or [],
            }
            for q in session.questions
        ],
        "final_rebuttal": session.final_rebuttal or "",
        "all_questions_processed": all_questions_processed,
    }


@router.post(
    "/paper2rebuttal/revise",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def revise_strategy(
    session_id: str = Form(...),
    question_idx: int = Form(...),
    feedback: str = Form(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    model: str = Form("deepseek-v3.1"),
):
    """Revise strategy based on feedback"""
    try:
        # Initialize LLM client
        init_llm_client(api_key=api_key.strip(), chat_api_url=chat_api_url.strip(), model=model)
        
        # Revise strategy
        q_state = rebuttal_service.revise_with_feedback(
            session_id, question_idx, feedback.strip()
        )
        
        return {
            "status": "success",
            "strategy": q_state.strategy_review_output,
            "strategy_text": q_state.strategy_text or "",
            "todo_list": q_state.todo_list or [],
            "draft_response": q_state.draft_response or "",
            "revision_count": q_state.revision_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/paper2rebuttal/mark-satisfied",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}},
)
async def mark_question_satisfied(
    session_id: str = Form(...),
    question_idx: int = Form(...),
):
    """Mark a question as satisfied"""
    try:
        q_state = rebuttal_service.mark_question_satisfied(session_id, question_idx)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/paper2rebuttal/generate-final",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def generate_final_rebuttal(
    request: Request,
    session_id: str = Form(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    model: str = Form("deepseek-v3.1"),
    email: Optional[str] = Form(None),
    user: Optional[AuthUser] = Depends(get_optional_user),
):
    """Generate final rebuttal"""
    try:
        # Check session first
        session = rebuttal_service.get_session(session_id)
        if not session:
            session = rebuttal_service.restore_session_from_disk(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Check if all questions are satisfied
        unsatisfied = [q for q in session.questions if not q.is_satisfied]
        if unsatisfied:
            unsatisfied_ids = [q.question_id for q in unsatisfied]
            unsatisfied_indices = [i for i, q in enumerate(session.questions) if not q.is_satisfied]
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": f"还有 {len(unsatisfied)} 个问题未标记为满意。请先处理所有问题。",
                    "unsatisfied_question_ids": unsatisfied_ids,
                    "unsatisfied_question_indices": unsatisfied_indices,
                }
            )
        
        # Initialize LLM client
        init_llm_client(api_key=api_key.strip(), chat_api_url=chat_api_url.strip(), model=model)
        
        # Generate final rebuttal
        final_text = rebuttal_service.generate_final_rebuttal(session_id)

        user_dir = _resolve_user_dir(user, email)
        output_dir = _export_rebuttal_outputs(session_id, user_dir)
        export_url = _to_outputs_url(str(output_dir), request) if request else str(output_dir)
        
        return {
            "status": "success",
            "final_rebuttal": final_text,
            "export_dir": export_url,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/paper2rebuttal/sessions",
    response_model=Dict[str, Any],
)
async def list_sessions():
    """List all active sessions"""
    sessions = rebuttal_service.list_active_sessions()
    return {
        "sessions": sessions,
    }


@router.get(
    "/paper2rebuttal/history",
    response_model=Dict[str, Any],
)
async def list_rebuttal_history(
    request: Request,
    email: Optional[str] = None,
    user: Optional[AuthUser] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """List saved rebuttal sessions for the current user."""
    user_dir = _resolve_user_dir(user, email)
    base_dir = OUTPUTS_ROOT / user_dir / "paper2rebuttal"

    if not base_dir.exists():
        return {"success": True, "sessions": []}

    sessions: List[Dict[str, Any]] = []
    for session_dir in base_dir.iterdir():
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        meta = _safe_read_json(session_dir / "session_meta.json")
        summary = _safe_read_json(session_dir / "session_summary.json")

        summary_path = session_dir / "session_summary.json"
        updated_ts = summary_path.stat().st_mtime if summary_path.exists() else session_dir.stat().st_mtime
        created_ts = session_dir.stat().st_mtime

        created_at = meta.get("timestamp") or summary.get("timestamp")
        if not created_at:
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_ts))
        updated_at = summary.get("timestamp") or meta.get("timestamp")
        if not updated_at:
            updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated_ts))

        questions = summary.get("questions") or []
        total_questions = summary.get("total_questions") or len(questions)
        processed_questions = sum(
            1
            for q in questions
            if (q.get("final_strategy") or q.get("strategy_text") or q.get("draft_response"))
        )
        satisfied_questions = sum(1 for q in questions if q.get("is_satisfied"))

        has_final = (session_dir / "final_rebuttal.md").exists()
        status = "completed" if has_final else ("ready" if questions else "processing")

        zip_files = list(session_dir.glob("*.zip"))
        zip_url = _to_outputs_url(str(zip_files[0]), request) if zip_files else ""

        sessions.append({
            "session_id": session_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "status": status,
            "total_questions": total_questions,
            "processed_questions": processed_questions,
            "satisfied_questions": satisfied_questions,
            "has_final": has_final,
            "export_dir": _to_outputs_url(str(session_dir), request),
            "zip_url": zip_url,
            "_updated_ts": updated_ts,
        })

    sessions.sort(key=lambda x: x.get("_updated_ts", 0), reverse=True)
    for item in sessions:
        item.pop("_updated_ts", None)

    return {"success": True, "sessions": sessions}


@router.get(
    "/paper2rebuttal/summary/{session_id}",
    responses={404: {"model": ErrorResponse}},
)
async def get_summary_markdown(session_id: str):
    """Get markdown summary of the session"""
    session = rebuttal_service.get_session(session_id)
    if not session:
        session = rebuttal_service.restore_session_from_disk(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    markdown = rebuttal_service.generate_summary_markdown(session_id)
    
    return {
        "session_id": session_id,
        "markdown": markdown,
    }


@router.post(
    "/paper2rebuttal/export-zip",
    response_model=Dict[str, Any],
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def export_rebuttal_zip(
    request: Request,
    session_id: str = Body(..., embed=True),
    email: Optional[str] = Body(None, embed=True),
    include_root_dir: bool = Body(True, embed=True),
    user: Optional[AuthUser] = Depends(get_optional_user),
):
    """Export a rebuttal session into a zip archive."""
    try:
        session_dir = _get_session_dir(session_id)
        if not os.path.isdir(session_dir):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        user_dir = _resolve_user_dir(user, email)
        output_dir = _export_rebuttal_outputs(session_id, user_dir)

        zip_name = f"paper2rebuttal_{session_id}.zip"
        zip_path = output_dir / zip_name

        def _safe_filename(s: str, max_len: int = 80) -> str:
            s = re.sub(r'[<>:"/\\|?*]', "_", (s or "").strip())
            return s[:max_len].strip() or "paper"

        count = 0
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(session_dir):
                for name in files:
                    abs_path = os.path.join(root, name)
                    rel_path = os.path.relpath(abs_path, session_dir)
                    arcname = f"{session_id}/{rel_path}" if include_root_dir else rel_path
                    try:
                        zf.write(abs_path, arcname)
                        count += 1
                    except Exception:
                        continue

            session = rebuttal_service.get_session(session_id) or rebuttal_service.restore_session_from_disk(session_id)
            if session and getattr(session, "questions", None):
                ref_dir = f"{session_id}/reference_papers" if include_root_dir else "reference_papers"
                for q in session.questions:
                    analyzed = getattr(q, "analyzed_papers", None) or []
                    for i, paper in enumerate(analyzed):
                        title = paper.get("title") or "Untitled"
                        fname = _safe_filename(title)
                        arcname = f"{ref_dir}/Q{q.question_id}_{i + 1:02d}_{fname}.md"
                        lines = [
                            f"# {title}",
                            "",
                        ]
                        if paper.get("authors"):
                            lines.append(f"**Authors:** {', '.join(paper['authors'][:5])}{' et al.' if len(paper.get('authors', [])) > 5 else ''}")
                            lines.append("")
                        if paper.get("abs_url"):
                            lines.append(f"**Link:** {paper['abs_url']}")
                            lines.append("")
                        if paper.get("abstract"):
                            lines.append("## Abstract")
                            lines.append("")
                            lines.append(paper["abstract"])
                            lines.append("")
                        if paper.get("analysis"):
                            lines.append("## Summary (for rebuttal)")
                            lines.append("")
                            lines.append(paper["analysis"])
                        content = "\n".join(lines)
                        try:
                            zf.writestr(arcname, content.encode("utf-8"))
                            count += 1
                        except Exception:
                            continue

        if not zip_path.exists():
            raise HTTPException(status_code=500, detail="Failed to create zip")

        return {
            "success": True,
            "zip_path": _to_outputs_url(str(zip_path), request),
            "count": count,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
