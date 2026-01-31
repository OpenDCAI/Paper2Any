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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi_app.schemas import ErrorResponse
from fastapi_app.services.rebuttal.rebuttal_service import (
    rebuttal_service,
    init_llm_client,
    get_llm_client,
    ProcessStatus,
)
from fastapi_app.services.rebuttal.tools import pdf_to_md

router = APIRouter(tags=["paper2rebuttal"])


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


REVIEW_EXTRACT_SYSTEM = """你是一个学术评审解析助手。用户会提供一段从「评审网页」或「评审 PDF」导出的原始文本。
请从中识别出每一条评审意见（可能是审稿人评论、编号问题、Q1/Q2 等），并输出为一个 JSON 数组。
每条意见对应一个对象，格式为：{"id": "review-1", "content": "该条评审的完整原文"}。
id 必须为 review-1, review-2, review-3 ... 连续编号。content 为该条评审的完整内容（以 Markdown 格式输出：保留标题、列表、加粗等结构，便于前端展示），不要截断。
只输出一个合法的 JSON 数组，不要输出其他解释或代码块；content 字段内使用 Markdown 语法。"""


def extract_reviews_with_llm(raw_text: str) -> List[Dict[str, str]]:
    """使用 LLM 从评审网页 PDF 的原始文本中解析出 review-1, review-2 ... 列表。"""
    client = get_llm_client()
    user_prompt = f"""请从下面这段「评审网页/评审 PDF」的原始文本中，提取出每一条独立的评审意见，输出为 JSON 数组。

原始文本：
---
{raw_text}
---

要求：输出仅包含一个 JSON 数组，每个元素为 {{"id": "review-1", "content": "该条完整内容"}}。id 从 review-1 起连续编号。content 以 Markdown 格式输出（标题、列表、加粗等），便于前端渲染。不要输出 ```json 等标记，直接输出数组。"""
    out, _ = client.generate(
        instructions=REVIEW_EXTRACT_SYSTEM,
        input_text=user_prompt,
        temperature=0.2,
        agent_name="review_extractor",
    )
    out = (out or "").strip()
    # 去掉可能的 ```json ... ``` 包裹
    if out.startswith("```"):
        lines = out.split("\n")
        if lines[0].lower().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        out = "\n".join(lines)
    try:
        arr = json.loads(out)
        if not isinstance(arr, list):
            return []
        items = []
        for i, x in enumerate(arr):
            if isinstance(x, dict):
                rid = x.get("id") or f"review-{i + 1}"
                content = x.get("content") or str(x.get("content", ""))
                items.append({"id": rid, "content": content})
            elif isinstance(x, str):
                items.append({"id": f"review-{i + 1}", "content": x})
        return items
    except json.JSONDecodeError:
        return []


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
                md_path = pdf_to_md(tmp_path, out_dir)
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
            # Use same logic as GET /session: check if ALL questions have agent7_output (strategy)
            if session.questions and len(session.questions) > 0:
                all_processed = all(
                    (q.agent7_output or "").strip() for q in session.questions
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
    if 'agent1' in message_lower or 'generating paper summary' in message_lower or 'paper summary' in message_lower:
        return '📄 正在生成论文摘要...'
    elif 'agent2' in message_lower and 'extract' in message_lower:
        return '🔍 正在提取评审问题...'
    elif 'agent2' in message_lower and ('check' in message_lower or 'validat' in message_lower):
        return '✓ 正在验证问题提取结果...'
    elif 'agent3' in message_lower or 'determining search strategy' in message_lower or 'search strategy' in message_lower:
        return '🔎 正在分析问题并确定搜索策略...'
    elif 'searching' in message_lower or ('search' in message_lower and 'query' in message_lower):
        return '📚 正在搜索相关论文...'
    elif 'agent4' in message_lower or 'selecting relevant papers' in message_lower or 'filter' in message_lower:
        return '📑 正在筛选相关论文...'
    elif 'agent5' in message_lower or 'analyzing reference' in message_lower or 'reference paper' in message_lower:
        return '📖 正在分析参考文献内容...'
    elif 'agent6' in message_lower or 'generating rebuttal strategy' in message_lower:
        return '💡 正在生成反驳策略...'
    elif 'agent7' in message_lower or 'optimizing rebuttal strategy' in message_lower or 'check and optimize' in message_lower:
        return '✨ 正在优化反驳策略...'
    elif 'agent8' in message_lower or 'generating rebuttal draft' in message_lower:
        return '📝 正在生成反驳信草稿...'
    elif 'agent9' in message_lower or 'proofreading' in message_lower or 'final version' in message_lower:
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
    model: Optional[str] = Form("gpt-4o-mini"),
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
                    md_path = pdf_to_md(tmp_path, out_dir)
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

        # 所有形式的输入：只要提供了 API 就用 LLM 形式化，否则用规则解析
        use_llm = bool(chat_api_url and api_key)

        reviews: List[Dict[str, str]] = []
        if use_llm and chat_api_url and api_key:
            init_llm_client(api_key=api_key.strip(), chat_api_url=chat_api_url.strip(), model=model or "gpt-4o-mini")
            try:
                reviews = extract_reviews_with_llm(review_text_parsed)
            except Exception as llm_e:
                import traceback
                traceback.print_exc()
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
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    model: str = Form("gpt-5.1"),
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
                print(f"[LOG] Processing {num_questions} questions with {max_workers} parallel workers")
                
                # Process all questions in parallel
                rebuttal_service.process_all_questions_parallel(session_id, max_workers=max_workers)
                
                # Ensure status is set after completion
                session = rebuttal_service.get_session(session_id)
                if session:
                    session.overall_status = ProcessStatus.WAITING_FEEDBACK
                    print(f"[LOG] Analysis completed, status set to WAITING_FEEDBACK")
            except Exception as e:
                print(f"[ERROR] Background analysis failed: {e}")
                import traceback
                traceback.print_exc()
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
    
    # Only consider "all processed" when every question has strategy (agent7_output)
    all_questions_processed = bool(
        session.questions
        and all((q.agent7_output or "").strip() for q in session.questions)
    )
    return {
        "session_id": session_id,
        "questions": [
            {
                "question_id": q.question_id,
                "question_text": q.question_text,
                "strategy": q.agent7_output or "",
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
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    model: str = Form("gpt-5.1"),
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
            "strategy": q_state.agent7_output,
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
    session_id: str = Form(...),
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    model: str = Form("gpt-5.1"),
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
            raise HTTPException(
                status_code=400, 
                detail=f"还有 {len(unsatisfied)} 个问题未标记为满意。请先处理所有问题。"
            )
        
        # Initialize LLM client
        init_llm_client(api_key=api_key.strip(), chat_api_url=chat_api_url.strip(), model=model)
        
        # Generate final rebuttal
        final_text = rebuttal_service.generate_final_rebuttal(session_id)
        
        return {
            "status": "success",
            "final_rebuttal": final_text,
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
