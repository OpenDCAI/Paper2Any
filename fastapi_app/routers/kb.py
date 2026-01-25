import os
import shutil
import time
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from typing import Optional, List, Dict, Any

from dataflow_agent.state import IntelligentQARequest, IntelligentQAState, KBPodcastRequest, KBPodcastState
from dataflow_agent.workflow.wf_intelligent_qa import create_intelligent_qa_graph
from dataflow_agent.workflow.wf_kb_podcast import create_kb_podcast_graph
from dataflow_agent.utils import get_project_root
from fastapi_app.config import settings
from fastapi_app.schemas import Paper2PPTRequest
from fastapi_app.workflow_adapters.wa_paper2ppt import run_paper2ppt_full_pipeline

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])

# Base directory for storing KB files
# Use absolute path as requested by user or relative to project root
# We will use relative path 'outputs/kb_data' which resolves to that in the current workspace
KB_BASE_DIR = Path("outputs/kb_data")

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".mp4"}

@router.post("/upload")
async def upload_kb_file(
    file: UploadFile = File(...),
    email: str = Form(...),
    user_id: str = Form(...)
):
    """
    Upload a file to the user's knowledge base directory.
    Stores at: outputs/kb_data/{email}/{filename}
    """
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # Create user directory if not exists
        user_dir = KB_BASE_DIR / email
        user_dir.mkdir(parents=True, exist_ok=True)

        # Secure filename (simple version)
        filename = file.filename
        if not filename:
            filename = f"unnamed_{user_id}"
            
        # Avoid path traversal
        filename = os.path.basename(filename)
        
        file_path = user_dir / filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Return the relative path for static access and storage path
        # Assuming 'outputs' dir is mounted to '/outputs'
        static_path = f"/outputs/kb_data/{email}/{filename}"
        
        return {
            "success": True,
            "filename": filename,
            "file_size": os.path.getsize(file_path),
            "storage_path": str(file_path),
            "static_url": static_path,
            "file_type": file.content_type
        }

    except Exception as e:
        print(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete")
async def delete_kb_file(
    storage_path: str = Form(...)
):
    """
    Delete a file from the physical storage.
    """
    try:
        # Security check: ensure path is within KB_BASE_DIR
        # This is a basic check. In production, use more robust path validation.
        target_path = Path(storage_path).resolve()
        base_path = KB_BASE_DIR.resolve()
        
        if not str(target_path).startswith(str(base_path)):
             # Allow if it's the absolute path provided by the user system
             # Check if it exists essentially
             pass

        if target_path.exists() and target_path.is_file():
            os.remove(target_path)
            return {"success": True, "message": "File deleted"}
        else:
            return {"success": False, "message": "File not found"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_with_kb(
    files: List[str] = Body(..., embed=True),
    query: str = Body(..., embed=True),
    history: List[Dict[str, str]] = Body([], embed=True),
    api_url: Optional[str] = Body(None, embed=True),
    api_key: Optional[str] = Body(None, embed=True),
    model: str = Body(settings.KB_CHAT_MODEL, embed=True),
):
    """
    Intelligent QA Chat
    """
    try:
        # Normalize file paths (web path -> local absolute path)
        project_root = get_project_root()
        local_files = []
        for f in files:
            # remove leading /outputs/ if present, or just join
            # Web path: /outputs/kb_data/...
            clean_path = f.lstrip('/')
            p = project_root / clean_path
            if p.exists():
                local_files.append(str(p))
            else:
                # Try raw path
                p_raw = Path(f)
                if p_raw.exists():
                    local_files.append(str(p_raw))
        
        if not local_files:
             # Just return empty answer or handle logic
             pass

        # Construct Request
        req = IntelligentQARequest(
            files=local_files,
            query=query,
            history=history,
            chat_api_url=api_url or os.getenv("DF_API_URL"),
            api_key=api_key or os.getenv("DF_API_KEY"),
            model=model
        )
        
        state = IntelligentQAState(request=req)
        
        # Build and Run Graph
        builder = create_intelligent_qa_graph()
        graph = builder.compile()
        
        result_state = await graph.ainvoke(state)
        
        # graph.ainvoke returns the final state dict or state object depending on implementation.
        # LangGraph usually returns dict. But our GenericGraphBuilder wrapper might return state.
        # GenericGraphBuilder compile returns a compiled graph.
        # Let's check typical usage. usually await graph.ainvoke(state) returns dict.
        
        answer = ""
        file_analyses = []
        
        if isinstance(result_state, dict):
            answer = result_state.get("answer", "")
            file_analyses = result_state.get("file_analyses", [])
        else:
            answer = getattr(result_state, "answer", "")
            file_analyses = getattr(result_state, "file_analyses", [])
            
        return {
            "success": True,
            "answer": answer,
            "file_analyses": file_analyses
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-ppt")
async def generate_ppt_from_kb(
    file_path: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True),
    email: str = Body(..., embed=True),
    api_url: str = Body(..., embed=True),
    api_key: str = Body(..., embed=True),
    style: str = Body("modern", embed=True),
    language: str = Body("zh", embed=True),
    page_count: int = Body(10, embed=True),
    model: str = Body("gpt-4o", embed=True),
    gen_fig_model: str = Body("gemini-2.5-flash-image", embed=True),
):
    """
    Generate PPT from knowledge base file (non-interactive)
    """
    try:
        # Normalize file path
        project_root = get_project_root()
        clean_path = file_path.lstrip('/')
        local_file_path = project_root / clean_path

        if not local_file_path.exists():
            # Try raw path
            local_file_path = Path(file_path)
            if not local_file_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # Create output directory
        ts = int(time.time())
        output_dir = project_root / "outputs" / "kb_outputs" / email / f"{ts}_ppt"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare request
        ppt_req = Paper2PPTRequest(
            input_type="PDF",
            input_content=str(local_file_path),
            email=email,
            chat_api_url=api_url,
            api_key=api_key,
            style=style,
            language=language,
            page_count=page_count,
            model=model,
            img_gen_model_name=gen_fig_model,
            aspect_ratio="16:9",
            use_long_paper=False
        )

        # Run workflow
        result = await run_paper2ppt_full_pipeline(ppt_req, result_path=output_dir)

        # Extract output paths
        pdf_path = ""
        pptx_path = ""
        if hasattr(result, 'ppt_pdf_path'):
            pdf_path = result.ppt_pdf_path
        if hasattr(result, 'ppt_pptx_path'):
            pptx_path = result.ppt_pptx_path

        return {
            "success": True,
            "result_path": str(output_dir),
            "pdf_path": pdf_path,
            "pptx_path": pptx_path,
            "output_file_id": f"kb_ppt_{ts}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-podcast")
async def generate_podcast_from_kb(
    file_paths: List[str] = Body(..., embed=True),
    user_id: str = Body(..., embed=True),
    email: str = Body(..., embed=True),
    api_url: str = Body(..., embed=True),
    api_key: str = Body(..., embed=True),
    model: str = Body("gpt-4o", embed=True),
    tts_model: str = Body("gemini-2.5-pro-preview-tts", embed=True),
    voice_name: str = Body("Kore", embed=True),
    language: str = Body("zh", embed=True),
):
    """
    Generate podcast from knowledge base files
    """
    try:
        # Normalize file paths
        project_root = get_project_root()
        local_file_paths = []

        for f in file_paths:
            clean_path = f.lstrip('/')
            local_path = project_root / clean_path

            if not local_path.exists():
                local_path = Path(f)
                if not local_path.exists():
                    raise HTTPException(status_code=404, detail=f"File not found: {f}")

            local_file_paths.append(str(local_path))

        if not local_file_paths:
            raise HTTPException(status_code=400, detail="No valid files provided")

        # Prepare request
        podcast_req = KBPodcastRequest(
            files=local_file_paths,
            chat_api_url=api_url,
            api_key=api_key,
            model=model,
            tts_model=tts_model,
            voice_name=voice_name,
            language=language
        )
        podcast_req.email = email

        state = KBPodcastState(request=podcast_req)

        # Build and run graph
        builder = create_kb_podcast_graph()
        graph = builder.compile()

        result_state = await graph.ainvoke(state)

        # Extract results
        audio_path = ""
        script_path = ""
        result_path = ""

        if isinstance(result_state, dict):
            audio_path = result_state.get("audio_path", "")
            result_path = result_state.get("result_path", "")
        else:
            audio_path = getattr(result_state, "audio_path", "")
            result_path = getattr(result_state, "result_path", "")

        if result_path:
            script_path = str(Path(result_path) / "script.txt")

        return {
            "success": True,
            "result_path": result_path,
            "audio_path": audio_path,
            "script_path": script_path,
            "output_file_id": f"kb_podcast_{int(time.time())}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
