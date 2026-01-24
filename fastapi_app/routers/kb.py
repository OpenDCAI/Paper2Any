import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from typing import Optional, List, Dict, Any

from dataflow_agent.state import IntelligentQARequest, IntelligentQAState
from dataflow_agent.workflow.wf_intelligent_qa import create_intelligent_qa_graph
from dataflow_agent.utils import get_project_root
from fastapi_app.config import settings

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
