"""
File management endpoints.

Handles file uploads and history retrieval with JWT authentication.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import FileResponse

from fastapi_app.dependencies import get_current_user, get_optional_user, AuthUser
from dataflow_agent.utils import get_project_root


router = APIRouter(prefix="/files", tags=["files"])
PROJECT_ROOT = get_project_root()


def _to_outputs_url(abs_path: str, request: Request) -> str:
    """Convert absolute file path to /outputs URL."""
    try:
        rel = Path(abs_path).relative_to(PROJECT_ROOT)
        return f"{request.url.scheme}://{request.url.netloc}/{rel.as_posix()}"
    except ValueError:
        return abs_path


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    workflow_type: str = Form(...),
    email: Optional[str] = Form(None),
    user: Optional[AuthUser] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """
    Upload a file to local storage.
    
    Args:
        file: File to upload
        workflow_type: Type of workflow (e.g., 'paper2ppt', 'ppt2polish')
        email: User email (fallback when JWT not available)
        user: Authenticated user (from JWT token, optional)
        
    Returns:
        File metadata including download URL
    """
    try:
        # Determine user directory: JWT user > email parameter > "default"
        if user:
            user_dir = user.email or user.id
        elif email:
            user_dir = email
        else:
            user_dir = "default"
        
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Create directory structure: outputs/{user_dir}/{workflow_type}/{timestamp}/
        save_dir = PROJECT_ROOT / "outputs" / user_dir / workflow_type / str(timestamp)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = save_dir / file.filename
        content = await file.read()
        file_path.write_bytes(content)
        
        return {
            "success": True,
            "file_name": file.filename,
            "file_size": len(content),
            "workflow_type": workflow_type,
            "file_path": str(file_path),
            "created_at": datetime.now().isoformat(),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )


@router.get("/history")
async def get_file_history(
    request: Request,
    email: Optional[str] = None,
    user: Optional[AuthUser] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """
    Get file history for authenticated user.
    
    Args:
        request: FastAPI request object (for URL generation)
        email: User email (fallback when JWT not available)
        user: Authenticated user (from JWT token, optional)
        
    Returns:
        List of file records
    """
    try:
        # Determine user directory: JWT user > email parameter > "default"
        if user:
            user_dir = user.email or user.id
        elif email:
            user_dir = email
        else:
            user_dir = "default"
        
        base_dir = PROJECT_ROOT / "outputs" / user_dir
        
        if not base_dir.exists():
            return {
                "success": True,
                "files": [],
            }
        
        files_data: List[Dict[str, Any]] = []
        
        # Recursively scan all files
        for p in base_dir.rglob("*"):
            if not p.is_file():
                continue
            
            # Exclude input directory files
            if "input" in p.parts:
                continue
            
            # Only include specific file types
            suffix = p.suffix.lower()
            filename = p.name
            
            if suffix in {".pptx", ".pdf", ".png", ".svg"}:
                # Filter logic:
                # 1. All .pptx files
                # 2. All files starting with "paper2ppt"
                # 3. All fig_*.png and fig_*.svg files
                should_show = False
                if suffix == ".pptx":
                    should_show = True
                elif filename.startswith("paper2ppt"):
                    should_show = True
                elif filename.startswith("fig_") and suffix in {".png", ".svg"}:
                    should_show = True
                
                if should_show:
                    stat = p.stat()
                    url = _to_outputs_url(str(p), request)
                    
                    # Infer workflow_type from path: outputs/{user_dir}/{workflow_type}/...
                    try:
                        rel = p.relative_to(base_dir)
                        wf_type = rel.parts[0] if len(rel.parts) > 0 else "unknown"
                        file_id = str(rel)  # Use relative path as unique ID
                    except Exception:
                        wf_type = "unknown"
                        file_id = str(p.name) + "_" + str(stat.st_mtime)
                    
                    files_data.append({
                        "id": file_id,
                        "file_name": p.name,
                        "file_size": stat.st_size,
                        "workflow_type": wf_type,
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "download_url": url
                    })
        
        # Sort by modification time descending
        files_data.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "success": True,
            "files": files_data,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get file history: {str(e)}"
        )
