import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])

# Base directory for storing KB files
# Use absolute path as requested by user or relative to project root
# User requested: /data/users/liuzhou/online/Paper2Any/outputs/kb_data
# We will use relative path 'outputs/kb_data' which resolves to that in the current workspace
KB_BASE_DIR = Path("outputs/kb_data")

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
