from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Optional
import os
from pathlib import Path
from dataflow_agent.toolkits.ragtool.vector_store_tool import process_knowledge_base_files
from dataflow_agent.utils import get_project_root
from fastapi_app.config import settings

router = APIRouter(prefix="/kb", tags=["Knowledge Base Embedding"])

@router.post("/embedding")
async def create_embedding(
    files: List[Dict[str, Optional[str]]] = Body(..., embed=True),
    api_url: Optional[str] = Body(None, embed=True),
    api_key: Optional[str] = Body(None, embed=True),
    model_name: Optional[str] = Body(None, embed=True),
    multimodal_model: Optional[str] = Body(settings.KB_EMBEDDING_MODEL, embed=True),
    image_model: Optional[str] = Body(None, embed=True),
    video_model: Optional[str] = Body(None, embed=True),
):
    """
    Generate embeddings for knowledge base files.
    
    Args:
        files: List of dicts, e.g. [{"path": "/outputs/kb_data/...", "description": "..."}]
        api_url: Custom Embedding API URL.
        api_key: Custom API Key.
        model_name: Custom Model Name.
        multimodal_model: Custom Multimodal Model Name (default: gemini-2.5-flash).
        image_model: Custom Image Model Name.
        video_model: Custom Video Model Name.
    """
    try:
        project_root = get_project_root()
        
        # Convert web paths to local absolute paths
        # Web path: /outputs/kb_data/user@example.com/file.pdf
        # Local path: {project_root}/outputs/kb_data/user@example.com/file.pdf
        
        process_list = []
        user_email = None

        for f in files:
            web_path = f.get("path")
            desc = f.get("description")
            
            if not web_path:
                continue
                
            # Remove leading slash if present to join correctly
            clean_path = web_path.lstrip('/')
            local_path = project_root / clean_path
            
            if local_path.exists():
                process_list.append({
                    "path": str(local_path),
                    "description": desc
                })
                
                # Extract email from path if not yet found
                # Path structure: .../outputs/kb_data/{email}/filename
                if not user_email:
                    try:
                        parts = local_path.parts
                        if "kb_data" in parts:
                            idx = parts.index("kb_data")
                            if idx + 1 < len(parts):
                                candidate = parts[idx + 1]
                                # Simple validation: check if it looks like an email or user dir
                                if "@" in candidate or len(candidate) > 0:
                                    user_email = candidate
                    except Exception:
                        pass
            else:
                print(f"Warning: File not found locally: {local_path}")
        
        if not process_list:
             return {
                "success": False,
                "message": "No valid files found to process."
            }

        # Define vector store location
        if user_email:
            vector_store_dir = project_root / "outputs" / "kb_data" / user_email / "vector_store"
        else:
            # Fallback to main if email not found
            vector_store_dir = project_root / "outputs" / "kb_data" / "vector_store_main"
        
        manifest = await process_knowledge_base_files(
            process_list, 
            base_dir=str(vector_store_dir),
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            multimodal_model=multimodal_model,
            image_model=image_model,
            video_model=video_model
        )
        
        return {
            "success": True,
            "message": f"Successfully processed {len(process_list)} files",
            "manifest": manifest
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_kb_files(email: Optional[str] = None):
    """
    List all processed files in the knowledge base (with UUIDs).
    Args:
        email: Optional user email to list specific user's vector store. 
               If None, tries to list global (though now we prefer user-specific).
    """
    try:
        project_root = get_project_root()
        
        if email:
            vector_store_dir = project_root / "outputs" / "kb_data" / email / "vector_store"
        else:
            vector_store_dir = project_root / "outputs" / "kb_data" / "vector_store_main"
            
        manifest_path = vector_store_dir / "knowledge_manifest.json"
        
        if manifest_path.exists():
            import json
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "project_name": "kb_project",
                "files": []
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
