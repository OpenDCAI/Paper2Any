from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Optional
import os
from pathlib import Path
from dataflow_agent.toolkits.ragtool.vector_store_tool import process_knowledge_base_files, VectorStoreManager
from dataflow_agent.utils import get_project_root
from fastapi_app.config import settings
from fastapi_app.utils import _to_outputs_url
from fastapi_app.dependencies.auth import get_supabase_client

router = APIRouter(prefix="/kb", tags=["Knowledge Base Embedding"])


def _extract_email_from_path(path_str: str) -> Optional[str]:
    try:
        parts = Path(path_str).parts
        if "kb_data" in parts:
            idx = parts.index("kb_data")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        return None
    return None


def _write_manifest_ids_to_supabase(manifest: Dict[str, Any]) -> None:
    supabase = get_supabase_client()
    if not supabase:
        return

    files = manifest.get("files", []) if isinstance(manifest, dict) else []
    for f in files:
        file_id = f.get("id")
        original_path = f.get("original_path", "")
        if not file_id or not original_path:
            continue

        outputs_url = _to_outputs_url(original_path)
        try:
            resp = supabase.table("knowledge_base_files").update(
                {"kb_file_id": file_id}
            ).eq("storage_path", outputs_url).execute()
            updated = bool(getattr(resp, "data", None))

            if not updated:
                email = _extract_email_from_path(original_path)
                filename = Path(original_path).name
                if email and filename:
                    supabase.table("knowledge_base_files").update(
                        {"kb_file_id": file_id}
                    ).eq("user_email", email).eq("file_name", filename).execute()
        except Exception as e:
            print(f"[kb_embedding] Supabase writeback failed: {e}")

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
        
        try:
            _write_manifest_ids_to_supabase(manifest)
        except Exception as e:
            print(f"[kb_embedding] writeback error: {e}")

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


@router.post("/search")
async def search_kb(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    email: Optional[str] = Body(None, embed=True),
    api_url: Optional[str] = Body(None, embed=True),
    api_key: Optional[str] = Body(None, embed=True),
    model_name: Optional[str] = Body(None, embed=True),
    file_ids: Optional[List[str]] = Body(None, embed=True),
):
    """
    Vector search in knowledge base.
    Returns matched text (or media description) with source file info.
    """
    try:
        project_root = get_project_root()
        if email:
            base_dir = project_root / "outputs" / "kb_data" / email / "vector_store"
        else:
            base_dir = project_root / "outputs" / "kb_data" / "vector_store_main"

        kwargs = {"base_dir": str(base_dir)}
        if api_url:
            if "/embeddings" not in api_url:
                api_url = api_url.rstrip("/") + "/embeddings"
            kwargs["embedding_api_url"] = api_url
        if api_key:
            kwargs["api_key"] = api_key
        if model_name:
            kwargs["embedding_model"] = model_name

        manager = VectorStoreManager(**kwargs)
        results = manager.search(query=query, top_k=top_k, file_ids=file_ids)

        # Build lookup for source file metadata
        manifest = manager.manifest or {"files": []}
        files_by_id = {f.get("id"): f for f in manifest.get("files", []) if f.get("id")}

        formatted = []
        for item in results:
            meta = item.get("metadata", {})
            source_id = item.get("source_file_id")
            src = files_by_id.get(source_id, {})
            src_path = src.get("original_path", "")
            src_url = _to_outputs_url(src_path) if src_path else ""

            media_path = meta.get("path") or ""
            media_url = _to_outputs_url(media_path) if media_path else ""

            formatted.append({
                "score": item.get("score"),
                "content": item.get("content"),
                "type": item.get("type"),
                "source_file": {
                    "id": source_id,
                    "file_type": src.get("file_type"),
                    "original_path": src_path,
                    "url": src_url
                },
                "media": {
                    "path": media_path,
                    "url": media_url
                } if media_path else None,
                "metadata": meta
            })

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": formatted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
