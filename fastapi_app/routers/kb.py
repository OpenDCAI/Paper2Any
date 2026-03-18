import json as _json
import os
import re
import shutil
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from typing import Optional, List, Dict, Any

import fitz  # PyMuPDF

from dataflow_agent.state import IntelligentQARequest, IntelligentQAState, KBPodcastRequest, KBPodcastState, KBMindMapRequest, KBMindMapState
from dataflow_agent.workflow.wf_intelligent_qa import create_intelligent_qa_graph
from dataflow_agent.workflow.wf_kb_podcast import create_kb_podcast_graph
from dataflow_agent.workflow.wf_kb_mindmap import create_kb_mindmap_graph
from dataflow_agent.toolkits.ragtool.vector_store_tool import process_knowledge_base_files, VectorStoreManager
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow import run_workflow
from dataflow_agent.logger import get_logger
from fastapi_app.config import settings
from fastapi_app.schemas import Paper2PPTRequest, DeepResearchRequest, DeepResearchResponse, KBReportRequest, KBReportResponse
from fastapi_app.utils import _from_outputs_url, _to_outputs_url
from fastapi_app.workflow_adapters.wa_paper2ppt import _init_state_from_request
from fastapi_app.services.kb_deepresearch_service import KBDeepResearchService
from fastapi_app.services.kb_report_service import KBReportService

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])
log = get_logger(__name__)

# Base directory for storing KB files
# Use absolute path as requested by user or relative to project root
# We will use relative path 'outputs/kb_data' which resolves to that in the current workspace
KB_BASE_DIR = Path("outputs/kb_data")

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".mp4"}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DOC_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}

NOTEBOOKS_DIR = KB_BASE_DIR / "_notebooks"

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None


# ---------------------------------------------------------------------------
# Notebook helpers
# ---------------------------------------------------------------------------

def _safe_dirname(name: str) -> str:
    """Turn a notebook name into a filesystem-safe directory name.
    Keeps CJK characters, alphanumerics, hyphens and underscores.
    Collapses whitespace to '_' and strips dangerous chars."""
    if not name:
        return "unnamed"
    # Replace path separators and other dangerous chars
    cleaned = re.sub(r'[/\\:*?"<>|]', "", name)
    # Collapse whitespace to single underscore
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_.")
    return cleaned or "unnamed"


def _notebooks_file(user_id: str) -> Path:
    """Return the path to the local notebooks JSON for a user."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    return NOTEBOOKS_DIR / f"{user_id or 'default'}.json"


def _load_notebooks(user_id: str) -> List[Dict[str, Any]]:
    path = _notebooks_file(user_id)
    if not path.exists():
        return []
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_notebooks(user_id: str, notebooks: List[Dict[str, Any]]) -> None:
    path = _notebooks_file(user_id)
    path.write_text(_json.dumps(notebooks, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_unique_dirname(user_id: str, base_name: str) -> str:
    """If base_name already used by another notebook, append _2, _3, etc."""
    notebooks = _load_notebooks(user_id)
    existing = {nb.get("dir_name") for nb in notebooks if nb.get("dir_name")}
    if base_name not in existing:
        return base_name
    seq = 2
    while f"{base_name}_{seq}" in existing:
        seq += 1
    return f"{base_name}_{seq}"


def _create_notebook_local(user_id: str, name: str, description: str = "") -> Dict[str, Any]:
    notebooks = _load_notebooks(user_id)
    nb_id = str(uuid.uuid4())
    dir_name = _safe_dirname(name)
    dir_name = _ensure_unique_dirname(user_id, dir_name)
    new_nb = {
        "id": nb_id,
        "name": name,
        "dir_name": dir_name,
        "description": description,
        "created_at": int(time.time()),
    }
    notebooks.append(new_nb)
    _save_notebooks(user_id, notebooks)
    return new_nb


def _get_notebook_local(user_id: str, notebook_id: str) -> Optional[Dict[str, Any]]:
    for nb in _load_notebooks(user_id):
        if nb.get("id") == notebook_id:
            return nb
    return None


def _get_notebook_dir_name(user_id: str, notebook_id: str) -> str:
    """Return the persisted dir_name for a notebook, or fall back to notebook_id."""
    nb = _get_notebook_local(user_id, notebook_id)
    if nb and nb.get("dir_name"):
        return nb["dir_name"]
    return notebook_id


def _notebook_base_dir(email: str, notebook_id: str, user_id: str = "default") -> Path:
    """kb_data/{email}/{notebook_dir_name}/
    Falls back to old path (using notebook_id) if new path doesn't exist."""
    dir_name = _get_notebook_dir_name(user_id, notebook_id)
    new_path = KB_BASE_DIR / email / dir_name
    if new_path.exists() or dir_name != notebook_id:
        return new_path
    # Fallback: old path using notebook_id directly
    old_path = KB_BASE_DIR / email / notebook_id
    if old_path.exists():
        return old_path
    # Neither exists yet — use new path
    return new_path


def _notebook_sources_dir(email: str, notebook_id: str, user_id: str = "default") -> Path:
    """kb_data/{email}/{notebook_name}/sources/"""
    return _notebook_base_dir(email, notebook_id, user_id) / "sources"


def _mineru_dir(email: str, notebook_id: str, user_id: str = "default") -> Path:
    """kb_data/{email}/{notebook_name}/mineru/"""
    return _notebook_base_dir(email, notebook_id, user_id) / "mineru"


def _vector_store_dir(email: str, notebook_id: str, user_id: str = "default") -> Path:
    """kb_data/{email}/{notebook_name}/vector_store/"""
    return _notebook_base_dir(email, notebook_id, user_id) / "vector_store"


def _generated_dir(
    email: str, notebook_id: str, output_type: str, user_id: str = "default"
) -> Path:
    """kb_data/{email}/{notebook_name}/generated/{output_type}/{next_seq}/
    Auto-increments the sequence number."""
    base = _notebook_base_dir(email, notebook_id, user_id) / "generated" / output_type
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        (int(d.name) for d in base.iterdir() if d.is_dir() and d.name.isdigit()),
        reverse=True,
    )
    next_seq = (existing[0] + 1) if existing else 1
    out = base / str(next_seq)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _extract_text_result(state, role_name: str) -> str:
    try:
        result = getattr(state, "agent_results", {}).get(role_name, {}).get("results", {})
        if isinstance(result, dict):
            return result.get("text") or result.get("raw") or result.get("content") or ""
        if isinstance(result, str):
            return result
    except Exception:
        return ""
    return ""


def _extract_file_text(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
        if suffix in {".docx", ".doc"}:
            if Document is None:
                return ""
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        if suffix in {".pptx", ".ppt"}:
            if Presentation is None:
                return ""
            prs = Presentation(path)
            text = ""
            for i, slide in enumerate(prs.slides):
                text += f"--- Slide {i+1} ---\n"
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        if suffix in {".md", ".txt"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        return ""
    except Exception:
        return ""


def _build_text_context(file_paths: List[str], max_chars: int = 60000) -> str:
    if not file_paths:
        return ""
    chunks: List[str] = []
    for raw in file_paths:
        try:
            local_path = _resolve_local_path(raw)
            _ensure_under_outputs(local_path)
            text = _extract_file_text(local_path)
            if not text:
                continue
            chunks.append(f"=== {local_path.name} ===\n{text}")
        except Exception:
            continue

    combined = "\n\n".join(chunks)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[...content truncated]"
    return combined


def _get_deepresearch_service() -> KBDeepResearchService:
    return KBDeepResearchService()


def _get_report_service() -> KBReportService:
    return KBReportService()


def _safe_zip_stem(name: str) -> str:
    if not name:
        return "knowledge_base"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned.strip("_") or "knowledge_base"


def _ensure_under_outputs(path: Path) -> None:
    outputs_root = (get_project_root() / "outputs").resolve()
    try:
        path.resolve().relative_to(outputs_root)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")


def _resolve_local_path(path_or_url: str) -> Path:
    if not path_or_url:
        raise HTTPException(status_code=400, detail="Empty file path")
    raw = _from_outputs_url(path_or_url)
    p = Path(raw)
    if not p.is_absolute():
        p = (get_project_root() / p).resolve()
    return p


def _convert_to_pdf(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(input_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pdf_path = output_dir / f"{input_path.stem}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=500, detail=f"PDF conversion failed for {input_path.name}")
    return pdf_path


def _merge_pdfs(pdf_paths: List[Path], output_path: Path) -> Path:
    if not pdf_paths:
        raise HTTPException(status_code=400, detail="No PDF files to merge")
    merged = fitz.open()
    for pdf in pdf_paths:
        with fitz.open(pdf) as src:
            merged.insert_pdf(src)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(output_path)
    merged.close()
    return output_path


def _append_images_to_pptx(pptx_path: Path, image_paths: List[Path]) -> None:
    try:
        from pptx import Presentation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"python-pptx not available: {e}")

    prs = Presentation(str(pptx_path))
    blank_layout = prs.slide_layouts[6] if len(prs.slide_layouts) > 6 else prs.slide_layouts[-1]
    for img_path in image_paths:
        slide = prs.slides.add_slide(blank_layout)
        slide.shapes.add_picture(
            str(img_path),
            0,
            0,
            width=prs.slide_width,
            height=prs.slide_height
        )
    prs.save(str(pptx_path))

@router.post("/notebooks/create")
async def create_notebook(
    name: str = Body(..., embed=True),
    user_id: str = Body("default", embed=True),
    description: str = Body("", embed=True),
):
    """Create a new notebook and persist its dir_name mapping."""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Notebook name is required")
    nb = _create_notebook_local(user_id, name.strip(), description)
    return {"success": True, "notebook": nb}


@router.get("/notebooks/list")
async def list_notebooks(user_id: str = "default"):
    """List all notebooks for a user."""
    return {"success": True, "notebooks": _load_notebooks(user_id)}


@router.get("/notebooks/{notebook_id}")
async def get_notebook(notebook_id: str, user_id: str = "default"):
    """Get a single notebook by id."""
    nb = _get_notebook_local(user_id, notebook_id)
    if not nb:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"success": True, "notebook": nb}


@router.post("/notebooks/{notebook_id}/list-outputs")
async def list_notebook_outputs(
    notebook_id: str,
    email: str = Body(..., embed=True),
    user_id: str = Body("default", embed=True),
):
    """Scan the generated/ directory of a notebook and return all outputs."""
    base = _notebook_base_dir(email, notebook_id, user_id) / "generated"
    if not base.exists():
        return {"success": True, "outputs": {}}

    outputs: Dict[str, List[Dict[str, Any]]] = {}
    for type_dir in sorted(base.iterdir()):
        if not type_dir.is_dir():
            continue
        items: List[Dict[str, Any]] = []
        for seq_dir in sorted(type_dir.iterdir(), key=lambda d: d.name):
            if not seq_dir.is_dir() or not seq_dir.name.isdigit():
                continue
            files = []
            for f in seq_dir.rglob("*"):
                if f.is_file():
                    files.append({
                        "name": f.name,
                        "path": _to_outputs_url(str(f)),
                        "size": f.stat().st_size,
                    })
            items.append({"seq": int(seq_dir.name), "files": files})
        outputs[type_dir.name] = items
    return {"success": True, "outputs": outputs}


@router.post("/upload")
async def upload_kb_file(
    file: UploadFile = File(...),
    email: str = Form(...),
    user_id: str = Form(...),
    notebook_id: Optional[str] = Form(None),
):
    """
    Upload a file to the user's knowledge base directory.
    When notebook_id is provided: outputs/kb_data/{email}/{notebook_name}/sources/{filename}
    Legacy (no notebook_id): outputs/kb_data/{email}/{filename}
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
        # Determine target directory
        if notebook_id:
            user_dir = _notebook_sources_dir(email, notebook_id, user_id)
        else:
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

        static_path = _to_outputs_url(str(file_path))

        return {
            "success": True,
            "filename": filename,
            "file_size": os.path.getsize(file_path),
            "storage_path": str(file_path),
            "static_url": static_path,
            "file_type": file.content_type
        }

    except Exception as e:
        log.error(f"Error uploading file: {e}")
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


@router.post("/delete-batch")
async def delete_kb_files_batch(
    storage_paths: List[str] = Body(..., embed=True),
):
    """
    Delete multiple files from physical storage.
    """
    if not storage_paths:
        raise HTTPException(status_code=400, detail="No storage paths provided")

    deleted = 0
    skipped: List[str] = []
    errors: List[str] = []

    for raw in storage_paths:
        try:
            local_path = Path(_from_outputs_url(raw)).resolve()
            _ensure_under_outputs(local_path)
            if local_path.exists() and local_path.is_file():
                local_path.unlink()
                deleted += 1
            else:
                skipped.append(raw)
        except Exception as e:
            errors.append(f"{raw}: {e}")

    return {
        "success": True,
        "deleted": deleted,
        "skipped": skipped,
        "errors": errors,
    }


@router.post("/export-zip")
async def export_kb_zip(
    files: List[str] = Body(..., embed=True),
    email: Optional[str] = Body(None, embed=True),
    kb_name: Optional[str] = Body(None, embed=True),
    include_root_dir: bool = Body(True, embed=True),
):
    """
    Export a list of KB files into a zip archive.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    project_root = get_project_root()
    outputs_root = project_root / "outputs"
    export_root = outputs_root / "kb_exports" / (email or "default")
    export_root.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    zip_stem = _safe_zip_stem(kb_name or "knowledge_base")
    zip_path = export_root / f"{zip_stem}_{ts}.zip"

    used_names: Dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for raw in files:
            try:
                if not raw:
                    continue
                local_path = Path(_from_outputs_url(raw)).resolve()
                _ensure_under_outputs(local_path)
                if not local_path.exists() or not local_path.is_file():
                    continue
                base_name = local_path.name
                if base_name in used_names:
                    used_names[base_name] += 1
                    stem = local_path.stem
                    suffix = local_path.suffix
                    base_name = f"{stem}_{used_names[base_name]}{suffix}"
                else:
                    used_names[base_name] = 0

                arcname = base_name
                if include_root_dir:
                    arcname = f"{zip_stem}/{base_name}"
                zf.write(local_path, arcname)
            except Exception:
                continue

    if not zip_path.exists():
        raise HTTPException(status_code=500, detail="Failed to create zip")

    return {
        "success": True,
        "zip_path": _to_outputs_url(str(zip_path)),
        "count": len(files),
    }

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
        
        # Run workflow via registry (统一使用 run_workflow)
        result_state = await run_workflow("intelligent_qa", state)
        
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
    file_path: Optional[str] = Body(None, embed=True),
    file_paths: Optional[List[str]] = Body(None, embed=True),
    image_paths: Optional[List[str]] = Body(None, embed=True),
    image_items: Optional[List[Dict[str, Any]]] = Body(None, embed=True),
    query: Optional[str] = Body("", embed=True),
    need_embedding: bool = Body(False, embed=True),
    search_top_k: int = Body(8, embed=True),
    user_id: str = Body(..., embed=True),
    email: str = Body(..., embed=True),
    notebook_id: Optional[str] = Body(None, embed=True),
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
        # Normalize and validate input files (PDF/PPT/DOC/IMG)
        input_paths = file_paths or ([file_path] if file_path else [])
        if not input_paths:
            raise HTTPException(status_code=400, detail="No input files provided")

        # Create output directory
        project_root = get_project_root()
        if notebook_id:
            output_dir = _generated_dir(email, notebook_id, "ppt", user_id)
        else:
            ts = int(time.time())
            output_dir = project_root / "outputs" / "kb_outputs" / email / f"{ts}_ppt"
            output_dir.mkdir(parents=True, exist_ok=True)

        # Split docs/images
        doc_paths: List[Path] = []
        user_image_items: List[Dict[str, Any]] = []
        for p in input_paths:
            local_path = _resolve_local_path(p)
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {p}")
            ext = local_path.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                user_image_items.append({"path": str(local_path), "description": ""})
            elif ext in {".pdf", ".pptx", ".ppt", ".docx", ".doc"}:
                doc_paths.append(local_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type for PPT: {local_path.name}")

        if not doc_paths:
            raise HTTPException(status_code=400, detail="At least one document file is required for PPT generation")

        # Convert docs to PDF for MinerU merge
        local_pdf_paths: List[Path] = []
        convert_dir = output_dir / "input"
        convert_dir.mkdir(parents=True, exist_ok=True)
        for p in doc_paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                local_pdf_paths.append(p)
            elif ext in {".pptx", ".ppt", ".docx", ".doc"}:
                local_pdf_paths.append(_convert_to_pdf(p, convert_dir))
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type for PPT: {p.name}")

        # Merge PDFs if multiple
        if len(local_pdf_paths) > 1:
            merge_dir = output_dir / "input"
            merged_pdf = merge_dir / "merged.pdf"
            local_file_path = _merge_pdfs(local_pdf_paths, merged_pdf)
        else:
            local_file_path = local_pdf_paths[0]

        # Normalize image items (optional)
        resolved_image_items: List[Dict[str, Any]] = []
        for item in image_items or []:
            raw_path = item.get("path") or item.get("url") or ""
            if not raw_path:
                continue
            img_path = _resolve_local_path(str(raw_path))
            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                resolved_image_items.append({
                    "path": str(img_path),
                    "description": item.get("description") or item.get("desc") or ""
                })

        for img in image_paths or []:
            img_path = _resolve_local_path(img)
            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                resolved_image_items.append({
                    "path": str(img_path),
                    "description": ""
                })

        resolved_image_items.extend(user_image_items)

        # Embedding + retrieval (optional)
        retrieval_text = ""
        if need_embedding:
            if notebook_id:
                base_dir = _vector_store_dir(email, notebook_id, user_id)
            else:
                base_dir = project_root / "outputs" / "kb_data" / email / "vector_store"
            embed_api_url = api_url
            if "/embeddings" not in embed_api_url:
                embed_api_url = embed_api_url.rstrip("/") + "/embeddings"

            files_for_embed = [{"path": str(p), "description": ""} for p in doc_paths]
            manifest = await process_knowledge_base_files(
                files_for_embed,
                base_dir=str(base_dir),
                api_url=embed_api_url,
                api_key=api_key,
                model_name=None,
                multimodal_model=None,
            )

            manager = VectorStoreManager(
                base_dir=str(base_dir),
                embedding_api_url=embed_api_url,
                api_key=api_key,
            )

            def _match_file_ids(m: Dict[str, Any], paths: List[Path]) -> List[str]:
                ids: List[str] = []
                target = {str(p.resolve()) for p in paths}
                for f in m.get("files", []):
                    try:
                        if str(Path(f.get("original_path", "")).resolve()) in target:
                            if f.get("id"):
                                ids.append(f["id"])
                    except Exception:
                        continue
                return ids

            file_ids = _match_file_ids(manifest or manager.manifest or {}, doc_paths)
            if query and file_ids:
                results = manager.search(query=query, top_k=search_top_k, file_ids=file_ids)
                retrieval_text = "\n\n".join([r.get("content", "") for r in results if r.get("content")])

        # Prepare request
        ppt_req = Paper2PPTRequest(
            input_type="PDF",
            input_content=str(local_file_path),
            email=email,
            chat_api_url=api_url,
            chat_api_key=api_key,
            api_key=api_key,
            style=style,
            language=language,
            page_count=page_count,
            model=model,
            gen_fig_model=gen_fig_model,
            aspect_ratio="16:9",
            use_long_paper=False
        )

        # Run KB pagecontent workflow
        state_pc = _init_state_from_request(ppt_req, result_path=output_dir)
        state_pc.kb_query = query or ""
        state_pc.kb_retrieval_text = retrieval_text
        state_pc.kb_user_images = resolved_image_items
        state_pc = await run_workflow("kb_page_content", state_pc)
        pagecontent = getattr(state_pc, "pagecontent", []) or []

        # Run PPT generation with injected pagecontent
        state_pc.pagecontent = pagecontent
        state_pp = await run_workflow("paper2ppt_parallel_consistent_style", state_pc)

        # Extract output paths
        pdf_path = ""
        pptx_path = ""
        if hasattr(state_pp, 'ppt_pdf_path'):
            pdf_path = state_pp.ppt_pdf_path
        if hasattr(state_pp, 'ppt_pptx_path'):
            pptx_path = state_pp.ppt_pptx_path

        return {
            "success": True,
            "result_path": str(output_dir),
            "pdf_path": _to_outputs_url(pdf_path) if pdf_path else "",
            "pptx_path": _to_outputs_url(pptx_path) if pptx_path else "",
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
    notebook_id: Optional[str] = Body(None, embed=True),
    api_url: str = Body(..., embed=True),
    api_key: str = Body(..., embed=True),
    model: str = Body("gpt-4o", embed=True),
    tts_model: str = Body("cosyvoice-v3-flash", embed=True),
    voice_name: str = Body("", embed=True),
    voice_name_b: str = Body("Puck", embed=True),
    podcast_mode: str = Body("monologue", embed=True),
    podcast_length: str = Body("standard", embed=True),
    language: str = Body("zh", embed=True),
):
    """
    Generate podcast from knowledge base files
    """
    try:
        project_root = get_project_root()
        if notebook_id:
            output_dir = _generated_dir(email, notebook_id, "podcast", user_id)
        else:
            ts = int(time.time())
            output_dir = project_root / "outputs" / "kb_outputs" / email / f"{ts}_podcast"
            output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize file paths
        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid files provided")

        local_paths: List[Path] = []
        for f in file_paths:
            local_path = _resolve_local_path(f)
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {f}")
            local_paths.append(local_path)

        # If multiple files, merge into a single PDF (doc/ppt will be converted)
        if len(local_paths) > 1:
            merge_dir = output_dir / "input"
            merge_dir.mkdir(parents=True, exist_ok=True)

            pdf_paths: List[Path] = []
            for p in local_paths:
                ext = p.suffix.lower()
                if ext == ".pdf":
                    pdf_paths.append(p)
                elif ext in {".docx", ".doc", ".pptx", ".ppt"}:
                    pdf_paths.append(_convert_to_pdf(p, merge_dir))
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type for podcast: {p.name}")

            merged_pdf = merge_dir / "merged.pdf"
            local_file_paths = [str(_merge_pdfs(pdf_paths, merged_pdf))]
        else:
            local_file_paths = [str(local_paths[0])]

        # Prepare request
        podcast_req = KBPodcastRequest(
            files=local_file_paths,
            chat_api_url=api_url,
            api_key=api_key,
            model=model,
            tts_model=tts_model,
            voice_name=voice_name,
            voice_name_b=voice_name_b,
            podcast_mode=podcast_mode,
            podcast_length=podcast_length,
            language=language
        )
        podcast_req.email = email

        state = KBPodcastState(request=podcast_req, result_path=str(output_dir))

        # Run workflow via registry (统一使用 run_workflow)
        result_state = await run_workflow("kb_podcast", state)

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

        audio_error = ""
        if not audio_path:
            audio_error = "No audio path returned from workflow"
        elif isinstance(audio_path, str) and audio_path.startswith("["):
            audio_error = audio_path
        else:
            audio_file = Path(audio_path)
            if not audio_file.is_absolute():
                audio_file = (get_project_root() / audio_file).resolve()
            if not audio_file.exists():
                audio_error = f"Audio file not found: {audio_file}"

        if audio_error:
            raise HTTPException(status_code=500, detail=audio_error)

        audio_url = _to_outputs_url(audio_path) if audio_path else ""
        script_url = _to_outputs_url(script_path) if script_path else ""
        result_url = _to_outputs_url(result_path) if result_path else ""

        return {
            "success": True,
            "result_path": result_url,
            "audio_path": audio_url,
            "script_path": script_url,
            "output_file_id": f"kb_podcast_{int(time.time())}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-mindmap")
async def generate_mindmap_from_kb(
    file_paths: List[str] = Body(..., embed=True),
    user_id: str = Body(..., embed=True),
    email: str = Body(..., embed=True),
    notebook_id: Optional[str] = Body(None, embed=True),
    api_url: str = Body(..., embed=True),
    api_key: str = Body(..., embed=True),
    model: str = Body("gpt-4o", embed=True),
    mindmap_style: str = Body("default", embed=True),
    max_depth: int = Body(3, embed=True),
    language: str = Body("zh", embed=True),
):
    """
    Generate mindmap from knowledge base files
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
        mindmap_req = KBMindMapRequest(
            files=local_file_paths,
            chat_api_url=api_url,
            api_key=api_key,
            model=model,
            mindmap_style=mindmap_style,
            max_depth=max_depth,
            language=language
        )
        mindmap_req.email = email

        if notebook_id:
            nb_output_dir = _generated_dir(email, notebook_id, "mindmap", user_id)
            state = KBMindMapState(request=mindmap_req, result_path=str(nb_output_dir))
        else:
            state = KBMindMapState(request=mindmap_req)

        # Run workflow via registry (统一使用 run_workflow)
        result_state = await run_workflow("kb_mindmap", state)

        # Extract results
        mermaid_code = ""
        result_path = ""

        if isinstance(result_state, dict):
            mermaid_code = result_state.get("mermaid_code", "")
            result_path = result_state.get("result_path", "")
        else:
            mermaid_code = getattr(result_state, "mermaid_code", "")
            result_path = getattr(result_state, "result_path", "")

        mindmap_path = ""
        if result_path:
            mmd_path = Path(result_path) / "mindmap.mmd"
            if (not mmd_path.exists()) and mermaid_code:
                try:
                    mmd_path.write_text(mermaid_code, encoding="utf-8")
                except Exception:
                    pass
            if mmd_path.exists():
                mindmap_path = _to_outputs_url(str(mmd_path))

        return {
            "success": True,
            "result_path": _to_outputs_url(result_path) if result_path else "",
            "mermaid_code": mermaid_code,
            "mindmap_path": mindmap_path,
            "output_file_id": f"kb_mindmap_{int(time.time())}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deep-research", response_model=DeepResearchResponse)
async def deep_research_from_kb(
    req: DeepResearchRequest,
):
    """
    Deep research workflow入口（router -> service -> wa -> wf）
    """
    if req.mode == "web" and not req.search_api_key:
        raise HTTPException(status_code=400, detail="Search API key required")
    if req.mode == "web" and req.search_provider == "google_cse" and not req.google_cse_id:
        raise HTTPException(status_code=400, detail="google_cse_id required")
    if not req.topic and not req.file_paths:
        raise HTTPException(status_code=400, detail="Topic or files required")
    service = _get_deepresearch_service()
    return await service.run(req)


@router.post("/generate-report", response_model=KBReportResponse)
async def generate_report_from_kb(
    req: KBReportRequest,
):
    """
    Generate a report with insights/analysis from KB documents (workflow).
    """
    if not req.file_paths:
        raise HTTPException(status_code=400, detail="No valid files provided")
    service = _get_report_service()
    return await service.run(req)


@router.post("/save-mindmap")
async def save_mindmap_to_file(
    file_url: str = Body(..., embed=True),
    content: str = Body(..., embed=True),
):
    """
    Save edited Mermaid mindmap code back to the output file.
    """
    try:
        if not file_url:
            raise HTTPException(status_code=400, detail="File URL is required")

        local_path = Path(_from_outputs_url(file_url))
        if not local_path.is_absolute():
            local_path = (get_project_root() / local_path).resolve()

        outputs_root = (get_project_root() / "outputs").resolve()
        try:
            local_path.relative_to(outputs_root)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid output path")

        if local_path.suffix.lower() not in {".mmd", ".mermaid", ".md"}:
            raise HTTPException(status_code=400, detail="Invalid mindmap file type")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content or "", encoding="utf-8")

        return {
            "success": True,
            "mindmap_path": _to_outputs_url(str(local_path))
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
