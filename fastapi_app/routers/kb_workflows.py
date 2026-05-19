from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from dataflow_agent.state import (
    IntelligentQARequest,
    IntelligentQAState,
    KBMindMapRequest,
    KBMindMapState,
    KBPodcastRequest,
    KBPodcastState,
)
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow import run_workflow
from fastapi_app.config import settings
from fastapi_app.dependencies import AuthUser, get_current_user
from fastapi_app.schemas import (
    DeepResearchRequest,
    DeepResearchResponse,
    KBReportRequest,
    KBReportResponse,
    Paper2PPTRequest,
)
from fastapi_app.services.managed_api_service import (
    is_free_billing_mode,
    resolve_llm_credentials,
    resolve_model_name,
)
from fastapi_app.utils import _to_outputs_url
from fastapi_app.routers.kb import (
    IMAGE_EXTENSIONS,
    _canonical_user_email,
    _canonical_user_id,
    _convert_to_pdf,
    _generated_dir,
    _get_deepresearch_service,
    _get_report_service,
    _merge_pdfs,
    _resolve_kb_identity,
    _resolve_user_owned_output_path,
    _vector_store_dir,
)


router = APIRouter(prefix="/kb", tags=["Knowledge Base Workflows"])


@router.post("/generate-ppt")
async def generate_ppt_from_kb(
    file_path: Optional[str] = Body(None, embed=True),
    file_paths: Optional[List[str]] = Body(None, embed=True),
    image_paths: Optional[List[str]] = Body(None, embed=True),
    image_items: Optional[List[Dict[str, Any]]] = Body(None, embed=True),
    query: Optional[str] = Body("", embed=True),
    need_embedding: bool = Body(False, embed=True),
    search_top_k: int = Body(8, embed=True),
    user_id: Optional[str] = Body(None, embed=True),
    email: Optional[str] = Body(None, embed=True),
    notebook_id: Optional[str] = Body(None, embed=True),
    api_url: Optional[str] = Body(None, embed=True),
    api_key: Optional[str] = Body(None, embed=True),
    style: str = Body("modern", embed=True),
    language: str = Body("zh", embed=True),
    page_count: int = Body(10, embed=True),
    model: Optional[str] = Body(None, embed=True),
    gen_fig_model: Optional[str] = Body(None, embed=True),
    user: AuthUser = Depends(get_current_user),
):
    """
    Generate PPT from KB documents.
    """
    try:
        api_url, api_key = resolve_llm_credentials(api_url, api_key, scope="kb")
        resolved_model = resolve_model_name(
            model,
            managed_default=settings.KB_CHAT_MODEL,
            fallback_default=settings.KB_CHAT_MODEL,
        )
        resolved_image_model = resolve_model_name(
            gen_fig_model,
            managed_default=settings.PAPER2PPT_DEFAULT_IMAGE_MODEL,
            fallback_default=settings.PAPER2PPT_DEFAULT_IMAGE_MODEL,
        )
        email, user_id = _resolve_kb_identity(user)
        input_paths = file_paths or ([file_path] if file_path else [])
        if not input_paths:
            raise HTTPException(status_code=400, detail="No input files provided")

        project_root = get_project_root()
        ts = int(time.time())
        if notebook_id:
            output_dir = _generated_dir(email, notebook_id, "ppt", user_id)
        else:
            output_dir = project_root / "outputs" / "kb_outputs" / email / f"{ts}_ppt"
            output_dir.mkdir(parents=True, exist_ok=True)

        doc_paths: List[Path] = []
        user_image_items: List[Dict[str, Any]] = []
        for p in input_paths:
            local_path = _resolve_user_owned_output_path(p, user)
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

        if len(local_pdf_paths) > 1:
            merged_pdf = convert_dir / "merged.pdf"
            local_file_path = _merge_pdfs(local_pdf_paths, merged_pdf)
        else:
            local_file_path = local_pdf_paths[0]

        resolved_image_items: List[Dict[str, Any]] = []
        for item in image_items or []:
            raw_path = item.get("path") or item.get("url") or ""
            if not raw_path:
                continue
            img_path = _resolve_user_owned_output_path(str(raw_path), user)
            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                resolved_image_items.append(
                    {
                        "path": str(img_path),
                        "description": item.get("description") or item.get("desc") or "",
                    }
                )

        for img in image_paths or []:
            img_path = _resolve_user_owned_output_path(img, user)
            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                resolved_image_items.append({"path": str(img_path), "description": ""})

        resolved_image_items.extend(user_image_items)

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
            from dataflow_agent.toolkits.ragtool.vector_store_tool import (
                VectorStoreManager,
                process_knowledge_base_files,
            )

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
                        if str(Path(f.get("original_path", "")).resolve()) in target and f.get("id"):
                            ids.append(f["id"])
                    except Exception:
                        continue
                return ids

            file_ids = _match_file_ids(manifest or manager.manifest or {}, doc_paths)
            if query and file_ids:
                results = manager.search(query=query, top_k=search_top_k, file_ids=file_ids)
                retrieval_text = "\n\n".join([r.get("content", "") for r in results if r.get("content")])

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
            model=resolved_model,
            gen_fig_model=resolved_image_model,
            aspect_ratio="16:9",
            use_long_paper=False,
        )

        from fastapi_app.workflow_adapters.wa_paper2ppt import _init_state_from_request

        state_pc = _init_state_from_request(ppt_req, result_path=output_dir)
        state_pc.kb_query = query or ""
        state_pc.kb_retrieval_text = retrieval_text
        state_pc.kb_user_images = resolved_image_items
        state_pc = await run_workflow("kb_page_content", state_pc)
        pagecontent = getattr(state_pc, "pagecontent", []) or []

        state_pc.pagecontent = pagecontent
        state_pp = await run_workflow("paper2ppt_parallel_consistent_style", state_pc)

        pdf_path = getattr(state_pp, "ppt_pdf_path", "")
        pptx_path = getattr(state_pp, "ppt_pptx_path", "")

        return {
            "success": True,
            "result_path": str(output_dir),
            "pdf_path": _to_outputs_url(pdf_path) if pdf_path else "",
            "pptx_path": _to_outputs_url(pptx_path) if pptx_path else "",
            "output_file_id": f"kb_ppt_{ts}",
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-podcast")
async def generate_podcast_from_kb(
    file_paths: List[str] = Body(..., embed=True),
    user_id: Optional[str] = Body(None, embed=True),
    email: Optional[str] = Body(None, embed=True),
    notebook_id: Optional[str] = Body(None, embed=True),
    api_url: Optional[str] = Body(None, embed=True),
    api_key: Optional[str] = Body(None, embed=True),
    model: Optional[str] = Body(None, embed=True),
    tts_model: Optional[str] = Body(None, embed=True),
    voice_name: str = Body("", embed=True),
    voice_name_b: str = Body("Puck", embed=True),
    podcast_mode: str = Body("monologue", embed=True),
    podcast_length: str = Body("standard", embed=True),
    language: str = Body("zh", embed=True),
    user: AuthUser = Depends(get_current_user),
):
    try:
        api_url, api_key = resolve_llm_credentials(api_url, api_key, scope="kb")
        resolved_model = resolve_model_name(
            model,
            managed_default=settings.KB_CHAT_MODEL,
            fallback_default=settings.KB_CHAT_MODEL,
        )
        resolved_tts_model = resolve_model_name(
            tts_model,
            managed_default=settings.PAPER2VIDEO_TTS_MODEL,
            fallback_default=settings.PAPER2VIDEO_TTS_MODEL,
        )
        email, user_id = _resolve_kb_identity(user)
        project_root = get_project_root()
        if notebook_id:
            output_dir = _generated_dir(email, notebook_id, "podcast", user_id)
        else:
            ts = int(time.time())
            output_dir = project_root / "outputs" / "kb_outputs" / email / f"{ts}_podcast"
            output_dir.mkdir(parents=True, exist_ok=True)

        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid files provided")

        local_paths: List[Path] = []
        for f in file_paths:
            local_path = _resolve_user_owned_output_path(f, user)
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {f}")
            local_paths.append(local_path)

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

        podcast_req = KBPodcastRequest(
            files=local_file_paths,
            chat_api_url=api_url,
            api_key=api_key,
            model=resolved_model,
            tts_model=resolved_tts_model,
            voice_name=voice_name,
            voice_name_b=voice_name_b,
            podcast_mode=podcast_mode,
            podcast_length=podcast_length,
            language=language,
        )
        podcast_req.email = email

        state = KBPodcastState(request=podcast_req, result_path=str(output_dir))
        result_state = await run_workflow("kb_podcast", state)

        audio_path = ""
        result_path = ""
        if isinstance(result_state, dict):
            audio_path = result_state.get("audio_path", "")
            result_path = result_state.get("result_path", "")
        else:
            audio_path = getattr(result_state, "audio_path", "")
            result_path = getattr(result_state, "result_path", "")

        script_path = str(Path(result_path) / "script.txt") if result_path else ""
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

        return {
            "success": True,
            "result_path": _to_outputs_url(result_path) if result_path else "",
            "audio_path": _to_outputs_url(audio_path) if audio_path else "",
            "script_path": _to_outputs_url(script_path) if script_path else "",
            "output_file_id": f"kb_podcast_{int(time.time())}",
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-mindmap")
async def generate_mindmap_from_kb(
    file_paths: List[str] = Body(..., embed=True),
    user_id: Optional[str] = Body(None, embed=True),
    email: Optional[str] = Body(None, embed=True),
    notebook_id: Optional[str] = Body(None, embed=True),
    api_url: Optional[str] = Body(None, embed=True),
    api_key: Optional[str] = Body(None, embed=True),
    model: Optional[str] = Body(None, embed=True),
    mindmap_style: str = Body("default", embed=True),
    max_depth: int = Body(3, embed=True),
    language: str = Body("zh", embed=True),
    user: AuthUser = Depends(get_current_user),
):
    try:
        api_url, api_key = resolve_llm_credentials(api_url, api_key, scope="kb")
        resolved_model = resolve_model_name(
            model,
            managed_default=settings.MINDMAP_DEFAULT_MODEL,
            fallback_default=settings.MINDMAP_DEFAULT_MODEL,
        )
        email, user_id = _resolve_kb_identity(user)

        local_file_paths = []
        for f in file_paths:
            local_path = _resolve_user_owned_output_path(f, user)
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {f}")
            local_file_paths.append(str(local_path))
        if not local_file_paths:
            raise HTTPException(status_code=400, detail="No valid files provided")

        mindmap_req = KBMindMapRequest(
            files=local_file_paths,
            chat_api_url=api_url,
            api_key=api_key,
            model=resolved_model,
            mindmap_style=mindmap_style,
            max_depth=max_depth,
            language=language,
        )
        mindmap_req.email = email

        if notebook_id:
            nb_output_dir = _generated_dir(email, notebook_id, "mindmap", user_id)
            state = KBMindMapState(request=mindmap_req, result_path=str(nb_output_dir))
        else:
            state = KBMindMapState(request=mindmap_req)

        result_state = await run_workflow("kb_mindmap", state)
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
            "output_file_id": f"kb_mindmap_{int(time.time())}",
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deep-research", response_model=DeepResearchResponse)
async def deep_research_from_kb(
    req: DeepResearchRequest,
    user: AuthUser = Depends(get_current_user),
):
    if req.mode == "web" and not (req.search_api_key or (is_free_billing_mode() and settings.DEFAULT_SEARCH_API_KEY)):
        raise HTTPException(status_code=400, detail="Search API key required")
    if req.mode == "web" and req.search_provider == "google_cse" and not (
        req.google_cse_id or (is_free_billing_mode() and settings.DEFAULT_GOOGLE_CSE_ID)
    ):
        raise HTTPException(status_code=400, detail="google_cse_id required")
    if not req.topic and not req.file_paths:
        raise HTTPException(status_code=400, detail="Topic or files required")
    req.email = _canonical_user_email(user)
    req.user_id = _canonical_user_id(user)
    if req.file_paths:
        req.file_paths = [str(_resolve_user_owned_output_path(path, user)) for path in req.file_paths]
    service = _get_deepresearch_service()
    return await service.run(req)


@router.post("/generate-report", response_model=KBReportResponse)
async def generate_report_from_kb(
    req: KBReportRequest,
    user: AuthUser = Depends(get_current_user),
):
    if not req.file_paths:
        raise HTTPException(status_code=400, detail="No valid files provided")
    req.email = _canonical_user_email(user)
    req.user_id = _canonical_user_id(user)
    req.file_paths = [str(_resolve_user_owned_output_path(path, user)) for path in req.file_paths]
    service = _get_report_service()
    return await service.run(req)


@router.post("/save-mindmap")
async def save_mindmap_to_file(
    file_url: str = Body(..., embed=True),
    content: str = Body(..., embed=True),
    user: AuthUser = Depends(get_current_user),
):
    try:
        if not file_url:
            raise HTTPException(status_code=400, detail="File URL is required")

        local_path = _resolve_user_owned_output_path(file_url, user)
        if local_path.suffix.lower() not in {".mmd", ".mermaid", ".md"}:
            raise HTTPException(status_code=400, detail="Invalid mindmap file type")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content or "", encoding="utf-8")

        return {
            "success": True,
            "mindmap_path": _to_outputs_url(str(local_path)),
        }
    except HTTPException:
        raise
