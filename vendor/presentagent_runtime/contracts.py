from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel


class EditablePPTRunRequest(BaseModel):
    result_path: str
    pagecontent: list[dict[str, Any]]
    language: str = "zh"
    style: str = ""
    model: str = ""
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    vlm_model: str = ""
    vlm_api_url: Optional[str] = None
    vlm_api_key: Optional[str] = None
    image_model: str = ""
    image_api_url: Optional[str] = None
    image_api_key: Optional[str] = None
    enable_agent_planner: bool = True
    enable_material_resolution: bool = True
    enable_llm_codegen: bool = False
    include_pdf_preview: bool = True


class EditablePPTInputRunRequest(BaseModel):
    """Run request that lets the runtime own the pagecontent parsing step."""
    result_path: str
    input_type: str = "PDF"
    input_content: str = ""
    language: str = "zh"
    style: str = ""
    page_count: int = 8
    use_long_paper: bool = False
    outline_model: str = ""
    model: str = ""
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    credential_scope: Optional[str] = None
    email: str = ""
    aspect_ratio: str = "16:9"
    ref_img: str = ""
    gen_fig_model: str = ""
    render_dpi: Optional[int] = None
    vlm_model: str = ""
    vlm_api_url: Optional[str] = None
    vlm_api_key: Optional[str] = None
    image_model: str = ""
    image_api_url: Optional[str] = None
    image_api_key: Optional[str] = None
    enable_agent_planner: bool = True
    enable_material_resolution: bool = True
    enable_llm_codegen: bool = False
    include_pdf_preview: bool = True


class EditablePPTRunArtifacts(BaseModel):
    run_dir: str
    materials_manifest_path: str = ""
    material_resolution_path: str = ""
    planned_ir_path: str = ""
    final_ir_path: str = ""
    ir_path: str
    recipe_path: str
    pptx_path: str
    pdf_path: str = ""
    log_path: str
    slides_dir: str = ""
    slide_artifacts: list[SlideArtifact] = []


class SlideArtifact(BaseModel):
    index: int
    slide_id: str = ""
    title: str = ""
    pptx_path: str
    preview_png_path: str = ""
    status: Literal["pending", "rendered", "failed"] = "rendered"


EditablePPTRunArtifacts.model_rebuild()
