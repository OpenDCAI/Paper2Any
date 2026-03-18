"""
Paper2Poster Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Converts academic papers to aesthetic conference posters using multi-agent LLMs.

This workflow integrates PosterGen functionality into Paper2Any:
1. Parser Agent - extracts and structures content from paper PDF
2. Curator Agent - designs narrative-based storyboard
3. Layout Agent - creates spatially balanced three-column layout
4. Color Agent - generates harmonious color palette from affiliation logo
5. Section Title Designer - applies hierarchical typography
6. Font Agent - applies typography and keyword highlighting
7. Renderer - generates final PPTX and PNG outputs
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

from dataflow_agent.state import Paper2PosterState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)


def _ensure_result_path(state: Paper2PosterState) -> Paper2PosterState:
    """Ensure output directory exists and return path"""
    raw = getattr(state, "result_path", None)
    if raw:
        return state

    root = get_project_root()
    run_id = f"{int(time.time())}-{uuid4().hex[:8]}"
    base_dir = (root / "outputs" / "paper2poster" / run_id).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    state.result_path = str(base_dir)
    return state


def _import_postergen():
    """Import postertool modules dynamically"""
    try:
        # Add postertool to path if not already there
        postergen_path = Path(__file__).parent.parent / "toolkits" / "postertool"
        if postergen_path.exists() and str(postergen_path) not in sys.path:
            sys.path.insert(0, str(postergen_path))

        # Import PosterGen modules
        from src.state.poster_state import create_state as create_poster_state
        from src.workflow.pipeline import create_workflow_graph

        return {
            'create_poster_state': create_poster_state,
            'create_workflow_graph': create_workflow_graph,
        }
    except ImportError as e:
        log.error(f"Failed to import PosterGen modules: {e}")
        log.error("Please ensure PosterGen is installed in the parent directory")
        raise


@register("paper2poster")
def create_paper2poster_graph() -> GenericGraphBuilder:
    """Create Paper2Poster workflow graph"""
    builder = GenericGraphBuilder(
        state_model=Paper2PosterState,
        entry_point="_start_"
    )

    # NOTE: We don't import PosterGen modules here because we need to set
    # environment variables first (based on the state), and LangChain
    # initializes API clients at import time.
    # The import will happen inside the workflow node after env vars are set.

    # ----------------------------------------------------------------------
    # Node: Run PosterGen Pipeline
    # ----------------------------------------------------------------------
    async def run_postergen_pipeline(state: Paper2PosterState) -> Paper2PosterState:
        """Execute the complete PosterGen pipeline"""
        log.info("Starting PosterGen pipeline execution")

        # Get API credentials from state
        api_key = state.request.api_key or state.request.chat_api_key
        api_url = state.request.chat_api_url

        env_backup = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
        }
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if api_url:
            os.environ["OPENAI_BASE_URL"] = api_url

        try:
            postergen_modules = _import_postergen()
            log.info("postertool modules imported with request-scoped env")

            # Ensure output directory exists
            state = _ensure_result_path(state)
            output_dir = state.result_path

            # Get paper path
            paper_path = state.paper_file
            if not paper_path or not Path(paper_path).exists():
                error_msg = f"Paper file not found: {paper_path}"
                log.error(error_msg)
                state.errors.append(error_msg)
                return state

            # Get model configurations
            text_model = state.request.model
            vision_model = state.request.vision_model

            # Get poster dimensions
            poster_width = state.poster_width
            poster_height = state.poster_height

            # Get asset paths
            logo_path = state.logo_path or state.request.logo_path
            aff_logo_path = state.aff_logo_path or state.request.aff_logo_path
            url = state.url or state.request.url

            log.info(f"Paper: {paper_path}")
            log.info(f"Output: {output_dir}")
            log.info(f"Dimensions: {poster_width}x{poster_height} inches")
            log.info(f"Text Model: {text_model}")
            log.info(f"Vision Model: {vision_model}")

            # Create PosterGen state
            from langgraph.graph import StateGraph, START, END

            poster_state = postergen_modules['create_poster_state'](
                pdf_path=paper_path,
                text_model=text_model,
                vision_model=vision_model,
                width=int(poster_width),
                height=int(poster_height),
                url=url,
                logo_path=logo_path,
                aff_logo_path=aff_logo_path
            )

            # Override output directory to use our custom path
            poster_state['output_dir'] = output_dir
            poster_state['poster_name'] = Path(paper_path).stem

            # Create PosterGen workflow using their official function
            graph = postergen_modules['create_workflow_graph']()
            workflow = graph.compile()
            log.info("Executing PosterGen workflow...")

            # Execute workflow (already in PosterGen directory)
            result_state = await workflow.ainvoke(poster_state)

            # Extract results
            state.poster_name = result_state.get('poster_name', '')
            state.structured_sections = result_state.get('structured_sections')
            state.classified_visuals = result_state.get('classified_visuals')
            state.narrative_content = result_state.get('narrative_content')
            state.story_board = result_state.get('story_board')
            state.optimized_story_board = result_state.get('optimized_story_board')
            state.initial_layout_data = result_state.get('initial_layout_data')
            state.optimized_layout = result_state.get('optimized_layout')
            state.final_design_layout = result_state.get('design_layout')
            state.color_scheme = result_state.get('color_scheme')
            state.section_title_design = result_state.get('section_title_design')
            state.keywords = result_state.get('keywords')
            state.styled_layout = result_state.get('styled_layout')

            # Set output paths
            poster_name = state.poster_name
            state.output_pptx_path = str(Path(output_dir) / f"{poster_name}.pptx")
            state.output_png_path = str(Path(output_dir) / f"{poster_name}.png")

            log.info(f"✓ PosterGen pipeline completed successfully")
            log.info(f"PPTX output: {state.output_pptx_path}")
            log.info(f"PNG output: {state.output_png_path}")

        except Exception as e:
            error_msg = f"PosterGen pipeline failed: {str(e)}"
            log.error(error_msg)
            import traceback
            log.error(traceback.format_exc())
            state.errors.append(error_msg)
        finally:
            for key, value in env_backup.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        return state

    # ----------------------------------------------------------------------
    # Workflow Structure
    # ----------------------------------------------------------------------
    nodes = {
        "_start_": _ensure_result_path,
        "run_postergen_pipeline": run_postergen_pipeline,
        "_end_": lambda s: s,
    }

    edges = [
        ("run_postergen_pipeline", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges)
    builder.add_edge("_start_", "run_postergen_pipeline")

    return builder
