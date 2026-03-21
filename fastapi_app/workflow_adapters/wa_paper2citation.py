from __future__ import annotations

from dataflow_agent.state import Paper2CitationRequest, Paper2CitationState
from dataflow_agent.workflow import run_workflow
from fastapi_app.schemas import (
    Paper2CitationAuthorDetailRequest,
    Paper2CitationAuthorDetailResponse,
    Paper2CitationAuthorPublicationsRequest,
    Paper2CitationAuthorPublicationsResponse,
    Paper2CitationPaperContextRequest,
    Paper2CitationPaperContextResponse,
    Paper2CitationAuthorSearchRequest,
    Paper2CitationAuthorSearchResponse,
    Paper2CitationPaperDetailRequest,
    Paper2CitationPaperDetailResponse,
)


def _state_value(state: Paper2CitationState | dict, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


async def run_paper2citation_author_search_wf_api(
    req: Paper2CitationAuthorSearchRequest,
) -> Paper2CitationAuthorSearchResponse:
    state = Paper2CitationState(
        request=Paper2CitationRequest(
            mode="author_search",
            author_name=req.author_name,
            max_author_candidates=req.max_author_candidates,
        )
    )
    final_state = await run_workflow("paper2citation", state)
    return Paper2CitationAuthorSearchResponse(
        success=True,
        mode="author_search",
        query=_state_value(final_state, "query", ""),
        candidates=_state_value(final_state, "author_candidates", []),
    )


async def run_paper2citation_author_detail_wf_api(
    req: Paper2CitationAuthorDetailRequest,
) -> Paper2CitationAuthorDetailResponse:
    state = Paper2CitationState(
        request=Paper2CitationRequest(
            mode="author_detail",
            openalex_author_id=req.openalex_author_id,
            dblp_id=req.dblp_id,
            display_name=req.display_name,
            affiliation_hint=req.affiliation_hint,
            candidate_source=req.candidate_source,
            max_publications=req.max_publications,
            max_citing_works=req.max_citing_works,
            publication_page=req.publication_page,
            publication_page_size=req.publication_page_size,
        )
    )
    final_state = await run_workflow("paper2citation", state)
    return Paper2CitationAuthorDetailResponse(
        success=True,
        mode="author_detail",
        query=_state_value(final_state, "query", ""),
        best_effort_notice=_state_value(final_state, "best_effort_notice", ""),
        author_profile=_state_value(final_state, "author_profile", {}),
        publication_stats=_state_value(final_state, "publication_stats", {}),
        citation_stats=_state_value(final_state, "citation_stats", {}),
        publication_pagination=_state_value(final_state, "publication_pagination", {}),
        publications=_state_value(final_state, "publications", []),
        citing_works=_state_value(final_state, "citing_works", []),
        citing_authors=_state_value(final_state, "citing_authors", []),
        citing_institutions=_state_value(final_state, "citing_institutions", []),
        honors_stats=_state_value(final_state, "honors_stats", []),
        matched_honorees=_state_value(final_state, "matched_honorees", []),
    )


async def run_paper2citation_author_publications_wf_api(
    req: Paper2CitationAuthorPublicationsRequest,
) -> Paper2CitationAuthorPublicationsResponse:
    state = Paper2CitationState(
        request=Paper2CitationRequest(
            mode="author_publications",
            openalex_author_id=req.openalex_author_id,
            dblp_id=req.dblp_id,
            display_name=req.display_name,
            affiliation_hint=req.affiliation_hint,
            candidate_source=req.candidate_source,
            max_publications=req.max_publications,
            publication_page=req.publication_page,
            publication_page_size=req.publication_page_size,
        )
    )
    final_state = await run_workflow("paper2citation", state)
    return Paper2CitationAuthorPublicationsResponse(
        success=True,
        mode="author_publications",
        query=_state_value(final_state, "query", ""),
        best_effort_notice=_state_value(final_state, "best_effort_notice", ""),
        publication_stats=_state_value(final_state, "publication_stats", {}),
        publication_pagination=_state_value(final_state, "publication_pagination", {}),
        publications=_state_value(final_state, "publications", []),
    )


async def run_paper2citation_paper_detail_wf_api(
    req: Paper2CitationPaperDetailRequest,
) -> Paper2CitationPaperDetailResponse:
    state = Paper2CitationState(
        request=Paper2CitationRequest(
            mode="paper_detail",
            doi_or_url=req.doi_or_url,
            max_citing_works=req.max_citing_works,
        )
    )
    final_state = await run_workflow("paper2citation", state)
    return Paper2CitationPaperDetailResponse(
        success=True,
        mode="paper_detail",
        query=_state_value(final_state, "query", ""),
        best_effort_notice=_state_value(final_state, "best_effort_notice", ""),
        paper_detail=_state_value(final_state, "paper_detail", {}),
        citation_stats=_state_value(final_state, "citation_stats", {}),
        citing_works=_state_value(final_state, "citing_works", []),
        citing_authors=_state_value(final_state, "citing_authors", []),
        citing_institutions=_state_value(final_state, "citing_institutions", []),
        honors_stats=_state_value(final_state, "honors_stats", []),
        matched_honorees=_state_value(final_state, "matched_honorees", []),
    )


async def run_paper2citation_paper_context_wf_api(
    req: Paper2CitationPaperContextRequest,
) -> Paper2CitationPaperContextResponse:
    state = Paper2CitationState(
        request=Paper2CitationRequest(
            mode="paper_context",
            doi_or_url=req.target_doi_or_url,
            citing_work_openalex_id=req.citing_work_openalex_id,
            citing_work_doi_or_url=req.citing_work_doi_or_url,
            citing_work_title=req.citing_work_title,
        )
    )
    final_state = await run_workflow("paper2citation", state)
    return Paper2CitationPaperContextResponse(
        success=True,
        mode="paper_context",
        query=_state_value(final_state, "query", ""),
        best_effort_notice=_state_value(final_state, "best_effort_notice", ""),
        source_url=_state_value(_state_value(final_state, "citation_context", {}), "source_url", ""),
        target_reference_match=_state_value(_state_value(final_state, "citation_context", {}), "target_reference_match", {}),
        citing_paper=_state_value(_state_value(final_state, "citation_context", {}), "citing_paper", {}),
        contexts=_state_value(_state_value(final_state, "citation_context", {}), "contexts", []),
        summary=_state_value(_state_value(final_state, "citation_context", {}), "summary", ""),
        citation_intents=_state_value(_state_value(final_state, "citation_context", {}), "citation_intents", []),
    )
