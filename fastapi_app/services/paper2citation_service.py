from __future__ import annotations

from fastapi import HTTPException

from dataflow_agent.toolkits.citationtool.citation_utils import CitationDataError
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
from fastapi_app.workflow_adapters.wa_paper2citation import (
    run_paper2citation_author_detail_wf_api,
    run_paper2citation_author_publications_wf_api,
    run_paper2citation_author_search_wf_api,
    run_paper2citation_paper_context_wf_api,
    run_paper2citation_paper_detail_wf_api,
)


class Paper2CitationService:
    async def search_authors(
        self,
        req: Paper2CitationAuthorSearchRequest,
    ) -> Paper2CitationAuthorSearchResponse:
        if not req.author_name.strip():
            raise HTTPException(status_code=400, detail="author_name is required")
        try:
            return await run_paper2citation_author_search_wf_api(req)
        except CitationDataError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    async def get_author_detail(
        self,
        req: Paper2CitationAuthorDetailRequest,
    ) -> Paper2CitationAuthorDetailResponse:
        if not any([
            req.openalex_author_id.strip(),
            req.dblp_id.strip(),
            req.display_name.strip(),
        ]):
            raise HTTPException(status_code=400, detail="author identifier is required")
        try:
            return await run_paper2citation_author_detail_wf_api(req)
        except CitationDataError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    async def get_author_publications(
        self,
        req: Paper2CitationAuthorPublicationsRequest,
    ) -> Paper2CitationAuthorPublicationsResponse:
        if not any([
            req.openalex_author_id.strip(),
            req.dblp_id.strip(),
            req.display_name.strip(),
        ]):
            raise HTTPException(status_code=400, detail="author identifier is required")
        try:
            return await run_paper2citation_author_publications_wf_api(req)
        except CitationDataError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    async def get_paper_detail(
        self,
        req: Paper2CitationPaperDetailRequest,
    ) -> Paper2CitationPaperDetailResponse:
        if not req.doi_or_url.strip():
            raise HTTPException(status_code=400, detail="doi_or_url is required")
        try:
            return await run_paper2citation_paper_detail_wf_api(req)
        except CitationDataError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    async def get_paper_context(
        self,
        req: Paper2CitationPaperContextRequest,
    ) -> Paper2CitationPaperContextResponse:
        if not req.target_doi_or_url.strip():
            raise HTTPException(status_code=400, detail="target_doi_or_url is required")
        if not any([
            req.citing_work_openalex_id.strip(),
            req.citing_work_doi_or_url.strip(),
            req.citing_work_title.strip(),
        ]):
            raise HTTPException(status_code=400, detail="citing work identifier is required")
        try:
            return await run_paper2citation_paper_context_wf_api(req)
        except CitationDataError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
