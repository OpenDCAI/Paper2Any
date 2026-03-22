from __future__ import annotations

from fastapi import APIRouter, Depends

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

router = APIRouter(tags=["paper2citation"])


def get_service() -> Paper2CitationService:
    from fastapi_app.services.paper2citation_service import Paper2CitationService

    return Paper2CitationService()


@router.post("/paper2citation/authors/search", response_model=Paper2CitationAuthorSearchResponse)
async def paper2citation_search_authors(
    body: Paper2CitationAuthorSearchRequest,
    service: Paper2CitationService = Depends(get_service),
) -> Paper2CitationAuthorSearchResponse:
    return await service.search_authors(body)


@router.post("/paper2citation/author/detail", response_model=Paper2CitationAuthorDetailResponse)
async def paper2citation_author_detail(
    body: Paper2CitationAuthorDetailRequest,
    service: Paper2CitationService = Depends(get_service),
) -> Paper2CitationAuthorDetailResponse:
    return await service.get_author_detail(body)


@router.post("/paper2citation/author/publications", response_model=Paper2CitationAuthorPublicationsResponse)
async def paper2citation_author_publications(
    body: Paper2CitationAuthorPublicationsRequest,
    service: Paper2CitationService = Depends(get_service),
) -> Paper2CitationAuthorPublicationsResponse:
    return await service.get_author_publications(body)


@router.post("/paper2citation/paper/detail", response_model=Paper2CitationPaperDetailResponse)
async def paper2citation_paper_detail(
    body: Paper2CitationPaperDetailRequest,
    service: Paper2CitationService = Depends(get_service),
) -> Paper2CitationPaperDetailResponse:
    return await service.get_paper_detail(body)


@router.post("/paper2citation/paper/context", response_model=Paper2CitationPaperContextResponse)
async def paper2citation_paper_context(
    body: Paper2CitationPaperContextRequest,
    service: Paper2CitationService = Depends(get_service),
) -> Paper2CitationPaperContextResponse:
    return await service.get_paper_context(body)
