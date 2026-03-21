from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2CitationState
from dataflow_agent.toolkits.citationtool.context_locator import (
    extract_citation_context_for_work,
)
from dataflow_agent.toolkits.citationtool.citation_utils import (
    _affiliation_match_score,
    _get_affiliations,
    _name_match_score,
    build_publication_sample_stats,
    CitationDataError,
    DBLPCitationClient,
    evaluate_dblp_openalex_bridge,
    is_close_title,
    OpenAlexCitationClient,
    aggregate_citation_network,
    build_author_profile,
    merge_crossref_metadata,
    resolve_doi_or_openalex_id,
    simplify_author_candidate,
    simplify_work,
)
from dataflow_agent.toolkits.citationtool.honor_enrichment import enrich_honors_with_websearch
from dataflow_agent.workflow.registry import register

AUTHOR_NETWORK_SEED_LIMIT = 20
MAX_CITING_WORKS_PER_SEED = 25
MAX_SEED_FETCH_CONCURRENCY = 6
log = get_logger(__name__)


def _prepare_work_items(raw_works: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for raw_work in raw_works:
        item = simplify_work(raw_work)
        item["raw_authorships"] = raw_work.get("authorships") or []
        items.append(item)
    return items


async def _resolve_work_from_request(
    client: OpenAlexCitationClient,
    raw_value: str,
) -> Dict[str, Any]:
    resolved = resolve_doi_or_openalex_id(raw_value)
    if resolved["openalex_work_id"]:
        return await client.get_work(resolved["openalex_work_id"])
    if resolved["doi"]:
        return await client.get_work_by_doi(resolved["doi"])
    return await client.search_work_by_bibliographic(raw_value.strip())

def _merge_publication_fields(base_item: Dict[str, Any], resolved_item: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base_item)
    for key in [
        "openalex_work_id",
        "publication_date",
        "type",
        "cited_by_count",
        "institutions",
        "landing_page_url",
        "raw_authorships",
    ]:
        value = resolved_item.get(key)
        if value or key in {"cited_by_count", "raw_authorships"}:
            merged[key] = value
    if resolved_item.get("venue"):
        merged["venue"] = resolved_item["venue"]
    return merged


def _find_matching_resolved_publication(
    publication: Dict[str, Any],
    resolved_items: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    publication_doi = (publication.get("doi") or "").strip()
    publication_title = (publication.get("title") or "").strip()
    for resolved in resolved_items:
        resolved_doi = (resolved.get("doi") or "").strip()
        if publication_doi and resolved_doi and publication_doi == resolved_doi:
            return resolved
    for resolved in resolved_items:
        if is_close_title(publication_title, resolved.get("title", "") or ""):
            return resolved
    return None
def _select_seed_works(
    publication_items: List[Dict[str, Any]],
    *,
    limit: int = AUTHOR_NETWORK_SEED_LIMIT,
) -> List[Dict[str, Any]]:
    return sorted(
        publication_items,
        key=lambda item: item.get("cited_by_count", 0),
        reverse=True,
    )[: max(limit, 1)]


def _select_dblp_seed_candidates(
    publication_items: List[Dict[str, Any]],
    *,
    limit: int = AUTHOR_NETWORK_SEED_LIMIT,
) -> List[Dict[str, Any]]:
    candidate_limit = max(limit * 3, limit)
    return sorted(
        publication_items,
        key=lambda item: (
            0 if (item.get("doi") or "").strip() else 1,
            -(item.get("year") or 0),
            item.get("title", ""),
        ),
    )[:candidate_limit]


def _build_citation_sample_stats(
    *,
    seed_publications_count: int,
    citing_works_count: int,
    citing_authors_count: int,
    citing_institutions_count: int,
    seed_citing_works_fetch_limit: int | None = None,
    built_from_publication_page: int | None = None,
    built_from_publication_page_size: int | None = None,
) -> Dict[str, Any]:
    stats = {
        "seed_publications_count": seed_publications_count,
        "citing_works_count": citing_works_count,
        "citing_authors_count": citing_authors_count,
        "citing_institutions_count": citing_institutions_count,
    }
    if seed_citing_works_fetch_limit is not None:
        stats["seed_citing_works_fetch_limit"] = seed_citing_works_fetch_limit
    if built_from_publication_page is not None:
        stats["built_from_publication_page"] = built_from_publication_page
    if built_from_publication_page_size is not None:
        stats["built_from_publication_page_size"] = built_from_publication_page_size
    return stats


def _build_publication_pagination(
    *,
    current_page: int,
    page_size: int,
    total_items: int,
) -> Dict[str, int]:
    safe_page_size = max(page_size, 1)
    total_pages = max((max(total_items, 0) + safe_page_size - 1) // safe_page_size, 1)
    safe_page = min(max(current_page, 1), total_pages)
    return {
        "page": safe_page,
        "page_size": safe_page_size,
        "total_items": max(total_items, 0),
        "total_pages": total_pages,
    }


def _slice_publications(
    publication_items: List[Dict[str, Any]],
    *,
    current_page: int,
    page_size: int,
) -> List[Dict[str, Any]]:
    safe_page = max(current_page, 1)
    safe_page_size = max(page_size, 1)
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size
    return publication_items[start:end]


async def _fetch_openalex_publication_page(
    client: OpenAlexCitationClient,
    *,
    author_id: str,
    page: int,
    page_size: int,
) -> List[Dict[str, Any]]:
    raw_publications = await client.get_author_works(
        author_id,
        per_page=page_size,
        page=page,
    )
    return _prepare_work_items(raw_publications)


async def _fetch_openalex_seed_works(
    client: OpenAlexCitationClient,
    *,
    author_id: str,
    seed_limit: int,
) -> List[Dict[str, Any]]:
    raw_seed_publications = await client.get_author_works(
        author_id,
        per_page=max(seed_limit, 1),
        page=1,
        sort="cited_by_count:desc",
    )
    return _prepare_work_items(raw_seed_publications)


async def _load_citing_works_for_author(
    client: OpenAlexCitationClient,
    publication_items: List[Dict[str, Any]],
    max_citing_works: int,
    *,
    seed_limit: int = AUTHOR_NETWORK_SEED_LIMIT,
) -> Dict[str, Any]:
    deduped: Dict[str, Dict[str, Any]] = {}
    seed_works = _select_seed_works(publication_items, limit=seed_limit)
    semaphore = asyncio.Semaphore(MAX_SEED_FETCH_CONCURRENCY)

    async def _fetch_seed_citing_works(work: Dict[str, Any]) -> List[Dict[str, Any]]:
        work_id = work.get("openalex_work_id", "")
        if not work_id:
            return []
        async with semaphore:
            return await client.get_citing_works(
                work_id,
                per_page=min(max_citing_works, MAX_CITING_WORKS_PER_SEED),
            )

    citing_work_lists = await asyncio.gather(
        *[_fetch_seed_citing_works(work) for work in seed_works],
        return_exceptions=True,
    )
    for raw_citing_works in citing_work_lists:
        if isinstance(raw_citing_works, Exception):
            continue
        for raw_citing_work in raw_citing_works:
            citing_item = simplify_work(raw_citing_work)
            citing_item["raw_authorships"] = raw_citing_work.get("authorships") or []
            if citing_item["openalex_work_id"]:
                deduped[citing_item["openalex_work_id"]] = citing_item

    return {
        "citing_works": sorted(
            deduped.values(),
            key=lambda item: (item.get("publication_date", ""), item.get("cited_by_count", 0)),
            reverse=True,
        )[:max_citing_works],
        "seed_publications_count": len(seed_works),
    }


async def _resolve_openalex_work_for_publication(
    client: OpenAlexCitationClient,
    publication: Dict[str, Any],
) -> Dict[str, Any] | None:
    doi = (publication.get("doi") or "").strip()
    if doi:
        try:
            return await client.get_work_by_doi(doi)
        except CitationDataError:
            pass

    title = (publication.get("title") or "").strip()
    if not title:
        return None

    try:
        resolved = await client.search_work_by_bibliographic(title)
    except CitationDataError:
        return None
    if not is_close_title(title, resolved.get("title", "") or ""):
        return None
    return resolved


async def _resolve_publications_with_openalex(
    client: OpenAlexCitationClient,
    publications: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_SEED_FETCH_CONCURRENCY)

    async def _resolve(publication: Dict[str, Any]) -> Dict[str, Any] | None:
        async with semaphore:
            resolved_raw_work = await _resolve_openalex_work_for_publication(client, publication)
        if not resolved_raw_work:
            return None
        resolved_item = simplify_work(resolved_raw_work)
        resolved_item["raw_authorships"] = resolved_raw_work.get("authorships") or []
        return _merge_publication_fields(publication, resolved_item)

    results = await asyncio.gather(*[_resolve(publication) for publication in publications], return_exceptions=True)
    resolved_publications: List[Dict[str, Any]] = []
    for item in results:
        if isinstance(item, Exception) or not item:
            continue
        resolved_publications.append(item)
    return resolved_publications


def _merge_author_candidates(
    dblp_candidates: List[Dict[str, Any]],
    openalex_candidates: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()

    for item in dblp_candidates + openalex_candidates:
        openalex_author_id = item.get("openalex_author_id", "").strip()
        dblp_id = item.get("dblp_id", "").strip()
        if openalex_author_id:
            key = ("openalex", openalex_author_id)
        elif dblp_id:
            key = ("dblp", dblp_id)
        else:
            key = (
                item.get("display_name", "").strip(),
                tuple(item.get("affiliations", [])[:1]),
            )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _score_resolution_candidate(
    *,
    display_name: str,
    affiliation_hint: str,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "candidate": candidate,
        "name_score": _name_match_score(display_name, candidate.get("display_name", "") or ""),
        "affiliation_score": _affiliation_match_score(affiliation_hint, _get_affiliations(candidate)),
        "cited_by_count": int(candidate.get("cited_by_count") or 0),
    }


async def _resolve_openalex_author(
    client: OpenAlexCitationClient,
    *,
    openalex_author_id: str,
    display_name: str,
    affiliation_hint: str,
) -> Dict[str, Any]:
    if openalex_author_id.strip():
        return await client.get_author(openalex_author_id)

    if not display_name.strip():
        raise CitationDataError("display_name is required to resolve a DBLP candidate")

    candidates = await client.search_authors(display_name.strip(), per_page=50)
    scored_candidates = [
        _score_resolution_candidate(
            display_name=display_name.strip(),
            affiliation_hint=affiliation_hint.strip(),
            candidate=item,
        )
        for item in candidates
    ]
    scored_candidates = [
        item for item in scored_candidates
        if item["name_score"] >= 3
    ]
    scored_candidates.sort(
        key=lambda item: (
            -item["name_score"],
            -item["affiliation_score"],
            -item["cited_by_count"],
        )
    )

    if not scored_candidates:
        raise CitationDataError("Failed to resolve the selected author to OpenAlex")

    chosen = scored_candidates[0]
    if affiliation_hint.strip() and chosen["affiliation_score"] <= 0:
        raise CitationDataError(
            "The selected DBLP author could not be confidently matched to an OpenAlex profile. "
            "This usually means OpenAlex does not have a clearly aligned author record for that institution yet."
        )

    return chosen["candidate"]


@register("paper2citation")
def create_paper2citation_graph() -> GenericGraphBuilder:
    builder = GenericGraphBuilder(state_model=Paper2CitationState, entry_point="_start_")

    def _start_(state: Paper2CitationState) -> Paper2CitationState:
        state.mode = (state.request.mode or "").strip() or "author_search"
        if state.mode == "author_search":
            state.query = state.request.author_name.strip()
        elif state.mode in {"author_detail", "author_publications"}:
            state.query = (
                state.request.openalex_author_id.strip()
                or state.request.dblp_id.strip()
                or state.request.display_name.strip()
            )
        elif state.mode == "paper_context":
            state.query = state.request.citing_work_openalex_id.strip() or state.request.citing_work_doi_or_url.strip() or state.request.citing_work_title.strip()
        else:
            state.query = state.request.doi_or_url.strip()
        state.errors = []
        return state

    def _route(state: Paper2CitationState) -> str:
        if state.mode == "author_search":
            return "author_search"
        if state.mode == "author_detail":
            return "author_detail"
        if state.mode == "author_publications":
            return "author_publications"
        if state.mode == "paper_detail":
            return "paper_detail"
        if state.mode == "paper_context":
            return "paper_context"
        raise CitationDataError(f"Unsupported paper2citation mode: {state.mode}")

    async def author_search_node(state: Paper2CitationState) -> Paper2CitationState:
        openalex_client = OpenAlexCitationClient()
        dblp_client = DBLPCitationClient()
        try:
            openalex_raw_candidates, dblp_candidates = await asyncio.gather(
                openalex_client.search_authors(
                    state.request.author_name.strip(),
                    per_page=max(state.request.max_author_candidates * 2, 24),
                ),
                dblp_client.search_authors(
                    state.request.author_name.strip(),
                    per_page=max(state.request.max_author_candidates * 2, 20),
                ),
            )
            openalex_candidates = [
                {
                    **simplify_author_candidate(item),
                    "source": "openalex",
                }
                for item in openalex_raw_candidates
            ]
            state.author_candidates = _merge_author_candidates(
                dblp_candidates,
                openalex_candidates,
                state.request.max_author_candidates,
            )
            return state
        finally:
            await openalex_client.close()
            await dblp_client.close()

    async def author_publications_node(state: Paper2CitationState) -> Paper2CitationState:
        client = OpenAlexCitationClient()
        dblp_client = DBLPCitationClient()
        try:
            publication_page = max(int(state.request.publication_page or 1), 1)
            publication_page_size = max(int(state.request.publication_page_size or state.request.max_publications or 20), 1)
            pagination_notice = (
                "Publication pagination is loaded independently; the citation network and prestige cards come from the last full author-detail load."
            )

            if state.request.dblp_id.strip() and not state.request.openalex_author_id.strip():
                dblp_bundle = await dblp_client.get_author_profile_and_works(
                    state.request.dblp_id.strip(),
                    max_records=publication_page_size,
                    offset=(publication_page - 1) * publication_page_size,
                )
                dblp_orcid = (dblp_bundle["author_profile"].get("orcid") or "").strip()
                if dblp_orcid:
                    try:
                        raw_author = await client.get_author_by_orcid(dblp_orcid)
                        publication_items = await _fetch_openalex_publication_page(
                            client,
                            author_id=raw_author.get("id", ""),
                            page=publication_page,
                            page_size=publication_page_size,
                        )
                        bridge_check = evaluate_dblp_openalex_bridge(
                            dblp_bundle,
                            raw_author,
                            publication_items,
                        )
                        if bridge_check["accepted"]:
                            state.publication_stats = build_publication_sample_stats(
                                loaded_publications_count=len(publication_items),
                                linked_publications_count=len(publication_items),
                                seed_publications_count=0,
                            )
                            state.publication_pagination = _build_publication_pagination(
                                current_page=publication_page,
                                page_size=publication_page_size,
                                total_items=int(build_author_profile(raw_author).get("works_count") or len(publication_items)),
                            )
                            state.publications = publication_items
                            state.best_effort_notice = " ".join([
                                pagination_notice,
                                "DBLP exact author was linked to OpenAlex by ORCID after validation "
                                f"({bridge_check['evidence_text']}).",
                            ])
                            return state
                    except CitationDataError:
                        pass

                resolved_publication_items = await _resolve_publications_with_openalex(client, dblp_bundle["publications"])
                merged_publications = [
                    _find_matching_resolved_publication(publication, resolved_publication_items) or publication
                    for publication in dblp_bundle["publications"]
                ]
                state.publication_stats = build_publication_sample_stats(
                    loaded_publications_count=len(merged_publications),
                    linked_publications_count=len(resolved_publication_items),
                    seed_publications_count=0,
                )
                state.publication_pagination = _build_publication_pagination(
                    current_page=publication_page,
                    page_size=publication_page_size,
                    total_items=int(dblp_bundle["author_profile"].get("works_count") or len(merged_publications)),
                )
                state.publications = merged_publications
                state.best_effort_notice = " ".join([
                    pagination_notice,
                    "This DBLP author page could not be safely bridged to a single OpenAlex author for author-level pagination, so page works were linked publication-by-publication where possible.",
                ])
                return state

            resolved_author = await _resolve_openalex_author(
                client,
                openalex_author_id=state.request.openalex_author_id,
                display_name=state.request.display_name,
                affiliation_hint=state.request.affiliation_hint,
            )
            raw_author = await client.get_author(resolved_author.get("id", ""))
            publication_items = await _fetch_openalex_publication_page(
                client,
                author_id=raw_author.get("id", ""),
                page=publication_page,
                page_size=publication_page_size,
            )
            state.publication_stats = build_publication_sample_stats(
                loaded_publications_count=len(publication_items),
                linked_publications_count=len(publication_items),
                seed_publications_count=0,
            )
            state.publication_pagination = _build_publication_pagination(
                current_page=publication_page,
                page_size=publication_page_size,
                total_items=int(build_author_profile(raw_author).get("works_count") or len(publication_items)),
            )
            state.publications = publication_items
            state.best_effort_notice = pagination_notice
            return state
        finally:
            await client.close()
            await dblp_client.close()

    async def author_detail_node(state: Paper2CitationState) -> Paper2CitationState:
        client = OpenAlexCitationClient()
        dblp_client = DBLPCitationClient()
        try:
            publication_page = max(int(state.request.publication_page or 1), 1)
            publication_page_size = max(int(state.request.publication_page_size or state.request.max_publications or 20), 1)
            seed_limit = max(int(state.request.max_seed_works or AUTHOR_NETWORK_SEED_LIMIT), 1)
            if state.request.dblp_id.strip() and not state.request.openalex_author_id.strip():
                dblp_bundle = await dblp_client.get_author_profile_and_works(
                    state.request.dblp_id.strip(),
                    max_records=publication_page_size,
                    offset=(publication_page - 1) * publication_page_size,
                )
                bridge_rejection_notice = ""
                dblp_orcid = (dblp_bundle["author_profile"].get("orcid") or "").strip()
                if dblp_orcid:
                    try:
                        raw_author = await client.get_author_by_orcid(dblp_orcid)
                        publication_items, seed_publications = await asyncio.gather(
                            _fetch_openalex_publication_page(
                                client,
                                author_id=raw_author.get("id", ""),
                                page=publication_page,
                                page_size=publication_page_size,
                            ),
                            _fetch_openalex_seed_works(
                                client,
                                author_id=raw_author.get("id", ""),
                                seed_limit=seed_limit,
                            ),
                        )
                        bridge_check = evaluate_dblp_openalex_bridge(
                            dblp_bundle,
                            raw_author,
                            publication_items,
                        )
                        if bridge_check["accepted"]:
                            citing_bundle = await _load_citing_works_for_author(
                                client,
                                seed_publications,
                                max_citing_works=state.request.max_citing_works,
                                seed_limit=seed_limit,
                            )
                            citing_works = citing_bundle["citing_works"]
                            citation_network = aggregate_citation_network(citing_works)
                            web_honor_bundle = await enrich_honors_with_websearch(
                                target_label="author",
                                target_payload=build_author_profile(raw_author),
                                citing_authors=citation_network["citing_authors"],
                            )
                            honor_bundle = web_honor_bundle

                            state.author_profile = build_author_profile(raw_author)
                            state.author_profile["dblp_id"] = dblp_bundle["author_profile"].get("dblp_id", "")
                            if dblp_bundle["author_profile"].get("homepage"):
                                state.author_profile["homepage"] = dblp_bundle["author_profile"]["homepage"]
                            if dblp_bundle["author_profile"].get("affiliations"):
                                # For DBLP-selected authors, keep DBLP affiliations as the primary display source.
                                # OpenAlex affiliation histories are often noisy and can mix historical/weakly linked institutions.
                                state.author_profile["affiliations"] = list(dblp_bundle["author_profile"]["affiliations"])
                                state.author_profile["affiliation_text"] = ", ".join(state.author_profile["affiliations"])

                            merged_profile_honors = list(dict.fromkeys(honor_bundle.get("target_honors") or []))
                            if merged_profile_honors:
                                state.author_profile["honors"] = merged_profile_honors
                            merged_profile_titles = list(dict.fromkeys(honor_bundle.get("target_titles") or []))
                            if merged_profile_titles:
                                state.author_profile["titles"] = merged_profile_titles
                            state.publication_stats = build_publication_sample_stats(
                                loaded_publications_count=len(publication_items),
                                linked_publications_count=len(publication_items),
                                seed_publications_count=citing_bundle["seed_publications_count"],
                            )
                            state.publication_pagination = _build_publication_pagination(
                                current_page=publication_page,
                                page_size=publication_page_size,
                                total_items=int(state.author_profile.get("works_count") or len(publication_items)),
                            )
                            state.publications = publication_items
                            state.citing_works = citing_works
                            state.citing_authors = citation_network["citing_authors"]
                            state.citing_institutions = citation_network["citing_institutions"]
                            state.honors_stats = honor_bundle["honor_stats"]
                            state.matched_honorees = honor_bundle["matched_honorees"]
                            notices = []
                            if honor_bundle["best_effort_notice"]:
                                notices.append(honor_bundle["best_effort_notice"])
                            notices.append(
                                "DBLP exact author was linked to OpenAlex by ORCID after validation "
                                f"({bridge_check['evidence_text']})."
                            )
                            notices.append(
                                f"Citation sampling was built from the author's top {citing_bundle['seed_publications_count']} cited works across the full OpenAlex profile; changing publication pages does not rebuild that sample."
                            )
                            notices.append(
                                "Author totals are shown in the profile card; publication and citation stats below "
                                "describe only the currently loaded sample."
                            )
                            state.best_effort_notice = " ".join(dict.fromkeys(notices))
                            state.citation_stats = {
                                **_build_citation_sample_stats(
                                    seed_publications_count=citing_bundle["seed_publications_count"],
                                    citing_works_count=len(citing_works),
                                    citing_authors_count=len(state.citing_authors),
                                    citing_institutions_count=len(state.citing_institutions),
                                    seed_citing_works_fetch_limit=min(state.request.max_citing_works, MAX_CITING_WORKS_PER_SEED),
                                    built_from_publication_page=publication_page,
                                    built_from_publication_page_size=publication_page_size,
                                ),
                                "citing_works_by_year": citation_network["citing_works_by_year"],
                            }
                            return state
                        bridge_rejection_notice = (
                            "DBLP ORCID matched an OpenAlex author, but the author-level bridge was rejected because "
                            f"validation evidence was too weak ({bridge_check['evidence_text']}). "
                            "Falling back to publication-level linking."
                        )
                    except CitationDataError:
                        pass

                full_dblp_bundle = await dblp_client.get_author_profile_and_works(
                    state.request.dblp_id.strip(),
                    max_records=max(int(dblp_bundle["author_profile"].get("works_count") or 0), publication_page_size),
                    offset=0,
                )
                all_publications = full_dblp_bundle["publications"]
                publication_items = _slice_publications(
                    all_publications,
                    current_page=publication_page,
                    page_size=publication_page_size,
                )
                seed_candidate_publications = _select_dblp_seed_candidates(
                    all_publications,
                    limit=seed_limit,
                )
                resolved_publication_items, resolved_seed_publications = await asyncio.gather(
                    _resolve_publications_with_openalex(client, publication_items),
                    _resolve_publications_with_openalex(client, seed_candidate_publications),
                )
                seed_publications = _select_seed_works(resolved_seed_publications, limit=seed_limit)
                if not seed_publications:
                    seed_publications = _select_seed_works(resolved_publication_items, limit=seed_limit)

                citing_bundle = await _load_citing_works_for_author(
                    client,
                    seed_publications,
                    max_citing_works=state.request.max_citing_works,
                    seed_limit=seed_limit,
                )
                citing_works = citing_bundle["citing_works"]
                citation_network = aggregate_citation_network(citing_works)
                web_honor_bundle = await enrich_honors_with_websearch(
                    target_label="author",
                    target_payload=dblp_bundle["author_profile"],
                    citing_authors=citation_network["citing_authors"],
                )
                honor_bundle = web_honor_bundle

                merged_publications = [
                    _find_matching_resolved_publication(publication, resolved_publication_items) or publication
                    for publication in publication_items
                ]

                state.author_profile = full_dblp_bundle["author_profile"]
                merged_profile_honors = list(dict.fromkeys(honor_bundle.get("target_honors") or []))
                if merged_profile_honors:
                    state.author_profile["honors"] = merged_profile_honors
                merged_profile_titles = list(dict.fromkeys(honor_bundle.get("target_titles") or []))
                if merged_profile_titles:
                    state.author_profile["titles"] = merged_profile_titles
                state.publication_stats = build_publication_sample_stats(
                    loaded_publications_count=len(merged_publications),
                    linked_publications_count=len(resolved_publication_items),
                    seed_publications_count=citing_bundle["seed_publications_count"],
                )
                state.publication_pagination = _build_publication_pagination(
                    current_page=publication_page,
                    page_size=publication_page_size,
                    total_items=int(state.author_profile.get("works_count") or len(all_publications)),
                )
                state.publications = merged_publications
                state.citing_works = citing_works
                state.citing_authors = citation_network["citing_authors"]
                state.citing_institutions = citation_network["citing_institutions"]
                state.honors_stats = honor_bundle["honor_stats"]
                state.matched_honorees = honor_bundle["matched_honorees"]
                notices = []
                if honor_bundle["best_effort_notice"]:
                    notices.append(honor_bundle["best_effort_notice"])
                if bridge_rejection_notice:
                    notices.append(bridge_rejection_notice)
                if not resolved_publication_items:
                    notices.append("DBLP author was resolved exactly, but none of the recent publications could be linked to OpenAlex for citation aggregation.")
                elif len(resolved_seed_publications) < len(seed_candidate_publications):
                    notices.append("Citation aggregation is based on the subset of this author's DBLP publications that could be linked to OpenAlex from a DOI/title-prioritized global seed candidate pool.")
                notices.append(
                    f"Citation sampling was built from the top {citing_bundle['seed_publications_count']} resolved seed works across the author's global DBLP publication list; changing publication pages does not rebuild that sample."
                )
                notices.append(
                    "Author totals are shown in the profile card; publication and citation stats below describe only "
                    "the currently loaded sample."
                )
                state.best_effort_notice = " ".join(dict.fromkeys(notices))
                state.citation_stats = {
                    **_build_citation_sample_stats(
                        seed_publications_count=citing_bundle["seed_publications_count"],
                        citing_works_count=len(citing_works),
                        citing_authors_count=len(state.citing_authors),
                        citing_institutions_count=len(state.citing_institutions),
                        seed_citing_works_fetch_limit=min(state.request.max_citing_works, MAX_CITING_WORKS_PER_SEED),
                        built_from_publication_page=publication_page,
                        built_from_publication_page_size=publication_page_size,
                    ),
                    "citing_works_by_year": citation_network["citing_works_by_year"],
                }
                return state

            resolved_author = await _resolve_openalex_author(
                client,
                openalex_author_id=state.request.openalex_author_id,
                display_name=state.request.display_name,
                affiliation_hint=state.request.affiliation_hint,
            )
            raw_author = await client.get_author(resolved_author.get("id", ""))
            publication_items, seed_publications = await asyncio.gather(
                _fetch_openalex_publication_page(
                    client,
                    author_id=raw_author.get("id", ""),
                    page=publication_page,
                    page_size=publication_page_size,
                ),
                _fetch_openalex_seed_works(
                    client,
                    author_id=raw_author.get("id", ""),
                    seed_limit=seed_limit,
                ),
            )
            citing_bundle = await _load_citing_works_for_author(
                client,
                seed_publications,
                max_citing_works=state.request.max_citing_works,
                seed_limit=seed_limit,
            )
            citing_works = citing_bundle["citing_works"]
            citation_network = aggregate_citation_network(citing_works)
            web_honor_bundle = await enrich_honors_with_websearch(
                target_label="author",
                target_payload=build_author_profile(raw_author),
                citing_authors=citation_network["citing_authors"],
            )
            honor_bundle = web_honor_bundle

            state.author_profile = build_author_profile(raw_author)
            merged_profile_honors = list(dict.fromkeys(honor_bundle.get("target_honors") or []))
            if merged_profile_honors:
                state.author_profile["honors"] = merged_profile_honors
            merged_profile_titles = list(dict.fromkeys(honor_bundle.get("target_titles") or []))
            if merged_profile_titles:
                state.author_profile["titles"] = merged_profile_titles
            state.publication_stats = build_publication_sample_stats(
                loaded_publications_count=len(publication_items),
                linked_publications_count=len(publication_items),
                seed_publications_count=citing_bundle["seed_publications_count"],
            )
            state.publication_pagination = _build_publication_pagination(
                current_page=publication_page,
                page_size=publication_page_size,
                total_items=int(state.author_profile.get("works_count") or len(publication_items)),
            )
            state.publications = publication_items
            state.citing_works = citing_works
            state.citing_authors = citation_network["citing_authors"]
            state.citing_institutions = citation_network["citing_institutions"]
            state.honors_stats = honor_bundle["honor_stats"]
            state.matched_honorees = honor_bundle["matched_honorees"]
            notices = []
            if honor_bundle["best_effort_notice"]:
                notices.append(honor_bundle["best_effort_notice"])
            notices.append(
                f"Citation sampling was built from the author's top {citing_bundle['seed_publications_count']} cited works across the full OpenAlex profile; changing publication pages does not rebuild that sample."
            )
            notices.append(
                "Author totals are shown in the profile card; publication and citation stats below describe only the "
                "currently loaded sample."
            )
            state.best_effort_notice = " ".join(dict.fromkeys(notices))
            state.citation_stats = {
                **_build_citation_sample_stats(
                    seed_publications_count=citing_bundle["seed_publications_count"],
                    citing_works_count=len(citing_works),
                    citing_authors_count=len(state.citing_authors),
                    citing_institutions_count=len(state.citing_institutions),
                    seed_citing_works_fetch_limit=min(state.request.max_citing_works, MAX_CITING_WORKS_PER_SEED),
                    built_from_publication_page=publication_page,
                    built_from_publication_page_size=publication_page_size,
                ),
                "citing_works_by_year": citation_network["citing_works_by_year"],
            }
            return state
        finally:
            await client.close()
            await dblp_client.close()

    async def paper_detail_node(state: Paper2CitationState) -> Paper2CitationState:
        client = OpenAlexCitationClient()
        try:
            raw_work = await _resolve_work_from_request(client, state.request.doi_or_url)

            paper_detail = simplify_work(raw_work)
            if paper_detail.get("doi"):
                crossref = await client.get_crossref_metadata(paper_detail["doi"])
                paper_detail = merge_crossref_metadata(paper_detail, crossref)

            citing_raw_works = await client.get_citing_works(
                paper_detail["openalex_work_id"],
                per_page=state.request.max_citing_works,
            )
            citing_works = _prepare_work_items(citing_raw_works)[: state.request.max_citing_works]
            citation_network = aggregate_citation_network(citing_works)
            web_honor_bundle = await enrich_honors_with_websearch(
                target_label="paper",
                target_payload=paper_detail,
                citing_authors=citation_network["citing_authors"],
            )
            honor_bundle = web_honor_bundle

            state.paper_detail = paper_detail
            state.citing_works = citing_works
            state.citing_authors = citation_network["citing_authors"]
            state.citing_institutions = citation_network["citing_institutions"]
            state.honors_stats = honor_bundle["honor_stats"]
            state.matched_honorees = honor_bundle["matched_honorees"]
            state.best_effort_notice = honor_bundle["best_effort_notice"]
            state.citation_stats = {
                "citing_works_count": len(citing_works),
                "citing_authors_count": len(state.citing_authors),
                "citing_institutions_count": len(state.citing_institutions),
                "citing_works_by_year": citation_network["citing_works_by_year"],
                "paper_cited_by_count": int(raw_work.get("cited_by_count") or 0),
            }
            return state
        finally:
            await client.close()

    async def paper_context_node(state: Paper2CitationState) -> Paper2CitationState:
        client = OpenAlexCitationClient()
        try:
            target_raw_work = await _resolve_work_from_request(client, state.request.doi_or_url)
            target_work = simplify_work(target_raw_work)
            if target_work.get("doi"):
                crossref = await client.get_crossref_metadata(target_work["doi"])
                target_work = merge_crossref_metadata(target_work, crossref)

            citing_raw_work = await _resolve_work_from_request(
                client,
                state.request.citing_work_openalex_id.strip()
                or state.request.citing_work_doi_or_url.strip()
                or state.request.citing_work_title.strip(),
            )
            citing_work = simplify_work(citing_raw_work)
            context_bundle = await extract_citation_context_for_work(
                target_raw_work=target_raw_work,
                target_work=target_work,
                citing_raw_work=citing_raw_work,
                citing_work=citing_work,
            )
            state.paper_detail = target_work
            state.citation_context = {
                **context_bundle,
                "citing_paper": citing_work,
            }
            notices = []
            if context_bundle.get("best_effort_notice"):
                notices.append(context_bundle["best_effort_notice"])
            if not context_bundle.get("contexts"):
                notices.append(
                    "Only publicly readable HTML pages are searched in this first version; PDF-only or script-rendered pages may not expose inline citation text."
                )
            state.best_effort_notice = " ".join(dict.fromkeys(notices))
            return state
        except Exception as exc:
            log.error(f"[paper2citation] paper_context failed: {exc}")
            raise
        finally:
            await client.close()

    nodes = {
        "_start_": _start_,
        "author_search": author_search_node,
        "author_publications": author_publications_node,
        "author_detail": author_detail_node,
        "paper_detail": paper_detail_node,
        "paper_context": paper_context_node,
        "_end_": lambda state: state,
    }
    edges = [
        ("author_search", "_end_"),
        ("author_publications", "_end_"),
        ("author_detail", "_end_"),
        ("paper_detail", "_end_"),
        ("paper_context", "_end_"),
    ]
    builder.add_nodes(nodes).add_edges(edges).add_conditional_edge("_start_", _route)
    return builder
