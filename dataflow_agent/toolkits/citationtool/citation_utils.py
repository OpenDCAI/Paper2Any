from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote, unquote, urlparse

import httpx


OPENALEX_BASE_URL = "https://api.openalex.org"
CROSSREF_BASE_URL = "https://api.crossref.org"
DBLP_AUTHOR_SEARCH_URL = "https://dblp.org/search/author/api"
DBLP_AUTHOR_PERSON_URL = "https://dblp.org/pid/{pid}.xml"
DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)


class CitationDataError(RuntimeError):
    """Raised when the citation data provider cannot satisfy a request."""


def normalize_name(name: str) -> str:
    value = (
        (name or "")
        .replace("ä", "ae")
        .replace("Ä", "Ae")
        .replace("ö", "oe")
        .replace("Ö", "Oe")
        .replace("ü", "ue")
        .replace("Ü", "Ue")
        .replace("ß", "ss")
    )
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value)
    return value


def _normalize_doi_suffix(value: str) -> str:
    return value.rstrip(").,;]}>\"' ").strip()


def extract_doi_from_input(raw: str) -> str:
    text = unquote((raw or "").strip())
    if not text:
        return ""

    parsed = urlparse(text)
    candidates = [text]
    if parsed.scheme and parsed.netloc:
        candidates.extend([parsed.path, f"{parsed.netloc}{parsed.path}", parsed.query])

    for candidate in candidates:
        match = DOI_RE.search(candidate)
        if match:
            return _normalize_doi_suffix(match.group(1).lower())

    return ""


def parse_openalex_id(raw: str, expected_prefix: str) -> str:
    value = (raw or "").strip().rstrip("/")
    if not value:
        return ""
    if value.startswith("https://openalex.org/"):
        value = value.rsplit("/", 1)[-1]
    value = value.upper()
    return value if value.startswith(expected_prefix) else ""


def _unique_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value:
            continue
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _get_affiliations(author: Dict[str, Any]) -> List[str]:
    institutions = author.get("last_known_institutions") or author.get("affiliations") or []
    names: List[str] = []
    for item in institutions:
        if isinstance(item, dict):
            display_name = item.get("display_name") or item.get("institution", {}).get("display_name")
            if display_name:
                names.append(display_name)
    return _unique_strings(names)


def _strip_dblp_numeric_suffix(name: str) -> str:
    return re.sub(r"\s+\d{4}$", "", (name or "").strip())


def _extract_dblp_pid(url: str) -> str:
    value = (url or "").strip().rstrip("/")
    if "/pid/" not in value:
        return ""
    return value.split("/pid/", 1)[1]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def simplify_dblp_author_candidate(hit: Dict[str, Any]) -> Dict[str, Any]:
    info = hit.get("info") or {}
    author_name = _strip_dblp_numeric_suffix(info.get("author", "") or "")
    notes = info.get("notes") or {}
    raw_notes = _as_list(notes.get("note"))
    affiliations = [
        item.get("text", "")
        for item in raw_notes
        if isinstance(item, dict) and item.get("@type") == "affiliation" and item.get("text")
    ]
    return {
        "openalex_author_id": "",
        "dblp_id": _extract_dblp_pid(info.get("url", "")),
        "orcid": "",
        "display_name": author_name,
        "affiliations": _unique_strings(affiliations),
        "works_count": 0,
        "cited_by_count": 0,
        "source": "dblp",
        "source_score": int(hit.get("@score") or 0),
    }


def simplify_author_candidate(author: Dict[str, Any]) -> Dict[str, Any]:
    ids = author.get("ids") or {}
    return {
        "openalex_author_id": parse_openalex_id(author.get("id", ""), "A"),
        "dblp_id": ids.get("dblp", "") or "",
        "orcid": ids.get("orcid", "") or "",
        "display_name": author.get("display_name", "") or "",
        "affiliations": _get_affiliations(author),
        "works_count": int(author.get("works_count") or 0),
        "cited_by_count": int(author.get("cited_by_count") or 0),
        "h_index": int(author.get("summary_stats", {}).get("h_index") or 0),
    }


def _name_match_score(query: str, candidate_name: str) -> int:
    normalized_query = normalize_name(query)
    normalized_candidate = normalize_name(candidate_name)
    if not normalized_query or not normalized_candidate:
        return 0
    if normalized_query == normalized_candidate:
        return 4

    query_tokens = [normalize_name(token) for token in re.split(r"\s+", query.strip()) if token.strip()]
    candidate_tokens = [normalize_name(token) for token in re.split(r"\s+", candidate_name.strip()) if token.strip()]
    if query_tokens and candidate_tokens and query_tokens == candidate_tokens:
        return 3
    if query_tokens and all(token in candidate_tokens for token in query_tokens):
        return 2
    if normalized_query in normalized_candidate or normalized_candidate in normalized_query:
        return 1
    return 0


def _is_equivalent_person_name(query: str, candidate_name: str) -> bool:
    normalized_query = normalize_name(query)
    normalized_candidate = normalize_name(candidate_name)
    if not normalized_query or not normalized_candidate:
        return False
    if normalized_query == normalized_candidate:
        return True

    query_tokens = [normalize_name(token) for token in re.split(r"\s+", query.strip()) if token.strip()]
    candidate_tokens = [normalize_name(token) for token in re.split(r"\s+", candidate_name.strip()) if token.strip()]
    if len(query_tokens) >= 2 and len(query_tokens) == len(candidate_tokens):
        return sorted(query_tokens) == sorted(candidate_tokens)
    return False


def _affiliation_match_score(affiliation_hint: str, affiliations: List[str]) -> int:
    normalized_hint = normalize_name(affiliation_hint)
    if not normalized_hint:
        return 0
    score = 0
    for affiliation in affiliations:
        normalized_aff = normalize_name(affiliation)
        if not normalized_aff:
            continue
        if normalized_hint == normalized_aff:
            return 3
        if normalized_hint in normalized_aff or normalized_aff in normalized_hint:
            score = max(score, 2)
        else:
            hint_tokens = [token for token in re.split(r"\s+", affiliation_hint) if token.strip()]
            if hint_tokens and all(normalize_name(token) in normalized_aff for token in hint_tokens[:3]):
                score = max(score, 1)
    return score


def is_close_title(left: str, right: str) -> bool:
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm or left_norm in right_norm or right_norm in left_norm


def compute_publication_overlap_metrics(
    dblp_publications: List[Dict[str, Any]],
    openalex_publications: List[Dict[str, Any]],
) -> Dict[str, int]:
    dblp_dois = {
        (item.get("doi") or "").strip().lower()
        for item in dblp_publications
        if (item.get("doi") or "").strip()
    }
    openalex_dois = {
        (item.get("doi") or "").strip().lower()
        for item in openalex_publications
        if (item.get("doi") or "").strip()
    }
    doi_overlap_count = len(dblp_dois & openalex_dois)

    title_overlap_count = 0
    matched_openalex_indexes = set()
    for dblp_item in dblp_publications:
        dblp_title = dblp_item.get("title", "") or ""
        if not dblp_title:
            continue
        for openalex_index, openalex_item in enumerate(openalex_publications):
            if openalex_index in matched_openalex_indexes:
                continue
            if is_close_title(dblp_title, openalex_item.get("title", "") or ""):
                matched_openalex_indexes.add(openalex_index)
                title_overlap_count += 1
                break

    return {
        "doi_overlap_count": doi_overlap_count,
        "title_overlap_count": title_overlap_count,
    }


def evaluate_dblp_openalex_bridge(
    dblp_bundle: Dict[str, Any],
    raw_author: Dict[str, Any],
    openalex_publications: List[Dict[str, Any]],
) -> Dict[str, Any]:
    dblp_profile = dblp_bundle.get("author_profile") or {}
    dblp_publications = dblp_bundle.get("publications") or []
    overlap_metrics = compute_publication_overlap_metrics(dblp_publications, openalex_publications)
    dblp_affiliation_hint = ((dblp_profile.get("affiliations") or [""])[0] or "").strip()
    affiliation_score = _affiliation_match_score(dblp_affiliation_hint, _get_affiliations(raw_author))
    name_score = _name_match_score(
        dblp_profile.get("display_name", "") or "",
        raw_author.get("display_name", "") or "",
    )

    is_confident = bool(
        name_score >= 3
        and (
            overlap_metrics["doi_overlap_count"] >= 1
            or overlap_metrics["title_overlap_count"] >= 2
            or (
                overlap_metrics["title_overlap_count"] >= 1
                and affiliation_score >= 1
            )
            or affiliation_score >= 2
        )
    )

    evidence_parts = []
    if overlap_metrics["doi_overlap_count"] > 0:
        evidence_parts.append(f"{overlap_metrics['doi_overlap_count']} DOI overlaps")
    if overlap_metrics["title_overlap_count"] > 0:
        evidence_parts.append(f"{overlap_metrics['title_overlap_count']} title overlaps")
    if affiliation_score > 0:
        evidence_parts.append(f"affiliation score {affiliation_score}")
    evidence_text = ", ".join(evidence_parts) or "no publication or affiliation overlap"

    return {
        "accepted": is_confident,
        "name_score": name_score,
        "affiliation_score": affiliation_score,
        **overlap_metrics,
        "evidence_text": evidence_text,
    }


def build_publication_sample_stats(
    *,
    loaded_publications_count: int,
    linked_publications_count: int,
    seed_publications_count: int,
) -> Dict[str, int]:
    return {
        "loaded_publications_count": loaded_publications_count,
        "linked_publications_count": linked_publications_count,
        "unlinked_publications_count": max(loaded_publications_count - linked_publications_count, 0),
        "seed_publications_count": seed_publications_count,
    }


def _extract_work_authors(work: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        display_name = author.get("display_name")
        if display_name:
            names.append(display_name)
    return _unique_strings(names)


def _extract_work_institutions(work: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for authorship in work.get("authorships") or []:
        for institution in authorship.get("institutions") or []:
            display_name = institution.get("display_name")
            if display_name:
                names.append(display_name)
    return _unique_strings(names)


def _extract_doi(work: Dict[str, Any]) -> str:
    doi = (work.get("doi") or "").strip()
    if doi:
        return extract_doi_from_input(doi)
    ids = work.get("ids") or {}
    return extract_doi_from_input(ids.get("doi", ""))


def _xml_text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def simplify_work(work: Dict[str, Any]) -> Dict[str, Any]:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    return {
        "openalex_work_id": parse_openalex_id(work.get("id", ""), "W"),
        "doi": _extract_doi(work),
        "title": work.get("title", "") or "",
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date", "") or "",
        "venue": source.get("display_name", "") or work.get("host_venue", {}).get("display_name", "") or "",
        "type": work.get("type", "") or "",
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "authors": _extract_work_authors(work),
        "institutions": _extract_work_institutions(work),
        "landing_page_url": location.get("landing_page_url", "") or "",
    }


def aggregate_publication_stats(author: Dict[str, Any], works: List[Dict[str, Any]]) -> Dict[str, Any]:
    venue_counter: Counter[str] = Counter()
    year_counts: List[Dict[str, Any]] = []

    for item in author.get("counts_by_year") or []:
        year = item.get("year")
        works_count = int(item.get("works_count") or 0)
        if year and works_count:
            year_counts.append({"year": int(year), "count": works_count})

    for work in works:
        venue = work.get("venue", "")
        if venue:
            venue_counter[venue] += 1

    return {
        "works_count": int(author.get("works_count") or 0),
        "cited_by_count": int(author.get("cited_by_count") or 0),
        "works_by_year": sorted(year_counts, key=lambda item: item["year"], reverse=True),
        "top_venues": [
            {"venue": venue, "count": count}
            for venue, count in venue_counter.most_common(10)
        ],
    }


def build_author_profile(author: Dict[str, Any]) -> Dict[str, Any]:
    candidate = simplify_author_candidate(author)
    candidate["summary_stats"] = {
        "h_index": int(author.get("summary_stats", {}).get("h_index") or 0),
        "i10_index": int(author.get("summary_stats", {}).get("i10_index") or 0),
        "two_year_mean_citedness": float(author.get("summary_stats", {}).get("2yr_mean_citedness") or 0.0),
    }
    candidate["affiliation_text"] = ", ".join(candidate["affiliations"])
    return candidate


def aggregate_citation_network(citing_works: List[Dict[str, Any]]) -> Dict[str, Any]:
    author_stats: Dict[str, Dict[str, Any]] = {}
    institution_stats: Dict[str, Dict[str, Any]] = {}
    year_counter: Counter[int] = Counter()

    for work in citing_works:
        work_id = work.get("openalex_work_id", "")
        year = work.get("year")
        if isinstance(year, int):
            year_counter[year] += 1

        seen_author_ids = set()
        seen_institution_ids = set()
        for authorship in work.get("raw_authorships") or []:
            raw_author = authorship.get("author") or {}
            display_name = raw_author.get("display_name", "") or ""
            author_id = parse_openalex_id(raw_author.get("id", ""), "A") or normalize_name(display_name)
            if display_name and author_id and author_id not in seen_author_ids:
                seen_author_ids.add(author_id)
                record = author_stats.setdefault(
                    author_id,
                    {
                        "openalex_author_id": parse_openalex_id(raw_author.get("id", ""), "A"),
                        "display_name": display_name,
                        "affiliations": [],
                        "citing_works_count": 0,
                    },
                )
                record["citing_works_count"] += 1
                for institution in authorship.get("institutions") or []:
                    inst_name = institution.get("display_name", "") or ""
                    if inst_name and inst_name not in record["affiliations"]:
                        record["affiliations"].append(inst_name)

            for institution in authorship.get("institutions") or []:
                inst_name = institution.get("display_name", "") or ""
                if not inst_name:
                    continue
                inst_id = parse_openalex_id(institution.get("id", ""), "I") or inst_name
                if inst_id in seen_institution_ids:
                    continue
                seen_institution_ids.add(inst_id)
                record = institution_stats.setdefault(
                    inst_id,
                    {
                        "openalex_institution_id": parse_openalex_id(institution.get("id", ""), "I"),
                        "display_name": inst_name,
                        "country_code": institution.get("country_code", "") or "",
                        "type": institution.get("type", "") or "",
                        "citing_works_count": 0,
                    },
                )
                record["citing_works_count"] += 1

    citing_authors = sorted(
        author_stats.values(),
        key=lambda item: (-item["citing_works_count"], item["display_name"]),
    )
    citing_institutions = sorted(
        institution_stats.values(),
        key=lambda item: (-item["citing_works_count"], item["display_name"]),
    )
    return {
        "citing_authors": citing_authors,
        "citing_institutions": citing_institutions,
        "citing_works_by_year": [
            {"year": year, "count": count}
            for year, count in sorted(year_counter.items(), reverse=True)
        ],
    }


def resolve_doi_or_openalex_id(raw: str) -> Dict[str, str]:
    value = (raw or "").strip()
    doi = extract_doi_from_input(value)
    work_id = parse_openalex_id(value, "W")
    if not work_id and "openalex.org/W" in value:
        work_id = parse_openalex_id(value.rsplit("/", 1)[-1], "W")
    return {"doi": doi, "openalex_work_id": work_id}


class OpenAlexCitationClient:
    def __init__(self, timeout: float = 30.0):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    async def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            response = await self._client.get(url, params=params or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise CitationDataError(f"Upstream request failed: {detail}") from exc
        except httpx.HTTPError as exc:
            raise CitationDataError(f"Failed to connect to citation data provider: {exc}") from exc

    async def search_authors(self, author_name: str, per_page: int = 12) -> List[Dict[str, Any]]:
        fetch_size = min(max(per_page * 4, 50), 100)
        data = await self._get_json(
            f"{OPENALEX_BASE_URL}/authors",
            params={"search": author_name, "per-page": fetch_size, "sort": "cited_by_count:desc"},
        )
        results = data.get("results") or []
        results.sort(
            key=lambda item: (
                -_name_match_score(author_name, item.get("display_name", "") or ""),
                -(int(item.get("cited_by_count") or 0)),
                item.get("display_name", "") or "",
            )
        )
        exact_matches = [
            item for item in results
            if _name_match_score(author_name, item.get("display_name", "") or "") >= 3
        ]
        equivalent_matches = [
            item for item in results
            if _is_equivalent_person_name(author_name, item.get("display_name", "") or "")
        ]
        if equivalent_matches:
            results = equivalent_matches + [
                item for item in results
                if item not in equivalent_matches
            ]
        if exact_matches:
            seen_ids = set()
            ordered: List[Dict[str, Any]] = []
            for item in exact_matches + results:
                author_id = item.get("id", "")
                if author_id in seen_ids:
                    continue
                seen_ids.add(author_id)
                ordered.append(item)
            results = ordered
        if equivalent_matches:
            results = [
                item for item in results
                if _is_equivalent_person_name(author_name, item.get("display_name", "") or "")
            ] or results
        return results[:per_page]

    async def search_authors_with_affiliation(
        self,
        author_name: str,
        affiliation_hint: str = "",
        per_page: int = 50,
    ) -> List[Dict[str, Any]]:
        results = await self.search_authors(author_name, per_page=per_page)
        if not affiliation_hint.strip():
            return results
        results.sort(
            key=lambda item: (
                -_name_match_score(author_name, item.get("display_name", "") or ""),
                -_affiliation_match_score(affiliation_hint, _get_affiliations(item)),
                -(int(item.get("cited_by_count") or 0)),
            )
        )
        return results

    async def get_author(self, author_id: str) -> Dict[str, Any]:
        short_id = parse_openalex_id(author_id, "A")
        if not short_id:
            raise CitationDataError("Invalid OpenAlex author id")
        return await self._get_json(f"{OPENALEX_BASE_URL}/authors/{quote(short_id)}")

    async def get_author_by_orcid(self, orcid: str) -> Dict[str, Any]:
        normalized = (orcid or "").strip()
        if not normalized:
            raise CitationDataError("Invalid ORCID input")
        data = await self._get_json(
            f"{OPENALEX_BASE_URL}/authors",
            params={"filter": f"orcid:{normalized}", "per-page": 5},
        )
        results = data.get("results") or []
        if not results:
            raise CitationDataError("No OpenAlex author matched the provided ORCID")
        for item in results:
            ids = item.get("ids") or {}
            if (ids.get("orcid") or "").strip() == normalized:
                return item
        return results[0]

    async def get_author_works(
        self,
        author_id: str,
        *,
        per_page: int = 25,
        page: int = 1,
        sort: str = "publication_date:desc",
    ) -> List[Dict[str, Any]]:
        short_id = parse_openalex_id(author_id, "A")
        if not short_id:
            raise CitationDataError("Invalid OpenAlex author id")
        data = await self._get_json(
            f"{OPENALEX_BASE_URL}/works",
            params={
                "filter": f"author.id:https://openalex.org/{short_id}",
                "per-page": per_page,
                "page": max(page, 1),
                "sort": sort,
            },
        )
        return data.get("results") or []

    async def get_work(self, work_id: str) -> Dict[str, Any]:
        short_id = parse_openalex_id(work_id, "W")
        if not short_id:
            raise CitationDataError("Invalid OpenAlex work id")
        return await self._get_json(f"{OPENALEX_BASE_URL}/works/{quote(short_id)}")

    async def get_work_by_doi(self, doi: str) -> Dict[str, Any]:
        normalized = extract_doi_from_input(doi)
        if not normalized:
            raise CitationDataError("Invalid DOI input")
        return await self._get_json(f"{OPENALEX_BASE_URL}/works/https://doi.org/{quote(normalized, safe='')}")

    async def get_citing_works(self, work_id: str, per_page: int = 40) -> List[Dict[str, Any]]:
        short_id = parse_openalex_id(work_id, "W")
        if not short_id:
            raise CitationDataError("Invalid OpenAlex work id")
        data = await self._get_json(
            f"{OPENALEX_BASE_URL}/works",
            params={
                "filter": f"cites:https://openalex.org/{short_id}",
                "per-page": per_page,
                "sort": "publication_date:desc",
            },
        )
        return data.get("results") or []

    async def search_work_by_bibliographic(self, query: str) -> Dict[str, Any]:
        data = await self._get_json(
            f"{OPENALEX_BASE_URL}/works",
            params={"search": query, "per-page": 1},
        )
        results = data.get("results") or []
        if not results:
            raise CitationDataError("No paper matched the provided DOI or URL")
        return results[0]

    async def get_crossref_metadata(self, doi: str) -> Dict[str, Any]:
        normalized = extract_doi_from_input(doi)
        if not normalized:
            return {}
        data = await self._get_json(f"{CROSSREF_BASE_URL}/works/{quote(normalized, safe='')}")
        return data.get("message") or {}


class DBLPCitationClient:
    def __init__(self, timeout: float = 15.0):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    async def search_authors(self, author_name: str, per_page: int = 20) -> List[Dict[str, Any]]:
        try:
            response = await self._client.get(
                DBLP_AUTHOR_SEARCH_URL,
                params={"q": author_name, "format": "json", "h": per_page},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise CitationDataError(f"Failed to query DBLP: {exc}") from exc

        hits = (((data.get("result") or {}).get("hits") or {}).get("hit")) or []
        items = [simplify_dblp_author_candidate(hit) for hit in _as_list(hits)]
        equivalent_items = [
            item for item in items
            if _is_equivalent_person_name(author_name, item.get("display_name", "") or "")
        ]
        if equivalent_items:
            items = equivalent_items
        items.sort(
            key=lambda item: (
                -_name_match_score(author_name, item.get("display_name", "") or ""),
                -(int(item.get("source_score") or 0)),
            )
        )
        return items

    async def get_author_profile_and_works(
        self,
        dblp_id: str,
        *,
        max_records: int = 25,
        offset: int = 0,
    ) -> Dict[str, Any]:
        pid = (dblp_id or "").strip()
        if not pid:
            raise CitationDataError("dblp_id is required")

        try:
            response = await self._client.get(DBLP_AUTHOR_PERSON_URL.format(pid=quote(pid, safe="/")))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CitationDataError(f"Failed to query DBLP person page: {exc}") from exc

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise CitationDataError(f"Failed to parse DBLP person XML for {pid}") from exc

        person = root.find("person")
        if person is None:
            raise CitationDataError(f"DBLP person record not found for {pid}")

        display_name = _strip_dblp_numeric_suffix(person.findtext("author") or root.attrib.get("name", ""))
        urls = [(_xml_text(url) or "").strip() for url in person.findall("url")]
        affiliations = [
            (_xml_text(note) or "").strip()
            for note in person.findall("note")
            if note.get("type") == "affiliation"
        ]
        orcid = next((url for url in urls if "orcid.org/" in url), "")
        homepage = next(
            (
                url for url in urls
                if url
                and "orcid.org/" not in url
                and "scholar.google.com/" not in url
                and "wikidata.org/" not in url
            ),
            "",
        )

        all_records = root.findall("r")
        works: List[Dict[str, Any]] = []
        for record in all_records[offset: offset + max_records]:
            item = next(iter(record), None)
            if item is None:
                continue

            authors = [
                _strip_dblp_numeric_suffix(_xml_text(author))
                for author in item.findall("author")
                if _xml_text(author)
            ]
            raw_ee = [_xml_text(ee) for ee in item.findall("ee") if _xml_text(ee)]
            landing_page_url = raw_ee[0] if raw_ee else ""
            works.append(
                {
                    "openalex_work_id": "",
                    "doi": next((extract_doi_from_input(link) for link in raw_ee if extract_doi_from_input(link)), ""),
                    "title": _xml_text(item.find("title")),
                    "year": int(item.findtext("year") or 0) or None,
                    "publication_date": item.findtext("year") or "",
                    "venue": (
                        item.findtext("journal")
                        or item.findtext("booktitle")
                        or item.findtext("school")
                        or item.findtext("publisher")
                        or ""
                    ),
                    "type": item.tag or "",
                    "cited_by_count": 0,
                    "authors": _unique_strings(authors),
                    "institutions": [],
                    "landing_page_url": landing_page_url,
                    "raw_authorships": [],
                }
            )

        works_count = int(root.attrib.get("n") or len(works) or 0)
        author_profile = {
            "openalex_author_id": "",
            "dblp_id": pid,
            "orcid": orcid,
            "display_name": display_name,
            "affiliations": _unique_strings(affiliations),
            "works_count": works_count,
            "cited_by_count": 0,
            "h_index": 0,
            "summary_stats": {
                "h_index": 0,
                "i10_index": 0,
                "two_year_mean_citedness": 0.0,
            },
            "homepage": homepage,
            "affiliation_text": ", ".join(_unique_strings(affiliations)),
        }
        return {
            "author_profile": author_profile,
            "publications": works,
        }


def merge_crossref_metadata(work: Dict[str, Any], crossref_message: Dict[str, Any]) -> Dict[str, Any]:
    if not crossref_message:
        return work
    merged = dict(work)
    if not merged.get("title"):
        title = crossref_message.get("title") or []
        merged["title"] = title[0] if title else ""
    if not merged.get("venue"):
        container = crossref_message.get("container-title") or []
        merged["venue"] = container[0] if container else ""
    if not merged.get("publication_date"):
        parts = (((crossref_message.get("issued") or {}).get("date-parts") or [[None]])[0])
        if parts and parts[0]:
            merged["publication_date"] = "-".join(str(part) for part in parts if part is not None)
    return merged
