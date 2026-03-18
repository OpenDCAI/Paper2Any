from __future__ import annotations

import asyncio
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import httpx

from dataflow_agent.logger import get_logger

log = get_logger(__name__)
_WEBSEARCH_DISABLED_UNTIL = 0.0
OPENALEX_BASE_URL = "https://api.openalex.org"
DBLP_PERSON_URL = "https://dblp.org/pid/{pid}.xml"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
DEFAULT_SOURCE_LOOKUP_CONCURRENCY = 12
DEFAULT_CITER_MAX_AUTHORS = 60
DEFAULT_CITER_BATCH_SIZE = 6

HONOR_LABELS = [
    "Turing Award",
    "Gödel Prize",
    "ACM Fellow",
    "ACM Distinguished Member",
    "IEEE Fellow",
    "AAAI Fellow",
    "ACL Fellow",
    "AAAS Fellow",
    "INFORMS Fellow",
    "SIAM Fellow",
    "Fellow of the Royal Society",
    "Member of NAS",
    "Member of NAE",
    "Member of NAM",
    "Member of Academia Europaea",
    "长江学者",
    "国家杰出青年科学基金获得者",
    "中国科学院院士",
    "中国工程院院士",
    "Microsoft Young Professorship",
    "CCF Young Scientist Award",
]

PRESTIGE_LABEL_PATTERNS = {
    "Turing Award": ["turing award"],
    "Gödel Prize": ["godel prize", "gödel prize"],
    "ACM Fellow": ["acm fellow", "association for computing machinery fellow"],
    "ACM Distinguished Member": ["acm distinguished member"],
    "IEEE Fellow": ["ieee fellow", "institute of electrical and electronics engineers fellow"],
    "AAAI Fellow": ["aaai fellow"],
    "ACL Fellow": ["acl fellow", "association for computational linguistics fellow"],
    "AAAS Fellow": ["aaas fellow", "american association for the advancement of science fellow"],
    "INFORMS Fellow": ["informs fellow"],
    "SIAM Fellow": ["siam fellow"],
    "Fellow of the Royal Society": ["fellow of the royal society", "royal society fellow"],
    "Member of NAS": ["member of the national academy of sciences", "elected to the national academy of sciences"],
    "Member of NAE": ["member of the national academy of engineering", "elected to the national academy of engineering"],
    "Member of NAM": ["member of the national academy of medicine", "elected to the national academy of medicine"],
    "Member of Academia Europaea": ["academia europaea"],
    "长江学者": ["长江学者", "chang jiang scholar", "cheung kong scholar", "cheung kong professor"],
    "国家杰出青年科学基金获得者": [
        "国家杰出青年科学基金获得者",
        "national science fund for distinguished young scholars",
        "distinguished young scholars",
    ],
    "中国科学院院士": [
        "中国科学院院士",
        "academic of the chinese academy of sciences",
        "academician of the chinese academy of sciences",
    ],
    "中国工程院院士": [
        "中国工程院院士",
        "academic of the chinese academy of engineering",
        "academician of the chinese academy of engineering",
    ],
    "Microsoft Young Professorship": [
        "microsoft young professorship",
    ],
    "CCF Young Scientist Award": [
        "ccf young scientist award",
        "young scientist award of china computer federation",
    ],
}

TITLE_PATTERNS = {
    "Professor": [
        " professor",
        "professor,",
        "professor of",
    ],
    "Associate Professor": [
        "associate professor",
    ],
    "Assistant Professor": [
        "assistant professor",
    ],
    "Distinguished Professor": [
        "distinguished professor",
        "boya distinguished professor",
        "chair professor",
    ],
    "Vice Dean": [
        "vice dean",
        "deputy dean",
    ],
    "Director": [
        "director of",
        "director,",
    ],
}

TARGET_TITLE_LABELS = [
    "Professor",
    "Associate Professor",
    "Assistant Professor",
    "Distinguished Professor",
    "Vice Dean",
    "Director",
]

WIKIDATA_PRESTIGE_CLAIM_PROPERTIES = {
    "P39",   # position held
    "P166",  # award received
    "P463",  # member of
    "P1416", # affiliation / association
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _normalize_base_url(raw: str) -> str:
    value = (raw or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def _normalize_lookup_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().casefold())


def _unique_strings(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = (value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _extract_qid(url: str) -> str:
    match = re.search(r"/(Q\d+)(?:[/?#]|$)", (url or "").strip())
    return match.group(1) if match else ""


def _extract_dblp_pid(raw: str) -> str:
    value = (raw or "").strip().rstrip("/")
    if not value:
        return ""
    if "/pid/" in value:
        return value.split("/pid/", 1)[1]
    return value


def _match_prestige_labels(texts: List[str]) -> List[str]:
    haystack = " | ".join(_normalize_lookup_text(text) for text in texts if text).strip()
    if not haystack:
        return []

    labels: List[str] = []
    for label, patterns in PRESTIGE_LABEL_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            labels.append(label)
    return labels


def _match_titles(texts: List[str]) -> List[str]:
    haystack = " | ".join(_normalize_lookup_text(text) for text in texts if text).strip()
    if not haystack:
        return []

    titles: List[str] = []
    for title, patterns in TITLE_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            titles.append(title)
    return titles


def _build_honor_stats(matched_honorees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for item in matched_honorees:
        label = (item.get("honor_label") or "").strip()
        if not label:
            continue
        bucket = buckets.setdefault(
            label,
            {"honor_label": label, "count": 0, "matched_authors": []},
        )
        bucket["count"] += 1
        bucket["matched_authors"].append(item)
    return sorted(buckets.values(), key=lambda item: (-item["count"], item["honor_label"]))


def _candidate_display_identity(candidate: Dict[str, Any]) -> str:
    parts: List[str] = []
    if candidate.get("display_name"):
        parts.append(f"name={candidate['display_name']}")
    if candidate.get("openalex_author_id"):
        parts.append(f"openalex_author_id={candidate['openalex_author_id']}")
    if candidate.get("affiliations"):
        parts.append(f"affiliations={', '.join(candidate['affiliations'])}")
    if candidate.get("citing_works_count"):
        parts.append(f"citing_works_count={int(candidate['citing_works_count'])}")
    return "; ".join(parts)


async def _http_get_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    response = await client.get(url, params=params or {})
    response.raise_for_status()
    return response.json()


async def _fetch_openalex_author_identity(
    client: httpx.AsyncClient,
    openalex_author_id: str,
) -> Dict[str, Any]:
    short_id = (openalex_author_id or "").strip().upper()
    if not short_id:
        return {}
    if short_id.startswith("HTTPS://OPENALEX.ORG/"):
        short_id = short_id.rsplit("/", 1)[-1]
    data = await _http_get_json(client, f"{OPENALEX_BASE_URL}/authors/{short_id}")
    ids = data.get("ids") or {}
    affiliations = [
        item.get("display_name", "").strip()
        for item in (data.get("last_known_institutions") or [])
        if item.get("display_name")
    ]
    return {
        "display_name": (data.get("display_name") or "").strip(),
        "affiliations": _unique_strings(affiliations),
        "dblp_id": _extract_dblp_pid(ids.get("dblp", "") or ""),
        "orcid": (ids.get("orcid") or "").strip(),
    }


def _xml_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


async def _fetch_dblp_identity(
    client: httpx.AsyncClient,
    dblp_id: str,
) -> Dict[str, Any]:
    pid = _extract_dblp_pid(dblp_id)
    if not pid:
        return {}
    response = await client.get(DBLP_PERSON_URL.format(pid=pid))
    response.raise_for_status()
    root = ET.fromstring(response.text)
    person = root.find("person")
    if person is None:
        return {}
    urls = [_xml_text(url) for url in person.findall("url") if _xml_text(url)]
    return {
        "display_name": _xml_text(person.find("author")) or root.attrib.get("name", ""),
        "affiliations": [
            _xml_text(note)
            for note in person.findall("note")
            if note.get("type") == "affiliation" and _xml_text(note)
        ],
        "wikidata_qids": [qid for qid in (_extract_qid(url) for url in urls) if qid],
        "urls": urls,
    }


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def _fetch_public_page_texts(
    client: httpx.AsyncClient,
    urls: List[str],
) -> List[str]:
    texts: List[str] = []
    allowed_urls: List[str] = []
    for raw in urls:
        url = (raw or "").strip()
        if not url:
            continue
        if any(blocked in url for blocked in ["orcid.org/", "scholar.google.com/"]):
            continue
        if url not in allowed_urls:
            allowed_urls.append(url)
    for url in allowed_urls[:3]:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            if "text/html" not in content_type:
                continue
            text = _html_to_text(response.text)
            if text:
                texts.append(text[:12000])
        except Exception as exc:
            log.debug(f"[paper2citation] public page fetch skipped for {url}: {exc}")
    return texts


async def _fetch_wikidata_texts(
    client: httpx.AsyncClient,
    qids: List[str],
) -> List[str]:
    normalized_qids = _unique_strings([qid for qid in qids if qid])
    if not normalized_qids:
        return []

    entity_data = await _http_get_json(
        client,
        WIKIDATA_API_URL,
        params={
            "action": "wbgetentities",
            "ids": "|".join(normalized_qids),
            "props": "labels|claims",
            "languages": "en|zh",
            "format": "json",
        },
    )
    entities = entity_data.get("entities") or {}
    referenced_ids: List[str] = []
    texts: List[str] = []

    for entity in entities.values():
        labels = entity.get("labels") or {}
        texts.extend(
            label_data.get("value", "").strip()
            for label_data in labels.values()
            if label_data.get("value")
        )
        claims = entity.get("claims") or {}
        for prop, claim_list in claims.items():
            if prop not in WIKIDATA_PRESTIGE_CLAIM_PROPERTIES:
                continue
            for claim in claim_list or []:
                datavalue = (((claim.get("mainsnak") or {}).get("datavalue")) or {}).get("value")
                if isinstance(datavalue, dict) and datavalue.get("id"):
                    referenced_ids.append(datavalue["id"])
                elif isinstance(datavalue, str):
                    texts.append(datavalue)

    referenced_ids = _unique_strings(referenced_ids)
    if referenced_ids:
        referenced_data = await _http_get_json(
            client,
            WIKIDATA_API_URL,
            params={
                "action": "wbgetentities",
                "ids": "|".join(referenced_ids),
                "props": "labels",
                "languages": "en|zh",
                "format": "json",
            },
        )
        for entity in (referenced_data.get("entities") or {}).values():
            for label_data in (entity.get("labels") or {}).values():
                if label_data.get("value"):
                    texts.append(label_data["value"].strip())

    return _unique_strings(texts)


async def _source_first_subject_labels(
    client: httpx.AsyncClient,
    subject: Dict[str, Any],
) -> Dict[str, Any]:
    display_name = (subject.get("display_name") or subject.get("name_or_title") or "").strip()
    affiliations = [item.strip() for item in (subject.get("affiliations") or []) if str(item).strip()]
    dblp_id = (subject.get("dblp_id") or "").strip()
    openalex_author_id = (subject.get("openalex_author_id") or "").strip()
    homepage = (subject.get("homepage") or "").strip()
    wikidata_qids: List[str] = []
    public_urls: List[str] = [homepage] if homepage else []

    openalex_bundle: Dict[str, Any] = {}
    dblp_bundle: Dict[str, Any] = {}

    if openalex_author_id and dblp_id:
        openalex_result, dblp_result = await asyncio.gather(
            _fetch_openalex_author_identity(client, openalex_author_id),
            _fetch_dblp_identity(client, dblp_id),
            return_exceptions=True,
        )
        if isinstance(openalex_result, Exception):
            log.debug(f"[paper2citation] source-first OpenAlex identity fetch skipped: {openalex_result}")
        else:
            openalex_bundle = openalex_result
        if isinstance(dblp_result, Exception):
            log.debug(f"[paper2citation] source-first DBLP identity fetch skipped: {dblp_result}")
        else:
            dblp_bundle = dblp_result
    else:
        try:
            if openalex_author_id:
                openalex_bundle = await _fetch_openalex_author_identity(client, openalex_author_id)
        except Exception as exc:
            log.debug(f"[paper2citation] source-first OpenAlex identity fetch skipped: {exc}")

        if openalex_bundle.get("display_name"):
            display_name = openalex_bundle["display_name"]
        if openalex_bundle.get("affiliations"):
            affiliations = openalex_bundle["affiliations"]
        if openalex_bundle.get("dblp_id"):
            dblp_id = openalex_bundle["dblp_id"]

        try:
            if dblp_id:
                dblp_bundle = await _fetch_dblp_identity(client, dblp_id)
        except Exception as exc:
            log.debug(f"[paper2citation] source-first DBLP identity fetch skipped: {exc}")

    if openalex_bundle.get("display_name"):
        display_name = openalex_bundle["display_name"]
    if openalex_bundle.get("affiliations"):
        affiliations = openalex_bundle["affiliations"]
    if openalex_bundle.get("dblp_id"):
        dblp_id = openalex_bundle["dblp_id"]

    if dblp_bundle.get("display_name"):
        display_name = re.sub(r"\s+\d{4}$", "", dblp_bundle["display_name"])
    if dblp_bundle.get("affiliations"):
        affiliations = _unique_strings(dblp_bundle["affiliations"])
    wikidata_qids.extend(dblp_bundle.get("wikidata_qids") or [])
    public_urls.extend(dblp_bundle.get("urls") or [])

    structured_claim_labels: List[str] = []
    public_page_texts: List[str] = []
    wikidata_result, public_page_result = await asyncio.gather(
        _fetch_wikidata_texts(client, wikidata_qids) if wikidata_qids else asyncio.sleep(0, result=[]),
        _fetch_public_page_texts(client, public_urls) if public_urls else asyncio.sleep(0, result=[]),
        return_exceptions=True,
    )
    if isinstance(wikidata_result, Exception):
        log.debug(f"[paper2citation] source-first Wikidata fetch skipped: {wikidata_result}")
    else:
        structured_claim_labels = wikidata_result
    if isinstance(public_page_result, Exception):
        log.debug(f"[paper2citation] source-first public page fetch skipped: {public_page_result}")
    else:
        public_page_texts = public_page_result

    combined_texts = [*structured_claim_labels, *public_page_texts]
    return {
        "display_name": display_name,
        "affiliations": _unique_strings(affiliations),
        "prestige_labels": _match_prestige_labels(combined_texts),
        "titles": _match_titles(combined_texts),
    }


async def enrich_honors_from_sources(
    *,
    target_label: str,
    target_payload: Dict[str, Any],
    citing_authors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    timeout = float(os.getenv("PAPER2CITATION_WEBSEARCH_TIMEOUT_SECONDS") or "45")
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122 Safari/537.36"},
    ) as client:
        target_honors: List[str] = []
        if target_label == "author":
            target_identity = await _source_first_subject_labels(
                client,
                {
                    "display_name": target_payload.get("display_name", ""),
                    "affiliations": target_payload.get("affiliations") or target_payload.get("institutions") or [],
                    "dblp_id": target_payload.get("dblp_id", ""),
                    "openalex_author_id": target_payload.get("openalex_author_id", ""),
                    "homepage": target_payload.get("homepage", ""),
                },
            )
            target_honors = target_identity["prestige_labels"]
            target_titles = target_identity["titles"]
        else:
            target_titles = []

        semaphore = asyncio.Semaphore(
            max(1, int(os.getenv("PAPER2CITATION_WEBSEARCH_SOURCE_CONCURRENCY") or str(DEFAULT_SOURCE_LOOKUP_CONCURRENCY)))
        )

        async def _resolve_candidate(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
            async with semaphore:
                identity = await _source_first_subject_labels(
                    client,
                    {
                        "display_name": candidate.get("display_name", ""),
                        "affiliations": candidate.get("affiliations") or [],
                        "openalex_author_id": candidate.get("openalex_author_id", ""),
                    },
                )
            matched: List[Dict[str, Any]] = []
            for label in identity["prestige_labels"]:
                matched.append(
                    {
                        "display_name": candidate.get("display_name", "").strip(),
                        "canonical_name": identity["display_name"] or candidate.get("display_name", "").strip(),
                        "honor_label": label,
                        "openalex_author_id": candidate.get("openalex_author_id", "").strip(),
                        "affiliations": identity["affiliations"] or candidate.get("affiliations") or [],
                        "citing_works_count": int(candidate.get("citing_works_count") or 0),
                    }
                )
            return matched

        resolved_lists = await asyncio.gather(*[_resolve_candidate(item) for item in citing_authors], return_exceptions=True)
        matched_honorees: List[Dict[str, Any]] = []
        for result in resolved_lists:
            if isinstance(result, Exception):
                continue
            matched_honorees.extend(result)

    notice = ""
    if target_honors or matched_honorees:
        notice = "Exact identity sources (OpenAlex / DBLP / Wikidata) were used first for prestige-label matching."
    return {
        "target_honors": _unique_strings(target_honors),
        "target_titles": _unique_strings(target_titles),
        "honor_stats": _build_honor_stats(matched_honorees),
        "matched_honorees": matched_honorees,
        "best_effort_notice": notice,
    }


def is_honor_enrichment_enabled() -> bool:
    global _WEBSEARCH_DISABLED_UNTIL
    if time.time() < _WEBSEARCH_DISABLED_UNTIL:
        return False
    return _env_flag("PAPER2CITATION_WEBSEARCH_ENABLED", False)


def _websearch_config() -> Dict[str, str]:
    api_key = (os.getenv("PAPER2CITATION_WEBSEARCH_API_KEY") or "").strip()
    base_url = _normalize_base_url(os.getenv("PAPER2CITATION_WEBSEARCH_API_URL") or "")
    if not api_key or not base_url:
        raise RuntimeError("paper2citation websearch config is incomplete")
    return {"api_key": api_key, "base_url": base_url.rstrip("/")}


def _configured_model_candidates() -> List[str]:
    primary = (os.getenv("PAPER2CITATION_WEBSEARCH_MODEL") or "").strip()
    fallback_raw = (os.getenv("PAPER2CITATION_WEBSEARCH_FALLBACK_MODELS") or "").strip()
    ordered: List[str] = []
    for item in [primary, *fallback_raw.split(",")]:
        model = item.strip()
        if model and model not in ordered:
            ordered.append(model)
    return ordered


def _extract_json_block(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _candidate_payloads(citing_authors: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for index, item in enumerate(citing_authors[:limit], start=1):
        name = (item.get("display_name") or "").strip()
        if not name:
            continue
        payloads.append(
            {
                "candidate_index": index,
                "display_name": name,
                "openalex_author_id": (item.get("openalex_author_id") or "").strip(),
                "affiliations": [str(value).strip() for value in (item.get("affiliations") or []) if str(value).strip()],
                "citing_works_count": int(item.get("citing_works_count") or 0),
            }
        )
    return payloads


def _make_target_author_prompt(target_payload: Dict[str, Any]) -> str:
    target_name = (target_payload.get("display_name") or target_payload.get("title") or "").strip()
    target_affiliations = target_payload.get("affiliations") or target_payload.get("institutions") or []
    target_meta = {
        "display_name": target_name,
        "affiliations": [str(value).strip() for value in target_affiliations if str(value).strip()],
        "works_count": int(target_payload.get("works_count") or 0),
        "cited_by_count": int(target_payload.get("cited_by_count") or 0),
        "doi": target_payload.get("doi", ""),
        "dblp_id": target_payload.get("dblp_id", ""),
        "openalex_author_id": target_payload.get("openalex_author_id", ""),
        "orcid": target_payload.get("orcid", ""),
        "homepage": target_payload.get("homepage", ""),
    }
    return f"""
You are enriching target-author prestige metadata for a citation explorer.
If your model supports built-in browsing or search, use it. Return valid JSON only.

Allowed honor labels:
{json.dumps(HONOR_LABELS, ensure_ascii=False)}

Allowed target titles:
{json.dumps(TARGET_TITLE_LABELS, ensure_ascii=False)}

Target author (must disambiguate by institution and IDs, not by name alone):
{json.dumps(target_meta, ensure_ascii=False, indent=2)}

Return JSON with this exact shape:
{{
  "target_honors": ["..."],
  "target_titles": ["..."],
  "matched_honorees": [],
  "best_effort_notice": "one short sentence"
}}

Rules:
- Identify the exact person using affiliation, DBLP/OpenAlex/ORCID/homepage information.
- `target_titles` must be chosen from the allowed target titles list above.
- `target_honors` must use only labels from the allowed list above.
- Do not return generic claims without disambiguating the person.
- If unsure, leave the array empty.
- Prefer precision over recall.
""".strip()


def _make_citer_batch_prompt(
    *,
    target_label: str,
    target_payload: Dict[str, Any],
    candidate_batch: List[Dict[str, Any]],
) -> str:
    target_meta = {
        "label": target_label,
        "display_name_or_title": (target_payload.get("display_name") or target_payload.get("title") or "").strip(),
        "affiliations": [str(value).strip() for value in (target_payload.get("affiliations") or target_payload.get("institutions") or []) if str(value).strip()],
        "doi": target_payload.get("doi", ""),
        "dblp_id": target_payload.get("dblp_id", ""),
        "openalex_author_id": target_payload.get("openalex_author_id", ""),
        "homepage": target_payload.get("homepage", ""),
    }
    batch_lines = [
        {
            "candidate_index": item["candidate_index"],
            "identity": _candidate_display_identity(item),
        }
        for item in candidate_batch
    ]
    return f"""
You are enriching citer prestige metadata for a citation explorer.
If your model supports built-in browsing or search, use it. Return valid JSON only.

Allowed honor labels:
{json.dumps(HONOR_LABELS, ensure_ascii=False)}

Target context:
{json.dumps(target_meta, ensure_ascii=False, indent=2)}

Candidate citing authors for this batch:
{json.dumps(batch_lines, ensure_ascii=False, indent=2)}

Return JSON with this exact shape:
{{
  "target_honors": [],
  "target_titles": [],
  "matched_honorees": [
    {{
      "candidate_index": 1,
      "canonical_name": "canonical person name if different, else same",
      "honor_label": "one allowed label",
      "evidence_summary": "short evidence summary based on the exact candidate"
    }}
  ],
  "best_effort_notice": "one short sentence"
}}

Rules:
- Only evaluate people in the candidate citing-author batch above.
- Use `candidate_index` from the provided batch for every match.
- Disambiguate using institution and OpenAlex ID, not name alone.
- Return only high-confidence honors from the allowed list.
- Do not return titles here.
- If unsure, leave `matched_honorees` empty.
""".strip()


async def _chat_completion_json(prompt: str, model: str) -> Dict[str, Any]:
    config = _websearch_config()
    request_payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": int(os.getenv("PAPER2CITATION_WEBSEARCH_MAX_OUTPUT_TOKENS") or "1200"),
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(float(os.getenv("PAPER2CITATION_WEBSEARCH_TIMEOUT_SECONDS") or "45"))) as raw_client:
        response = await raw_client.post(
            f"{config['base_url']}/chat/completions",
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            },
            json=request_payload,
        )
        response.raise_for_status()
        data = response.json()
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
    return _extract_json_block(content)


async def _call_model_with_fallback(prompt: str) -> Dict[str, Any]:
    timeout_seconds = float(os.getenv("PAPER2CITATION_WEBSEARCH_TIMEOUT_SECONDS") or "45")
    models = _configured_model_candidates()
    if not models:
        raise RuntimeError("PAPER2CITATION_WEBSEARCH_MODEL is not configured")

    last_error: Exception | None = None
    for model in models:
        try:
            return await asyncio.wait_for(
                _chat_completion_json(prompt, model),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            last_error = exc
            log.warning(f"[paper2citation] websearch chat model failed, trying next fallback: model={model}, error={exc}")
            continue

    raise RuntimeError(f"all configured websearch models failed: {last_error}")


def _normalize_llm_bundle(
    bundle: Dict[str, Any],
    citing_authors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    allowed_honor_labels = {label.casefold(): label for label in HONOR_LABELS}
    target_honors = [
        allowed_honor_labels[str(item).strip().casefold()]
        for item in (bundle.get("target_honors") or [])
        if str(item).strip().casefold() in allowed_honor_labels
    ]
    target_titles = [
        str(item).strip()
        for item in (bundle.get("target_titles") or [])
        if str(item).strip()
    ]
    candidate_lookup: Dict[int, Dict[str, Any]] = {
        index: item
        for index, item in enumerate(citing_authors, start=1)
        if (item.get("display_name") or "").strip()
    }
    candidate_name_lookup: Dict[str, Dict[str, Any]] = {
        (item.get("display_name") or "").strip().casefold(): item
        for item in citing_authors
        if (item.get("display_name") or "").strip()
    }
    matched_honorees: List[Dict[str, Any]] = []
    for item in bundle.get("matched_honorees") or []:
        honor_key = str(item.get("honor_label") or "").strip().casefold()
        if honor_key not in allowed_honor_labels:
            continue
        candidate_index = item.get("candidate_index")
        try:
            candidate = candidate_lookup.get(int(candidate_index))
        except (TypeError, ValueError):
            candidate = None
        if not candidate:
            display_name = str(item.get("display_name") or "").strip().casefold()
            candidate = candidate_name_lookup.get(display_name)
        if not candidate:
            continue
        display_name = (candidate.get("display_name") or "").strip()
        if not display_name:
            continue
        matched_honorees.append(
            {
                "display_name": display_name,
                "canonical_name": str(item.get("canonical_name") or display_name).strip(),
                "honor_label": allowed_honor_labels[honor_key],
                "openalex_author_id": str(candidate.get("openalex_author_id") or "").strip(),
                "affiliations": [str(value).strip() for value in (candidate.get("affiliations") or []) if str(value).strip()],
                "citing_works_count": int(candidate.get("citing_works_count") or 0),
                "evidence_summary": str(item.get("evidence_summary") or "").strip(),
            }
        )
    return {
        "target_honors": target_honors,
        "target_titles": target_titles,
        "matched_honorees": matched_honorees,
        "best_effort_notice": str(bundle.get("best_effort_notice") or "").strip(),
    }


async def enrich_honors_with_websearch(
    *,
    target_label: str,
    target_payload: Dict[str, Any],
    citing_authors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    global _WEBSEARCH_DISABLED_UNTIL
    source_bundle = await enrich_honors_from_sources(
        target_label=target_label,
        target_payload=target_payload,
        citing_authors=citing_authors,
    )

    notices: List[str] = []
    if source_bundle.get("best_effort_notice"):
        notices.append(source_bundle["best_effort_notice"].strip())

    should_try_websearch = bool(
        is_honor_enrichment_enabled()
        and (
            (target_label == "author" and (not source_bundle.get("target_honors") or not source_bundle.get("target_titles") or bool(citing_authors)))
            or (target_label == "paper" and bool(citing_authors))
        )
    )

    if should_try_websearch:
        try:
            llm_target_honors: List[str] = []
            llm_target_titles: List[str] = []
            llm_matched_honorees: List[Dict[str, Any]] = []

            if target_label == "author":
                target_prompt = _make_target_author_prompt(target_payload)
                target_bundle = _normalize_llm_bundle(
                    await _call_model_with_fallback(target_prompt),
                    [],
                )
                llm_target_honors = _unique_strings(target_bundle.get("target_honors") or [])
                llm_target_titles = _unique_strings(target_bundle.get("target_titles") or [])

            max_authors = int(
                os.getenv("PAPER2CITATION_WEBSEARCH_MAX_AUTHORS") or str(DEFAULT_CITER_MAX_AUTHORS)
            )
            batch_size = max(
                1,
                int(os.getenv("PAPER2CITATION_WEBSEARCH_BATCH_SIZE") or str(DEFAULT_CITER_BATCH_SIZE)),
            )
            candidate_payloads = _candidate_payloads(citing_authors, limit=max_authors)
            if candidate_payloads:
                batch_prompts = [
                    _make_citer_batch_prompt(
                        target_label=target_label,
                        target_payload=target_payload,
                        candidate_batch=candidate_payloads[index:index + batch_size],
                    )
                    for index in range(0, len(candidate_payloads), batch_size)
                ]
                batch_results = await asyncio.gather(
                    *[_call_model_with_fallback(prompt) for prompt in batch_prompts],
                    return_exceptions=True,
                )
                for result in batch_results:
                    if isinstance(result, Exception):
                        continue
                    normalized = _normalize_llm_bundle(result, citing_authors)
                    llm_matched_honorees.extend(normalized.get("matched_honorees") or [])

            if llm_target_honors:
                source_bundle["target_honors"] = _unique_strings([*source_bundle.get("target_honors", []), *llm_target_honors])
            if llm_target_titles:
                source_bundle["target_titles"] = _unique_strings([*source_bundle.get("target_titles", []), *llm_target_titles])
            if llm_matched_honorees:
                merged = list(source_bundle.get("matched_honorees") or [])
                seen_pairs = {
                    (
                        (item.get("openalex_author_id") or item.get("display_name") or "").strip().casefold(),
                        (item.get("honor_label") or "").strip(),
                    )
                    for item in merged
                }
                for item in llm_matched_honorees:
                    key = (
                        (item.get("openalex_author_id") or item.get("display_name") or "").strip().casefold(),
                        (item.get("honor_label") or "").strip(),
                    )
                    if not key[0] or not key[1] or key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    merged.append(item)
                source_bundle["matched_honorees"] = merged

            if llm_target_honors or llm_target_titles or llm_matched_honorees:
                notices.append(
                    "Structured identity sources were used first; web search was then used to supplement target-author honors/titles and map verified citer honors back to the currently loaded candidate list."
                )
        except Exception as exc:
            _WEBSEARCH_DISABLED_UNTIL = time.time() + float(
                os.getenv("PAPER2CITATION_WEBSEARCH_DISABLE_SECONDS") or "600"
            )
            log.warning(f"[paper2citation] target-honor websearch fallback failed, disabled temporarily: {exc}")
            notices.append(
                "Web search enrichment is temporarily unavailable; the page falls back to structured identity sources only."
            )

    notices.append(
        "Matched citer honors are limited to the currently loaded citing-author candidates and are mapped back to the page by exact candidate index or exact name."
    )
    source_bundle["best_effort_notice"] = " ".join(dict.fromkeys(value for value in notices if value))
    source_bundle["honor_stats"] = _build_honor_stats(source_bundle.get("matched_honorees") or [])
    return source_bundle
