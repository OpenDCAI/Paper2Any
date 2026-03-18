from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.toolkits.citationtool.citation_utils import (
    _name_match_score,
    aggregate_citation_network,
    build_publication_sample_stats,
    evaluate_dblp_openalex_bridge,
    extract_doi_from_input,
    normalize_name,
    resolve_doi_or_openalex_id,
)
from dataflow_agent.toolkits.citationtool.honor_enrichment import (
    _match_titles,
    _normalize_base_url,
    _normalize_llm_bundle,
)


def test_extract_doi_from_raw_doi_and_url() -> None:
    assert extract_doi_from_input("10.1145/3746027.3758205") == "10.1145/3746027.3758205"
    assert (
        extract_doi_from_input("https://doi.org/10.1609/aaai.v39i18.34057")
        == "10.1609/aaai.v39i18.34057"
    )


def test_resolve_doi_or_openalex_id_prefers_explicit_inputs() -> None:
    assert resolve_doi_or_openalex_id("https://openalex.org/W1234567890") == {
        "doi": "",
        "openalex_work_id": "W1234567890",
    }
    assert resolve_doi_or_openalex_id("https://publisher.org/paper/10.1145/3746027.3758205") == {
        "doi": "10.1145/3746027.3758205",
        "openalex_work_id": "",
    }


def test_aggregate_citation_network_and_honors() -> None:
    citing_works = [
        {
            "openalex_work_id": "W1",
            "title": "Paper A",
            "year": 2025,
            "publication_date": "2025-01-01",
            "raw_authorships": [
                {
                    "author": {"id": "https://openalex.org/A1", "display_name": "Yoshua Bengio"},
                    "institutions": [{"id": "https://openalex.org/I1", "display_name": "Mila", "country_code": "CA"}],
                },
                {
                    "author": {"id": "https://openalex.org/A2", "display_name": "Random Researcher"},
                    "institutions": [{"id": "https://openalex.org/I2", "display_name": "Peking University", "country_code": "CN"}],
                },
            ],
        },
        {
            "openalex_work_id": "W2",
            "title": "Paper B",
            "year": 2024,
            "publication_date": "2024-05-01",
            "raw_authorships": [
                {
                    "author": {"id": "https://openalex.org/A1", "display_name": "Yoshua Bengio"},
                    "institutions": [{"id": "https://openalex.org/I1", "display_name": "Mila", "country_code": "CA"}],
                }
            ],
        },
    ]

    citation_network = aggregate_citation_network(citing_works)
    assert citation_network["citing_authors"][0]["display_name"] == "Yoshua Bengio"
    assert citation_network["citing_authors"][0]["citing_works_count"] == 2
    assert citation_network["citing_institutions"][0]["display_name"] == "Mila"
    assert citation_network["citing_institutions"][0]["citing_works_count"] == 2
    assert normalize_name("Bernhard Schölkopf") == normalize_name("Bernhard Schoelkopf")


def test_name_match_score_prefers_exact_name() -> None:
    assert _name_match_score("Bin Cui", "Bin Cui") > _name_match_score("Bin Cui", "Bin Peng")
    assert _name_match_score("Bin Cui", "Bin Cui") > _name_match_score("Bin Cui", "Zhibin Hu")


def test_orcid_bridge_requires_publication_or_affiliation_evidence() -> None:
    dblp_bundle = {
        "author_profile": {
            "display_name": "Wentao Zhang",
            "affiliations": ["Peking University, Beijing, China"],
        },
        "publications": [
            {"title": "Data-Centric Perspectives on Agentic Retrieval-Augmented Generation: A Survey", "doi": "10.36227/techrxiv.176316052.24300253/v1"},
            {"title": "SQLGovernor: An LLM-powered SQL Toolkit for Real World Application", "doi": "10.48550/arxiv.2509.08575"},
        ],
    }
    raw_author = {
        "display_name": "Wentao Zhang",
        "last_known_institutions": [{"display_name": "Peking University"}],
    }
    openalex_publications = [
        {"title": "Data-Centric Perspectives on Agentic Retrieval-Augmented Generation: A Survey", "doi": "10.36227/techrxiv.176316052.24300253/v1"},
        {"title": "SQLGovernor: An LLM-powered SQL Toolkit for Real World Application", "doi": "10.48550/arxiv.2509.08575"},
    ]

    result = evaluate_dblp_openalex_bridge(dblp_bundle, raw_author, openalex_publications)

    assert result["accepted"] is True
    assert result["doi_overlap_count"] == 2


def test_orcid_bridge_rejects_weak_same_name_match() -> None:
    dblp_bundle = {
        "author_profile": {
            "display_name": "Wentao Zhang",
            "affiliations": ["Peking University, Beijing, China"],
        },
        "publications": [
            {"title": "Data-Centric Perspectives on Agentic Retrieval-Augmented Generation: A Survey", "doi": "10.36227/techrxiv.176316052.24300253/v1"},
        ],
    }
    raw_author = {
        "display_name": "Wentao Zhang",
        "last_known_institutions": [{"display_name": "Jingdezhen Ceramic Institute"}],
    }
    openalex_publications = [
        {"title": "Transcriptomic Analysis Provides Insights into Flowering in Precocious-Fruiting Amomum villosum Lour.", "doi": "10.3390/plants15020198"},
    ]

    result = evaluate_dblp_openalex_bridge(dblp_bundle, raw_author, openalex_publications)

    assert result["accepted"] is False
    assert result["doi_overlap_count"] == 0
    assert result["title_overlap_count"] == 0


def test_publication_sample_stats_separate_loaded_and_linked_counts() -> None:
    assert build_publication_sample_stats(
        loaded_publications_count=25,
        linked_publications_count=7,
        seed_publications_count=5,
    ) == {
        "loaded_publications_count": 25,
        "linked_publications_count": 7,
        "unlinked_publications_count": 18,
        "seed_publications_count": 5,
    }


def test_normalize_base_url_adds_v1_once() -> None:
    assert _normalize_base_url("https://api.ikuncode.cc/") == "https://api.ikuncode.cc/v1"
    assert _normalize_base_url("https://api.ikuncode.cc/v1") == "https://api.ikuncode.cc/v1"


def test_match_titles_from_public_profile_text() -> None:
    titles = _match_titles(
        [
            "Bin Cui is a Boya Distinguished Professor and Vice Dean in the School of Computer Science at Peking University.",
            "He is also the director of the Institute of Data Science and Engineering.",
        ]
    )

    assert "Distinguished Professor" in titles
    assert "Vice Dean" in titles
    assert "Director" in titles


def test_normalize_llm_bundle_maps_candidate_index_back_to_citers() -> None:
    citing_authors = [
        {
            "display_name": "Alexandra Brintrup",
            "openalex_author_id": "A1",
            "affiliations": ["University of Cambridge"],
            "citing_works_count": 2,
        },
        {
            "display_name": "Yoshua Bengio",
            "openalex_author_id": "A2",
            "affiliations": ["Mila"],
            "citing_works_count": 1,
        },
    ]
    bundle = _normalize_llm_bundle(
        {
            "target_honors": ["IEEE Fellow", "Not In Allowed List"],
            "target_titles": ["Professor"],
            "matched_honorees": [
                {
                    "candidate_index": 2,
                    "canonical_name": "Yoshua Bengio",
                    "honor_label": "IEEE Fellow",
                    "evidence_summary": "Public profile lists IEEE Fellow.",
                }
            ],
            "best_effort_notice": "ok",
        },
        citing_authors,
    )

    assert bundle["target_honors"] == ["IEEE Fellow"]
    assert bundle["target_titles"] == ["Professor"]
    assert bundle["matched_honorees"] == [
        {
            "display_name": "Yoshua Bengio",
            "canonical_name": "Yoshua Bengio",
            "honor_label": "IEEE Fellow",
            "openalex_author_id": "A2",
            "affiliations": ["Mila"],
            "citing_works_count": 1,
            "evidence_summary": "Public profile lists IEEE Fellow.",
        }
    ]
