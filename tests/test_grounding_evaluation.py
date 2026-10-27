import math

from lnp_rag.evaluation import (
    evaluate_ranking,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from lnp_rag.models import ChunkFeatures, ChunkType, Evidence
from lnp_rag.synthesis.grounding import extractive_explanation, remap_citations, validate_synthesis


def _ev(eid: str, **feat) -> Evidence:
    return Evidence(
        evidence_id=eid, chunk_id="c", text="t", chunk_type=ChunkType.PARAGRAPH, section="Results",
        page=3, score=0.5, rerank_score=0.5, features=ChunkFeatures(**feat), pdf_url="u",
    )


def test_validate_synthesis_drops_hallucinated_citations():
    evidence = [_ev("P1-E1"), _ev("P1-E2")]
    item = {
        "explanation": "MFI rose over time [P1-E1] and doubled at 24 h [P1-E7].",
        "cited_evidence_ids": ["P1-E1", "P1-E9"],
        "evidence_strength": "strong",
        "relevance": 95,
    }
    out = validate_synthesis(item, evidence)
    assert out["cited_evidence_ids"] == ["P1-E1"]
    assert "[P1-E7]" not in out["explanation"] and "[P1-E1]" in out["explanation"]
    assert out["grounded"] and out["evidence_strength"] == "strong"


def test_validate_synthesis_downgrades_uncited_claims():
    out = validate_synthesis(
        {"explanation": "Contains uptake data.", "cited_evidence_ids": [], "evidence_strength": "strong",
         "relevance": 90},
        [_ev("P1-E1")],
    )
    assert not out["grounded"]
    assert out["evidence_strength"] == "insufficient"
    assert out["relevance"] <= 20


def test_remap_citations():
    assert remap_citations("see [P3-E1, P1-E2]", {"P3": "P1", "P1": "P2"}) == "see [P1-E1, P2-E2]"


def test_extractive_explanation_reports_missing_elements():
    text, strength, missing = extractive_explanation(
        [_ev("P1-E1", has_uptake_terms=True)], {"multi_timepoint": True, "quantitative": True}, 0.6
    )
    assert "[P1-E1]" in text
    assert "multiple time points" in missing and "quantitative values" in missing
    assert strength == "partial"


def test_metrics_known_values():
    ranked = ["a", "x", "b", "y", "c"]
    rel = {"a", "b", "c", "d"}
    assert precision_at_k(ranked, rel, 5) == 3 / 5
    assert recall_at_k(ranked, rel, 5) == 3 / 4
    assert hit_at_k(["x", "y"], rel, 2) == 0.0
    assert reciprocal_rank(["x", "b"], rel) == 0.5
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2) == 1.0
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2) < 1.0
    m = evaluate_ranking(ranked, {p: 1 for p in rel}, [5], {"a": {3}}, {"a": {3, 4}, "zzz": {1}})
    assert m["page_hit_rate"] == 0.5
    assert math.isclose(m["mrr"], 1.0)
