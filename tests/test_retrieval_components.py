from pathlib import Path

from lnp_rag.config import PROJECT_ROOT, RetrievalConfig
from lnp_rag.models import Chunk, ChunkFeatures, ChunkType, PaperMeta
from lnp_rag.retrieval.fusion import reciprocal_rank_fusion
from lnp_rag.retrieval.query_expansion import SynonymTable, expand_with_rules
from lnp_rag.retrieval.scoring import ScoredChunk, aggregate_papers, evidence_boost
from lnp_rag.store.bm25_index import tokenize

SYNONYMS = SynonymTable.load(PROJECT_ROOT / "config" / "synonyms.yaml")


def _chunk(pid: str, i: int, ctype=ChunkType.PARAGRAPH, cat="results", **feat) -> Chunk:
    return Chunk(
        chunk_id=f"{pid}:{i:04d}", paper_id=pid, text="x", chunk_type=ctype, section="S",
        section_category=cat, page_start=1, page_end=1, features=ChunkFeatures(**feat),
    )


def test_rrf_rewards_agreement_between_lists():
    fused = reciprocal_rank_fusion([(1.0, ["a", "b", "c"]), (1.0, ["b", "c", "a"])], k=60)
    assert fused[0][0] == "b"
    assert abs(fused[0][1] - (1 / 62 + 1 / 61)) < 1e-9
    # weights matter
    fused = reciprocal_rank_fusion([(1.0, ["a"]), (0.1, ["b"])])
    assert [d for d, _ in fused] == ["a", "b"]


def test_tokenizer_keeps_scientific_compounds():
    toks = tokenize("DLin-MC3-DMA and SM-102 LNPs at 24 h with 0.5 μg/mL")
    assert "dlin-mc3-dma" in toks and "mc3" in toks and "sm-102" in toks
    assert "0.5" in toks and "24" in toks
    assert "and" not in toks
    assert tokenize("nanoparticles") == tokenize("nanoparticle")  # stemmed


def test_rule_expansion_preserves_intent():
    e = expand_with_rules("cellular internalization of lipid-based nanocarriers", SYNONYMS)
    assert e.original == "cellular internalization of lipid-based nanocarriers"
    joined = " ".join(e.variants + e.keywords).lower()
    assert "cellular uptake" in joined and "lnp" in joined
    assert "liposome" not in joined  # not a synonym: would change intent
    # related evidence terms (BM25 only) are added via the synonym of the anchor
    assert "flow cytometry" in [k.lower() for k in e.keywords]
    assert e.variants and all(v != e.original for v in e.variants)
    assert len(e.all_dense) == len(e.variants) + 1 and e.all_dense[0] == e.original


def test_expansion_noop_for_unrelated_query():
    e = expand_with_rules("zeta potential of gold particles", SYNONYMS)
    assert e.variants == [] and e.keywords == []


def test_evidence_boosts_are_intent_gated():
    cfg = RetrievalConfig()
    cap = _chunk("p", 0, ChunkType.FIGURE_CAPTION, "figure", has_quantitative=True, n_timepoints=4)
    no_intent, _ = evidence_boost(cap, {}, cfg.boosts)
    with_intent, why = evidence_boost(cap, {"multi_timepoint": True, "quantitative": True}, cfg.boosts)
    assert with_intent > no_intent > 0
    assert "4 time points" in why and "figure caption" in why


def test_paper_aggregation_prefers_multiple_evidence_and_demotes_reviews():
    cfg = RetrievalConfig()
    one = [ScoredChunk(_chunk("A", 0), 0.80, 0)]
    many = [ScoredChunk(_chunk("B", i), s, 0) for i, s in enumerate([0.78, 0.75, 0.7])]
    papers = {
        "A": PaperMeta(paper_id="A", title="a", filename="a.pdf"),
        "B": PaperMeta(paper_id="B", title="b", filename="b.pdf"),
    }
    ranked = aggregate_papers(one + many, papers, cfg)
    assert ranked[0].paper_id == "B"
    papers["B"].is_review = True
    ranked = aggregate_papers(one + many, papers, cfg)
    assert ranked[0].paper_id == "A"
