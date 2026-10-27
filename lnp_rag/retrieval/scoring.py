"""Chunk-level evidence boosts and paper-level score aggregation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from lnp_rag.config import EvidenceBoosts, RetrievalConfig
from lnp_rag.models import Chunk, ChunkType, PaperMeta


@dataclass
class ScoredChunk:
    chunk: Chunk
    relevance: float  # reranker probability (or normalised fusion score)
    fusion: float
    boost: float = 0.0
    boost_reasons: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.relevance * (1.0 + self.boost)


def evidence_boost(chunk: Chunk, intents: dict[str, bool], b: EvidenceBoosts) -> tuple[float, list[str]]:
    """Additive boost (applied multiplicatively to relevance) and the reasons for it."""
    boost, why = 0.0, []
    f = chunk.features

    def add(value: float, reason: str):
        nonlocal boost
        if value:
            boost += value
            why.append(reason)

    if chunk.chunk_type == ChunkType.FIGURE_CAPTION:
        add(b.figure_caption, "figure caption")
    elif chunk.chunk_type == ChunkType.TABLE_CAPTION:
        add(b.table_caption, "table caption")
    elif chunk.chunk_type == ChunkType.ABSTRACT:
        add(b.abstract, "abstract")
    if chunk.section_category == "results":
        add(b.results_section, "results section")
    elif chunk.section_category == "methods":
        add(b.methods_section, "methods section")
    if f.has_quantitative:
        add(b.quantitative, "quantitative values")
        if intents.get("quantitative"):
            add(b.intent_quantitative, "quantitative (requested)")
    if f.mentions_in_vitro or f.mentions_in_vivo:
        add(b.in_vitro_or_vivo_label, "experimental system stated")
    if intents.get("multi_timepoint") and f.n_timepoints >= 2:
        add(b.intent_multi_timepoint * min(1.0, f.n_timepoints / 3), f"{f.n_timepoints} time points")
    if intents.get("dose") and f.has_dose:
        add(b.intent_dose, "dose/concentration")
    if intents.get("figure") and chunk.chunk_type in (ChunkType.FIGURE_CAPTION, ChunkType.TABLE_CAPTION):
        add(b.intent_figure, "figure/table (requested)")
    return boost, why


@dataclass
class PaperScore:
    paper_id: str
    score: float
    chunks: list[ScoredChunk]  # best first
    evidence_types: set[str]


def aggregate_papers(
    scored: list[ScoredChunk], papers: dict[str, PaperMeta], cfg: RetrievalConfig
) -> list[PaperScore]:
    """Paper score = decayed sum of its best chunk scores, plus a small bonus for
    evidence-type diversity, times a review-article multiplier.

    The decayed sum rewards papers with several independent pieces of supporting
    evidence over papers with a single lucky match, while the top chunk still
    dominates (decay 0.5 -> weights 1, 0.5, 0.25, ...).
    """
    by_paper: dict[str, list[ScoredChunk]] = defaultdict(list)
    for sc in scored:
        by_paper[sc.chunk.paper_id].append(sc)
    out: list[PaperScore] = []
    norm = sum(cfg.paper_aggregation_decay**i for i in range(cfg.paper_aggregation_top_n))
    for pid, chunks in by_paper.items():
        chunks.sort(key=lambda c: c.score, reverse=True)
        top = chunks[: cfg.paper_aggregation_top_n]
        base = sum(c.score * cfg.paper_aggregation_decay**i for i, c in enumerate(top)) / norm
        # Blend with the single best chunk so one decisive caption is not diluted.
        base = 0.5 * top[0].score + 0.5 * base
        types = {_evidence_type(c.chunk) for c in top if c.relevance >= cfg.min_chunk_relevance * 2}
        bonus = min(3, max(0, len(types) - 1)) * cfg.evidence_type_coverage_bonus
        score = base * (1 + bonus)
        meta = papers.get(pid)
        if meta and meta.is_review:
            score *= cfg.boosts.review_paper_multiplier
        out.append(PaperScore(pid, score, chunks, types))
    out.sort(key=lambda p: p.score, reverse=True)
    return out


def _evidence_type(chunk: Chunk) -> str:
    if chunk.chunk_type in (ChunkType.FIGURE_CAPTION, ChunkType.TABLE_CAPTION):
        return "caption"
    return chunk.section_category
