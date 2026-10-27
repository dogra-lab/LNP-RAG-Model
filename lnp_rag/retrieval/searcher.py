"""Search orchestration.

query -> expansion -> (dense + BM25) per query variant -> RRF fusion ->
top-N candidates -> cross-encoder rerank -> evidence boosts ->
paper-level aggregation -> LLM synthesis (grounded) -> ranked papers.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

from lnp_rag.config import RetrievalConfig, Settings
from lnp_rag.embeddings import Embedder
from lnp_rag.ingest.chunker import embedding_text
from lnp_rag.ingest.features import query_intents
from lnp_rag.models import (
    ChunkType,
    Evidence,
    PaperMeta,
    PaperResult,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    StudyType,
)
from lnp_rag.retrieval.fusion import reciprocal_rank_fusion
from lnp_rag.retrieval.query_expansion import QueryExpander
from lnp_rag.retrieval.reranker import CrossEncoderReranker
from lnp_rag.retrieval.scoring import ScoredChunk, aggregate_papers, evidence_boost
from lnp_rag.store.bm25_index import BM25Index
from lnp_rag.store.catalog import Catalog
from lnp_rag.store.vector_store import VectorStore
from lnp_rag.synthesis.prompts import LLMClient, SynthesisError
from lnp_rag.synthesis.grounding import (
    extractive_explanation,
    remap_citations,
    study_type_from_evidence,
    validate_synthesis,
)

log = logging.getLogger(__name__)


def paper_matches(meta: PaperMeta, f: SearchFilters) -> bool:
    if f.in_vitro and meta.study_type not in (StudyType.IN_VITRO, StudyType.BOTH):
        return False
    if f.in_vivo and meta.study_type not in (StudyType.IN_VIVO, StudyType.BOTH):
        return False
    if f.year_min and (meta.year is None or meta.year < f.year_min):
        return False
    if f.year_max and (meta.year is None or meta.year > f.year_max):
        return False
    if f.title_contains and f.title_contains.lower() not in meta.title.lower():
        return False
    if f.has_figures and meta.n_figures + meta.n_tables == 0:
        return False
    if f.has_quantitative and not meta.has_quantitative:
        return False
    if f.exclude_reviews and meta.is_review:
        return False
    if f.folder and f.folder not in meta.folder_labels:
        return False
    return True


def _filters_active(f: SearchFilters) -> bool:
    return any(
        [f.in_vitro, f.in_vivo, f.year_min, f.year_max, f.title_contains, f.has_figures,
         f.has_quantitative, f.exclude_reviews, f.folder]
    )


class Searcher:
    def __init__(
        self,
        settings: Settings,
        cfg: RetrievalConfig,
        catalog: Catalog,
        vectors: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        reranker: CrossEncoderReranker | None,
        expander: QueryExpander,
        llm: LLMClient | None,
    ):
        self.settings = settings
        self.cfg = cfg
        self.catalog = catalog
        self.vectors = vectors
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.expander = expander
        self.llm = llm
        self._pool = ThreadPoolExecutor(max_workers=8)
        self._papers_cache: tuple[int, dict[str, PaperMeta]] | None = None

    def papers(self) -> dict[str, PaperMeta]:
        # Cache paper metadata; invalidated when the chunk count changes (re-index).
        n = self.catalog.count_chunks()
        if self._papers_cache is None or self._papers_cache[0] != n:
            self._papers_cache = (n, self.catalog.all_papers())
        return self._papers_cache[1]

    def pdf_url(self, paper_id: str, page: int | None = None) -> str:
        base = f"{self.settings.api_public_url.rstrip('/')}/papers/{quote(paper_id)}/pdf"
        return f"{base}#page={page}" if page else base

    # ------------------------------------------------------------------------
    def search(self, req: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []
        cfg = self.cfg

        papers = self.papers()
        if not papers:
            return SearchResponse(query=req.query, expanded_queries=[], total_candidates=0, results=[],
                                  timings_ms={}, warnings=["The index is empty - index your PDFs first."])
        allowed: set[str] | None = None
        if _filters_active(req.filters):
            allowed = {pid for pid, m in papers.items() if paper_matches(m, req.filters)}
            if not allowed:
                return SearchResponse(query=req.query, expanded_queries=[], total_candidates=0, results=[],
                                      timings_ms={}, warnings=["No papers match the selected filters."])

        # 1. query expansion
        expanded = self.expander.expand(req.query)
        timings["expansion"] = _ms(t0)

        # 2. dense + BM25 retrieval for every query variant, in parallel
        t1 = time.perf_counter()
        allowed_list = sorted(allowed) if allowed is not None else None
        dense_jobs = []
        for i, q in enumerate(expanded.all_dense):
            weight = cfg.original_query_weight if i == 0 else cfg.expanded_query_weight
            vec = self.embedder.embed_query(q)
            dense_jobs.append((weight, self._pool.submit(self.vectors.query, vec, cfg.dense_top_k, allowed_list)))
        bm25_jobs = [
            (cfg.original_query_weight, self._pool.submit(self.bm25.search, req.query, cfg.bm25_top_k, allowed)),
        ]
        if expanded.keywords:
            bm25_jobs.append(
                (cfg.expanded_query_weight,
                 self._pool.submit(self.bm25.search, expanded.bm25_query, cfg.bm25_top_k, allowed))
            )
        ranked_lists = []
        for weight, fut in dense_jobs:
            ranked_lists.append((weight * cfg.dense_weight, [cid for cid, _ in fut.result()]))
        for weight, fut in bm25_jobs:
            ranked_lists.append((weight * cfg.bm25_weight, [cid for cid, _ in fut.result()]))
        fused = reciprocal_rank_fusion(ranked_lists, k=cfg.rrf_k)
        timings["retrieval"] = _ms(t1)
        candidates = fused[: cfg.rerank_candidates]
        if not candidates:
            return SearchResponse(query=req.query, expanded_queries=expanded.variants, total_candidates=0,
                                  results=[], timings_ms=timings, warnings=["No matching text found."])

        # 3. rerank
        t2 = time.perf_counter()
        chunks = self.catalog.get_chunks([cid for cid, _ in candidates])
        candidates = [(cid, s) for cid, s in candidates if cid in chunks]
        max_fusion = candidates[0][1]
        if self.reranker is not None:
            passages = [embedding_text(chunks[cid], papers[chunks[cid].paper_id].title) for cid, _ in candidates]
            relevance = self.reranker.score(req.query, passages)
        else:
            relevance = [s / max_fusion for _, s in candidates]
            warnings.append("Reranker disabled: scores are fusion-based.")
        timings["rerank"] = _ms(t2)

        # 4. boosts + paper aggregation
        intents = query_intents(req.query)
        scored: list[ScoredChunk] = []
        for (cid, fusion), rel in zip(candidates, relevance):
            if rel < cfg.min_chunk_relevance:
                continue
            boost, why = evidence_boost(chunks[cid], intents, cfg.boosts)
            scored.append(ScoredChunk(chunks[cid], float(rel), fusion, boost, why))
        ranked = aggregate_papers(scored, papers, cfg)
        n_candidates = len(ranked)
        ranked = ranked[: max(req.top_k, cfg.synthesis_top_papers if req.synthesize else req.top_k)]

        # 5. build results with evidence
        results: list[PaperResult] = []
        for i, ps in enumerate(ranked, start=1):
            meta = papers[ps.paper_id]
            key = f"P{i}"
            evidence = self._evidence(key, ps.chunks[: cfg.max_evidence_per_paper])
            caption_chunks = [c for c in ps.chunks if c.chunk.chunk_type in
                              (ChunkType.FIGURE_CAPTION, ChunkType.TABLE_CAPTION)][:3]
            by_chunk = {e.chunk_id: e for e in evidence}
            new_caps = [c for c in caption_chunks if c.chunk.chunk_id not in by_chunk]
            extra = self._evidence(key, new_caps, start=len(evidence) + 1)
            extra_by_chunk = {e.chunk_id: e for e in extra}
            captions = [by_chunk.get(c.chunk.chunk_id) or extra_by_chunk[c.chunk.chunk_id] for c in caption_chunks]
            text, strength, missing = extractive_explanation(evidence, intents, ps.chunks[0].relevance)
            pages = sorted({e.page for e in evidence + captions})
            sections = list(dict.fromkeys(e.section for e in evidence))
            results.append(
                PaperResult(
                    rank=i,
                    paper_id=ps.paper_id,
                    title=meta.title,
                    authors=meta.authors,
                    filename=meta.filename,
                    doi=meta.doi,
                    year=meta.year,
                    relevance_score=round(ps.score, 4),
                    retrieval_score=round(ps.score, 4),
                    pages=pages,
                    sections=sections,
                    study_type=study_type_from_evidence(evidence, meta.study_type),
                    study_type_basis="detected from methods/results/caption text"
                    if meta.study_type != StudyType.UNKNOWN else "detected from retrieved excerpts",
                    is_review=meta.is_review,
                    explanation=text,
                    explanation_source="extractive",
                    evidence_strength=strength,
                    missing_evidence=missing,
                    cited_evidence_ids=[e.evidence_id for e in evidence[:3]],
                    evidence=evidence,
                    captions=captions,
                    pdf_url=self.pdf_url(ps.paper_id, pages[0] if pages else None),
                )
            )

        # 6. LLM synthesis
        if req.synthesize and self.llm is not None and results:
            t3 = time.perf_counter()
            try:
                self._synthesize(req.query, results[: cfg.synthesis_top_papers])
            except SynthesisError as exc:
                warnings.append(f"LLM synthesis unavailable ({exc}); showing extractive explanations.")
            timings["synthesis"] = _ms(t3)
        elif req.synthesize and self.llm is None:
            warnings.append("LLM synthesis disabled (no OPENAI_API_KEY or ANTHROPIC_API_KEY); "
                            "showing extractive explanations.")

        results = self._rerank_final(results)[: req.top_k]
        if results and results[0].evidence_strength == "insufficient":
            warnings.append("No paper has strong supporting evidence for this query; results are the closest matches.")
        timings["total"] = _ms(t0)
        log.info("search", extra={"query": req.query[:120], "results": len(results), **timings})
        return SearchResponse(
            query=req.query,
            expanded_queries=expanded.variants + ([f"BM25 +{len(expanded.keywords)} terms"] if expanded.keywords else []),
            total_candidates=n_candidates,
            results=results,
            timings_ms=timings,
            warnings=warnings,
        )

    def _evidence(self, key: str, chunks: list[ScoredChunk], start: int = 1) -> list[Evidence]:
        return [
            Evidence(
                evidence_id=f"{key}-E{j}",
                chunk_id=sc.chunk.chunk_id,
                text=sc.chunk.text,
                chunk_type=sc.chunk.chunk_type,
                section=sc.chunk.label or sc.chunk.section,
                page=sc.chunk.page_start,
                label=sc.chunk.label,
                score=round(sc.score, 4),
                rerank_score=round(sc.relevance, 4),
                features=sc.chunk.features,
                pdf_url=self.pdf_url(sc.chunk.paper_id, sc.chunk.page_start),
            )
            for j, sc in enumerate(chunks, start=start)
        ]

    def _synthesize(self, query: str, results: list[PaperResult]) -> None:
        payload = []
        for r in results:
            key = r.evidence[0].evidence_id.split("-")[0] if r.evidence else f"P{r.rank}"
            items = {e.evidence_id: e for e in r.evidence + r.captions}
            payload.append({
                "key": key,
                "title": r.title,
                "is_review": r.is_review,
                "evidence": [
                    {"id": e.evidence_id, "type": e.chunk_type.value, "section": e.section, "page": e.page,
                     "text": e.text}
                    for e in items.values()
                ],
            })
        out = self.llm.synthesize(query, payload)
        w = self.cfg.synthesis_score_weight
        top_retrieval = max(r.retrieval_score for r in results) or 1.0
        for r, p in zip(results, payload):
            item = out.get(p["key"])
            if not item:
                continue
            checked = validate_synthesis(item, r.evidence + r.captions)
            r.explanation = checked["explanation"]
            r.explanation_source = self.llm.name
            r.evidence_strength = checked["evidence_strength"]
            r.missing_evidence = checked.get("missing_evidence", "")
            r.cited_evidence_ids = checked["cited_evidence_ids"]
            if checked["grounded"] and checked.get("study_type") in {s.value for s in StudyType} \
                    and checked["study_type"] != "unknown":
                r.study_type = StudyType(checked["study_type"])
                r.study_type_basis = f"stated in cited excerpts ({self.llm.name})"
            r.relevance_score = round(
                (1 - w) * (r.retrieval_score / top_retrieval) + w * checked["relevance"] / 100, 4
            )

    def _rerank_final(self, results: list[PaperResult]) -> list[PaperResult]:
        synthesized = [r for r in results if r.explanation_source != "extractive"]
        if synthesized:
            # Papers the LLM assessed are ranked by the blended score; the rest keep order after them.
            top = max(r.retrieval_score for r in results) or 1.0
            for r in results:
                if r.explanation_source == "extractive":
                    r.relevance_score = round((1 - self.cfg.synthesis_score_weight) * r.retrieval_score / top, 4)
            results = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        mapping = {}
        for rank, r in enumerate(results, start=1):
            old = r.evidence[0].evidence_id.split("-")[0] if r.evidence else f"P{r.rank}"
            mapping[old] = f"P{rank}"
            r.rank = rank
        for r in results:
            for e in {id(e): e for e in r.evidence + r.captions}.values():
                e.evidence_id = remap_citations(e.evidence_id, mapping)
            r.explanation = remap_citations(r.explanation, mapping)
            r.cited_evidence_ids = [remap_citations(c, mapping) for c in r.cited_evidence_ids]
        return results


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
