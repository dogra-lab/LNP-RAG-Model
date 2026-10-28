"""Multi-agent folder search.

One retriever agent and one critic agent per folder (by default ``In vitro``,
``In vitro + in vivo``, ``Unclassified``, ``Cellular_Uptake_Papers``). The
folder pairs run in parallel; the orchestrator then merges the accepted papers,
de-duplicating PDFs that sit in more than one folder, and returns every
accepted paper - there is no top-k. Rejected papers are reported per folder so
the critics' decisions can be audited.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lnp_rag.agents.critic_agent import CriticAgent
from lnp_rag.agents.retriever_agent import RetrieverAgent
from lnp_rag.models import (
    AgentPaper,
    AgentSearchRequest,
    AgentSearchResponse,
    FolderReport,
    PaperMeta,
    Verdict,
)
from lnp_rag.retrieval.searcher import Searcher, _filters_active, paper_matches
from lnp_rag.synthesis.prompts import LLMClient

log = logging.getLogger(__name__)


class AgentOrchestrator:
    def __init__(self, searcher: Searcher, critic_llm: LLMClient | None):
        self.searcher = searcher
        self.cfg = searcher.cfg
        self.critic_llm = critic_llm
        self._rerank_lock = threading.Lock()
        self._retrievers: dict[str, RetrieverAgent] = {}
        self._critic_pool = ThreadPoolExecutor(max_workers=max(1, self.cfg.agents.critic_concurrency))

    def retriever(self, folder: str) -> RetrieverAgent:
        if folder not in self._retrievers:
            s = self.searcher
            self._retrievers[folder] = RetrieverAgent(
                folder, self.cfg, s.catalog, s.vectors, s.bm25, s.reranker, self._rerank_lock
            )
        return self._retrievers[folder]

    def search(self, req: AgentSearchRequest) -> AgentSearchResponse:
        t0 = time.perf_counter()
        warnings: list[str] = []
        papers = self.searcher.papers()
        folders = list(dict.fromkeys(req.folders or self.cfg.agents.folders))
        if not papers:
            return AgentSearchResponse(query=req.query, expanded_queries=[], accepted=[], folders=[],
                                       timings_ms={}, warnings=["The index is empty - index your PDFs first."])
        allowed = None
        if _filters_active(req.filters):
            allowed = {pid for pid, m in papers.items() if paper_matches(m, req.filters)}
        known = {lbl for m in papers.values() for lbl in m.folder_labels}
        for f in folders:
            if f not in known:
                warnings.append(f"No indexed papers in folder '{f}'.")
        if req.use_critic and self.critic_llm is None:
            warnings.append("No LLM configured (OPENAI_API_KEY or ANTHROPIC_API_KEY): critics use "
                            "heuristic verdicts.")

        # Shared query work, done once for all agents.
        expanded = self.searcher.expander.expand(req.query)
        vecs = np.vstack([self.searcher.embedder.embed_query(q) for q in expanded.all_dense]).astype(np.float32)
        vecs /= np.where((n := np.linalg.norm(vecs, axis=1, keepdims=True)) == 0, 1, n)
        bm25_queries = list(dict.fromkeys([req.query, expanded.bm25_query]))
        prep_ms = _ms(t0)

        critic_llm = self.critic_llm if req.use_critic else None
        with ThreadPoolExecutor(max_workers=max(1, len(folders))) as pool:
            jobs = [pool.submit(self._run_folder, f, req.query, bm25_queries, vecs, papers, allowed, critic_llm)
                    for f in folders]
            reports = [j.result() for j in jobs]
        for r in reports:
            warnings.extend(r.warnings)

        accepted = merge_accepted(reports)
        timings = {"prepare": prep_ms, "total": _ms(t0)}
        log.info("agent_search", extra={"query": req.query[:120], "accepted": len(accepted), **timings,
                                        **{f"{r.folder}_accepted": len(r.accepted) for r in reports}})
        return AgentSearchResponse(
            query=req.query,
            expanded_queries=expanded.variants + ([f"BM25 +{len(expanded.keywords)} terms"] if expanded.keywords else []),
            accepted=accepted,
            folders=reports,
            timings_ms=timings,
            warnings=warnings,
        )

    def _run_folder(
        self, folder: str, query: str, bm25_queries: list[str], vecs: np.ndarray,
        papers: dict[str, PaperMeta], allowed: set[str] | None, critic_llm: LLMClient | None,
    ) -> FolderReport:
        t0 = time.perf_counter()
        found = self.retriever(folder).run(query, bm25_queries, vecs, papers, allowed)
        t1 = time.perf_counter()
        critic = CriticAgent(folder, self.cfg.agents, critic_llm, self._critic_pool, self.searcher.pdf_url)
        reviewed, warnings = critic.review(query, found.candidates, papers)
        for p in reviewed:
            if p.verdict == Verdict.ACCEPT:
                p.found_by = [folder]
        return FolderReport(
            folder=folder,
            papers_in_folder=found.papers_in_folder,
            chunks_scanned=found.chunks_scanned,
            candidates=sum(c.route == "candidate" for c in found.candidates),
            near_misses=sum(c.route == "near_miss" for c in found.candidates),
            dropped=found.dropped,
            accepted=[p for p in reviewed if p.verdict == Verdict.ACCEPT],
            rejected=[p for p in reviewed if p.verdict == Verdict.REJECT],
            timings_ms={"retriever": round((t1 - t0) * 1000, 1), "critic": _ms(t1)},
            warnings=warnings,
        )


def merge_accepted(reports: list[FolderReport]) -> list[AgentPaper]:
    """One entry per paper (a PDF in two folders is found by two agents); keeps the
    higher-scoring copy and records every folder whose critic accepted it."""
    merged: dict[str, AgentPaper] = {}
    for r in reports:
        for p in r.accepted:
            if (prev := merged.get(p.paper_id)) is None:
                merged[p.paper_id] = p.model_copy(deep=True)
                continue
            found_by = list(dict.fromkeys(prev.found_by + p.found_by))
            if p.retriever_score > prev.retriever_score:
                merged[p.paper_id] = p.model_copy(deep=True)
            merged[p.paper_id].found_by = found_by
    return sorted(merged.values(), key=lambda p: p.retriever_score, reverse=True)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
