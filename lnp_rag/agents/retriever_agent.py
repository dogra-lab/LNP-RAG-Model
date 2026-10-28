"""Folder retriever agent: finds every paper in one folder that may answer the query.

Recall first, with no rank cutoffs. Every chunk in the folder is scored
exhaustively - dense cosine against all stored chunk vectors and BM25 against all
chunk texts - and a paper becomes a candidate if ANY chunk (paragraph, figure
caption or table caption) clears an absolute threshold. The candidates' best
chunks are rescored by the cross-encoder, then papers are routed by score:

    >= candidate_min_score  -> critic
    >= near_miss_min_score  -> critic re-check (the retriever may have under-scored it)
    below                   -> dropped, but counted in the folder report
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from lnp_rag.config import AgentConfig, RetrievalConfig
from lnp_rag.ingest.chunker import embedding_text
from lnp_rag.ingest.features import query_intents
from lnp_rag.models import PaperMeta
from lnp_rag.retrieval.reranker import CrossEncoderReranker
from lnp_rag.retrieval.scoring import PaperScore, ScoredChunk, aggregate_papers, evidence_boost
from lnp_rag.store.bm25_index import BM25Index
from lnp_rag.store.catalog import Catalog
from lnp_rag.store.vector_store import VectorStore


@dataclass
class FolderCorpus:
    """Every indexed chunk of one folder, with L2-normalised vectors aligned to it."""

    folder: str
    paper_ids: set[str]
    chunk_ids: list[str]
    chunk_papers: np.ndarray  # paper_id per chunk
    bm25_rows: np.ndarray  # row of each chunk in the BM25 index
    is_caption: np.ndarray  # figure/table caption chunk
    vectors: np.ndarray


@dataclass
class Candidate:
    paper: PaperScore
    route: str  # "candidate" | "near_miss"


@dataclass
class RetrieverResult:
    folder: str
    papers_in_folder: int
    chunks_scanned: int
    candidates: list[Candidate] = field(default_factory=list)  # both routes, best first
    dropped: int = 0


class RetrieverAgent:
    def __init__(
        self,
        folder: str,
        cfg: RetrievalConfig,
        catalog: Catalog,
        vectors: VectorStore,
        bm25: BM25Index,
        reranker: CrossEncoderReranker | None,
        rerank_lock: threading.Lock,
    ):
        self.folder = folder
        self.cfg = cfg
        self.acfg: AgentConfig = cfg.agents
        self.catalog = catalog
        self.vectors = vectors
        self.bm25 = bm25
        self.reranker = reranker
        self._rerank_lock = rerank_lock
        self._corpus: tuple[int, FolderCorpus] | None = None

    def corpus(self, papers: dict[str, PaperMeta]) -> FolderCorpus:
        # Cached; invalidated when the chunk count changes (re-index).
        n = self.catalog.count_chunks()
        if self._corpus is None or self._corpus[0] != n:
            pids = {pid for pid, m in papers.items() if self.folder in m.folder_labels}
            chunk_ids, chunk_papers, rows = self.bm25.rows_for_papers(pids)
            captions = self.catalog.caption_chunk_ids()
            is_caption = np.array([cid in captions for cid in chunk_ids], dtype=bool)
            vecs = self.vectors.get_vectors(chunk_ids)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) if vecs.size else np.ones((len(chunk_ids), 1))
            vecs = vecs / np.where(norms == 0, 1, norms)
            self._corpus = (n, FolderCorpus(self.folder, pids, chunk_ids, chunk_papers, rows, is_caption, vecs))
        return self._corpus[1]

    def run(
        self,
        query: str,
        bm25_queries: list[str],
        query_vectors: np.ndarray,
        papers: dict[str, PaperMeta],
        allowed: set[str] | None,
    ) -> RetrieverResult:
        """``query_vectors``: one L2-normalised row per query variant (original first)."""
        a = self.acfg
        corpus = self.corpus(papers)
        in_scope = corpus.paper_ids if allowed is None else corpus.paper_ids & allowed
        result = RetrieverResult(self.folder, len(in_scope), 0)
        if not corpus.chunk_ids or not in_scope:
            return result
        mask = np.isin(corpus.chunk_papers, list(in_scope))
        result.chunks_scanned = int(mask.sum())

        # 1. exhaustive scoring of every chunk in the folder
        if corpus.vectors.shape[1] == query_vectors.shape[1]:
            dense = (corpus.vectors @ query_vectors.T).max(axis=1)
        else:  # vectors missing from the store
            dense = np.zeros(len(corpus.chunk_ids), dtype=np.float32)
        lexical = np.zeros(len(corpus.chunk_ids), dtype=np.float32)
        for q in bm25_queries:
            s = self.bm25.score_rows(q, corpus.bm25_rows) * mask
            if s.max() > 0:
                lexical = np.maximum(lexical, s / s.max())
        hit = mask & ((dense >= a.dense_min_similarity) | (lexical >= a.bm25_min_ratio))
        prelim = dense + lexical

        # 2. per candidate paper, pick its best chunks (and best captions) for reranking
        by_paper: dict[str, list[int]] = defaultdict(list)
        for i in np.flatnonzero(hit):
            by_paper[corpus.chunk_papers[i]].append(int(i))
        picked: list[int] = []
        for idx in by_paper.values():
            idx.sort(key=lambda i: prelim[i], reverse=True)
            caps = [i for i in idx if corpus.is_caption[i]][: a.caption_chunks_per_paper]
            picked.extend(dict.fromkeys(idx[: a.rerank_chunks_per_paper] + caps))
        if not picked:
            return result
        chunks = self.catalog.get_chunks([corpus.chunk_ids[i] for i in picked])
        picked = [i for i in picked if corpus.chunk_ids[i] in chunks]

        # 3. rerank, boost, aggregate to papers
        if self.reranker is not None:
            passages = [embedding_text(chunks[corpus.chunk_ids[i]], papers[corpus.chunk_papers[i]].title)
                        for i in picked]
            with self._rerank_lock:  # one cross-encoder shared by all folder agents
                relevance = self.reranker.score(query, passages)
        else:
            relevance = [max(float(dense[i]), float(lexical[i])) for i in picked]
        intents = query_intents(query)
        scored = []
        for i, rel in zip(picked, relevance):
            chunk = chunks[corpus.chunk_ids[i]]
            boost, why = evidence_boost(chunk, intents, self.cfg.boosts)
            scored.append(ScoredChunk(chunk, float(rel), float(prelim[i]), boost, why))

        # 4. route by absolute score - no rank cutoff
        for ps in aggregate_papers(scored, papers, self.cfg):
            if ps.score >= a.candidate_min_score:
                result.candidates.append(Candidate(ps, "candidate"))
            elif ps.score >= a.near_miss_min_score:
                result.candidates.append(Candidate(ps, "near_miss"))
            else:
                result.dropped += 1
        return result

