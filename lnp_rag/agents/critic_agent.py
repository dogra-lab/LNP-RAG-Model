"""Folder critic agent: checks each paper the retriever proposed against the query.

Candidates and near-misses are sent to the LLM in batches. For every paper it
returns ACCEPT/REJECT, a reason (for ACCEPT: why the paper is a correct answer),
the excerpt IDs it relied on, and verbatim quotes. The verdict is then checked
against the excerpts the critic was actually shown: unknown IDs and quotes that do
not occur in the cited excerpt are discarded, and an ACCEPT left with no
verifiable evidence becomes a REJECT.

Without an LLM (or if a batch call fails) the critic falls back to a heuristic:
retriever candidates are accepted, near-misses rejected, and the reason says so.
"""
from __future__ import annotations

import logging
import re
import textwrap
from collections.abc import Callable
from concurrent.futures import Executor

from lnp_rag.agents.retriever_agent import Candidate
from lnp_rag.config import AgentConfig
from lnp_rag.models import AgentEvidence, AgentPaper, ChunkType, CitedQuote, PaperMeta, Verdict
from lnp_rag.synthesis.prompts import LLMClient, SynthesisError

log = logging.getLogger(__name__)

_CAPTIONS = (ChunkType.FIGURE_CAPTION, ChunkType.TABLE_CAPTION)
_CITATION = re.compile(r"\bP\d+-(E\d+)\b")


class CriticAgent:
    def __init__(
        self,
        folder: str,
        acfg: AgentConfig,
        llm: LLMClient | None,
        pool: Executor,
        pdf_url: Callable[[str, int | None], str],
    ):
        self.folder = folder
        self.acfg = acfg
        self.llm = llm
        self.pool = pool
        self.pdf_url = pdf_url

    def review(
        self, query: str, candidates: list[Candidate], papers: dict[str, PaperMeta]
    ) -> tuple[list[AgentPaper], list[str]]:
        """Every candidate gets a verdict. Returns (papers, warnings)."""
        items = [self._paper(c, papers[c.paper.paper_id]) for c in candidates]
        if not items:
            return [], []
        if self.llm is None:
            return [self._heuristic(p) for p in items], []

        size = max(1, self.acfg.critic_batch_size)
        batches = [items[i : i + size] for i in range(0, len(items), size)]
        futures = [self.pool.submit(self._review_batch, query, b, papers) for b in batches]
        out: list[AgentPaper] = []
        warnings: list[str] = []
        for batch, fut in zip(batches, futures):
            try:
                out.extend(fut.result())
            except SynthesisError as exc:
                warnings.append(f"{self.folder}: critic unavailable for {len(batch)} papers ({exc}); "
                                "heuristic verdicts used.")
                out.extend(self._heuristic(p) for p in batch)
        return out, warnings

    # ------------------------------------------------------------------------
    def _paper(self, c: Candidate, meta: PaperMeta) -> AgentPaper:
        """Paper with its evidence: the best chunks, plus every reranked caption."""
        n = self.acfg.evidence_per_paper
        chosen = c.paper.chunks[:n] + [sc for sc in c.paper.chunks[n:] if sc.chunk.chunk_type in _CAPTIONS]
        evidence = [
            AgentEvidence(
                evidence_id=f"E{j}",
                chunk_id=sc.chunk.chunk_id,
                chunk_type=sc.chunk.chunk_type,
                section=sc.chunk.label or sc.chunk.section,
                label=sc.chunk.label,
                page=sc.chunk.page_start,
                text=sc.chunk.text,
                rerank_score=round(sc.relevance, 4),
                pdf_url=self.pdf_url(sc.chunk.paper_id, sc.chunk.page_start),
            )
            for j, sc in enumerate(chosen, start=1)
        ]
        return AgentPaper(
            paper_id=meta.paper_id,
            title=meta.title,
            filename=meta.filename,
            folder_labels=meta.folder_labels,
            year=meta.year,
            doi=meta.doi,
            retriever_score=round(c.paper.score, 4),
            route=c.route,
            caption_match=any(e.chunk_type in _CAPTIONS for e in evidence),
            verdict=Verdict.REJECT,
            reason="",
            critic="pending",
            evidence=evidence,
            pdf_url=self.pdf_url(meta.paper_id, evidence[0].page if evidence else None),
        )

    def _review_batch(self, query: str, batch: list[AgentPaper], papers: dict[str, PaperMeta]) -> list[AgentPaper]:
        payload = [
            {
                "key": f"P{k}",
                "title": p.title,
                "is_review": papers[p.paper_id].is_review,
                "near_miss": p.route == "near_miss",
                "evidence": [
                    {"id": f"P{k}-{e.evidence_id}", "type": e.chunk_type.value, "section": e.section,
                     "page": e.page, "text": e.text}
                    for e in p.evidence
                ],
            }
            for k, p in enumerate(batch, start=1)
        ]
        verdicts = self.llm.review(query, self.folder, payload)
        out = []
        for k, p in enumerate(batch, start=1):
            item = verdicts.get(f"P{k}")
            if item is None:
                p = self._heuristic(p)
                p.reason = "Critic returned no verdict for this paper. " + p.reason
                out.append(p)
            else:
                out.append(apply_verdict(p, item, f"P{k}", self.llm.name))
        return out

    def _heuristic(self, p: AgentPaper) -> AgentPaper:
        best = p.evidence[0] if p.evidence else None
        where = f"{best.label or best.section}, p.{best.page}" if best else "n/a"
        snippet = textwrap.shorten(best.text, 220) if best else ""
        if p.route == "candidate":
            p.verdict = Verdict.ACCEPT
            p.reason = (f"No LLM critic: accepted on retriever score {p.retriever_score:.2f}. "
                        f"Best match [{best.evidence_id if best else ''}] {where}: \"{snippet}\"")
            p.cited_evidence_ids = [best.evidence_id] if best else []
        else:
            p.verdict = Verdict.REJECT
            p.reason = f"No LLM critic: near-miss (retriever score {p.retriever_score:.2f}) not re-checked."
        p.critic = "heuristic"
        return p


def apply_verdict(p: AgentPaper, item: dict, key: str, critic: str) -> AgentPaper:
    """Apply an LLM verdict to ``p``, keeping only evidence the critic was really shown."""
    by_id = {e.evidence_id: e for e in p.evidence}

    def local(eid: str) -> str:
        return eid.removeprefix(f"{key}-")

    cited = [local(e) for e in item.get("cited_evidence_ids", []) if local(e) in by_id]
    quotes = []
    for q in item.get("quotes", []):
        ev = by_id.get(local(q.get("evidence_id", "")))
        text = q.get("quote", "").strip()
        if ev and text and _norm(text) in _norm(ev.text):
            quotes.append(CitedQuote(evidence_id=ev.evidence_id, quote=text, page=ev.page, label=ev.label))
    cited = list(dict.fromkeys(cited + [q.evidence_id for q in quotes]))
    reason = _CITATION.sub(r"\1", item.get("reason", "").strip())

    p.critic = critic
    p.cited_evidence_ids = cited
    p.quotes = quotes
    if item.get("verdict") == "ACCEPT" and cited:
        p.verdict, p.reason = Verdict.ACCEPT, reason
    elif item.get("verdict") == "ACCEPT":
        p.verdict = Verdict.REJECT
        p.reason = f"Critic accepted but cited no evidence that exists in this paper's excerpts. ({reason})"
    else:
        p.verdict, p.reason = Verdict.REJECT, reason
    return p


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("­", "")).strip().lower()
