"""Retrieval metrics and labeled-dataset handling.

Dataset format (JSONL, one query per line)::

    {"query_id": "uptake-kinetics",
     "query": "quantitative cellular uptake of LNPs at multiple time points",
     "relevant": [
        {"paper": "CU_008_Development_of_amino_acid....pdf", "grade": 2,
         "pages": [9], "evidence": "Flow cytometry uptake in PC3 cells at 4 h"}
     ],
     "relevant_folder": "Cellular_Uptake_Papers",   # optional weak labels
     "filters": {"in_vitro": true}}                 # optional SearchFilters

``paper`` may be a paper_id, a DOI, a filename, or a relative path. ``grade`` is
graded relevance (1 = relevant, 2 = highly relevant) used by NDCG; binary
metrics treat grade >= 1 as relevant. ``relevant_folder`` marks every paper in
that sub-folder as relevant (grade 1) - useful as a cheap, noisy baseline.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from lnp_rag.models import PaperMeta


@dataclass
class LabeledQuery:
    query_id: str
    query: str
    grades: dict[str, int]  # paper_id -> grade
    pages: dict[str, set[int]] = field(default_factory=dict)  # paper_id -> relevant pages
    filters: dict = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)


def resolve_paper(ref: str, papers: dict[str, PaperMeta]) -> str | None:
    ref_l = ref.strip().lower()
    if ref in papers:
        return ref
    for pid, m in papers.items():
        if m.doi and m.doi.lower() == ref_l.removeprefix("https://doi.org/"):
            return pid
    for pid, m in papers.items():
        if m.filename.lower() == ref_l or any(p.lower() == ref_l or p.lower().endswith("/" + ref_l) for p in m.paths):
            return pid
    return None


def load_dataset(path: Path, papers: dict[str, PaperMeta]) -> list[LabeledQuery]:
    out = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        row = json.loads(line)
        lq = LabeledQuery(row.get("query_id", f"q{line_no}"), row["query"], {}, {}, row.get("filters", {}))
        if folder := row.get("relevant_folder"):
            for pid, m in papers.items():
                if folder in m.folder_labels:
                    lq.grades[pid] = max(lq.grades.get(pid, 0), 1)
        for rel in row.get("relevant", []):
            pid = resolve_paper(str(rel["paper"]), papers)
            if pid is None:
                lq.unresolved.append(str(rel["paper"]))
                continue
            lq.grades[pid] = max(lq.grades.get(pid, 0), int(rel.get("grade", 1)))
            if rel.get("pages"):
                lq.pages.setdefault(pid, set()).update(int(p) for p in rel["pages"])
        out.append(lq)
    return out


# --- metrics -------------------------------------------------------------------


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for p in ranked[:k] if p in relevant) / k if k else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for p in ranked[:k] if p in relevant) / len(relevant) if relevant else 0.0


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(p in relevant for p in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, p in enumerate(ranked, start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    def dcg(gains: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))

    actual = dcg([grades.get(p, 0) for p in ranked[:k]])
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def evaluate_ranking(
    ranked: list[str], grades: dict[str, int], ks: list[int],
    returned_pages: dict[str, set[int]] | None = None, gold_pages: dict[str, set[int]] | None = None,
) -> dict[str, float]:
    relevant = {p for p, g in grades.items() if g >= 1}
    m: dict[str, float] = {"mrr": reciprocal_rank(ranked, relevant)}
    for k in ks:
        m[f"p@{k}"] = precision_at_k(ranked, relevant, k)
        m[f"r@{k}"] = recall_at_k(ranked, relevant, k)
        m[f"ndcg@{k}"] = ndcg_at_k(ranked, grades, k)
        m[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
    if gold_pages:
        # Of the labeled (paper, page) evidence locations for papers we returned,
        # how many did the returned evidence point at?
        returned_pages = returned_pages or {}
        total = hits = 0
        for pid, pages in gold_pages.items():
            if pid not in ranked:
                continue
            total += len(pages)
            hits += len(pages & returned_pages.get(pid, set()))
        m["page_hit_rate"] = hits / total if total else float("nan")
    return m


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and not math.isnan(r[k])]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out
