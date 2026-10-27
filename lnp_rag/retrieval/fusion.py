"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = sum over ranked lists L of  w_L / (k + rank_L(d))

RRF only uses ranks, so it fuses cosine similarities and BM25 scores without
having to calibrate them against each other.
"""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[float, list[str]]], k: int = 60
) -> list[tuple[str, float]]:
    """``ranked_lists`` is [(weight, [doc_id best-first])]. Returns [(doc_id, score)] best-first."""
    scores: dict[str, float] = defaultdict(float)
    for weight, ids in ranked_lists:
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
