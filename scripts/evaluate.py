#!/usr/bin/env python
"""Evaluate retrieval quality against a labeled query set.

    python scripts/evaluate.py eval/labeled_queries.jsonl
    python scripts/evaluate.py eval/labeled_queries.jsonl -k 1 5 10 20 --out reports/eval.json
    python scripts/evaluate.py eval/labeled_queries.jsonl --synthesis    # include LLM re-scoring
    python scripts/evaluate.py eval/labeled_queries.jsonl --no-rerank    # ablation: fusion only

Metrics are computed at the paper level (Precision@K, Recall@K, MRR, NDCG@K,
Hit@K) plus a page-level evidence hit rate when labels include pages.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lnp_rag.config import get_settings  # noqa: E402
from lnp_rag.evaluation import evaluate_ranking, load_dataset, mean_metrics  # noqa: E402
from lnp_rag.logging_setup import setup_logging  # noqa: E402
from lnp_rag.models import SearchFilters, SearchRequest  # noqa: E402
from lnp_rag.services import Services  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("-k", "--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    ap.add_argument("--synthesis", action="store_true", help="include LLM synthesis (costs API calls)")
    ap.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder (ablation)")
    ap.add_argument("--expansion", choices=["none", "rules", "llm"], help="override QUERY_EXPANSION_MODE")
    ap.add_argument("--out", type=Path, help="write per-query and mean metrics as JSON")
    args = ap.parse_args()

    settings = get_settings()
    if args.no_rerank:
        settings = settings.model_copy(update={"reranker_enabled": False})
    if args.expansion:
        settings = settings.model_copy(update={"query_expansion_mode": args.expansion})
    setup_logging("WARNING", settings.log_format)
    services = Services(settings)
    papers = services.catalog.all_papers()
    if not papers:
        print("Index is empty - run scripts/index_papers.py first.", file=sys.stderr)
        return 1
    dataset = load_dataset(args.dataset, papers)
    depth = max(args.ks)

    rows, per_query = [], []
    for lq in dataset:
        if lq.unresolved:
            print(f"[{lq.query_id}] WARNING unresolved paper refs: {lq.unresolved}", file=sys.stderr)
        if not lq.grades:
            print(f"[{lq.query_id}] skipped: no resolvable relevant papers", file=sys.stderr)
            continue
        resp = services.searcher.search(
            SearchRequest(query=lq.query, top_k=depth, synthesize=args.synthesis,
                          filters=SearchFilters(**lq.filters))
        )
        ranked = [r.paper_id for r in resp.results]
        returned_pages = {r.paper_id: set(r.pages) for r in resp.results}
        m = evaluate_ranking(ranked, lq.grades, args.ks, returned_pages, lq.pages)
        rows.append(m)
        per_query.append({
            "query_id": lq.query_id,
            "n_relevant": sum(1 for g in lq.grades.values() if g >= 1),
            "metrics": m,
            "top": [{"rank": r.rank, "paper_id": r.paper_id, "title": r.title,
                     "relevant": lq.grades.get(r.paper_id, 0)} for r in resp.results[:10]],
        })
        print(f"[{lq.query_id}] n_rel={per_query[-1]['n_relevant']:<4} " +
              " ".join(f"{k}={v:.3f}" for k, v in m.items() if k in ("mrr", "p@5", "r@10", "ndcg@10", "hit@5")))

    if not rows:
        print("No evaluable queries.", file=sys.stderr)
        return 1
    mean = mean_metrics(rows)
    print("\nMean over", len(rows), "queries")
    width = max(len(k) for k in mean)
    for k, v in mean.items():
        print(f"  {k:<{width}}  {v:.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"mean": mean, "queries": per_query}, indent=2))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
