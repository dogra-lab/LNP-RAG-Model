#!/usr/bin/env python
"""Search from the command line (no server needed).

    python scripts/search.py "quantitative cellular uptake of LNPs over time" -k 5
    python scripts/search.py "..." --in-vitro --no-synthesis --json
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lnp_rag.config import get_settings  # noqa: E402
from lnp_rag.logging_setup import setup_logging  # noqa: E402
from lnp_rag.models import SearchFilters, SearchRequest  # noqa: E402
from lnp_rag.services import Services  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("-k", "--top-k", type=int, default=10)
    ap.add_argument("--in-vitro", action="store_true")
    ap.add_argument("--in-vivo", action="store_true")
    ap.add_argument("--year-min", type=int)
    ap.add_argument("--year-max", type=int)
    ap.add_argument("--exclude-reviews", action="store_true")
    ap.add_argument("--has-quantitative", action="store_true")
    ap.add_argument("--no-synthesis", action="store_true", help="skip the LLM call")
    ap.add_argument("--json", action="store_true", help="print the raw JSON response")
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    req = SearchRequest(
        query=args.query,
        top_k=args.top_k,
        synthesize=not args.no_synthesis,
        filters=SearchFilters(
            in_vitro=args.in_vitro or None,
            in_vivo=args.in_vivo or None,
            year_min=args.year_min,
            year_max=args.year_max,
            exclude_reviews=args.exclude_reviews,
            has_quantitative=args.has_quantitative or None,
        ),
    )
    resp = Services(settings).searcher.search(req)
    if args.json:
        print(resp.model_dump_json(indent=2))
        return 0

    for w in resp.warnings:
        print(f"! {w}")
    print(f"Query variants: {resp.expanded_queries}")
    print(f"Timings (ms): {resp.timings_ms}\n")
    for r in resp.results:
        authors = ", ".join(r.authors[:3]) + (" et al." if len(r.authors) > 3 else "")
        print(f"#{r.rank}  [{r.relevance_score:.3f}] {r.title}")
        print(f"     {authors or 'authors n/a'} | {r.year or 'year n/a'} | DOI: {r.doi or 'n/a'}")
        print(f"     {r.filename} | pages {r.pages} | {r.study_type.value}"
              f"{' | REVIEW' if r.is_review else ''} | evidence: {r.evidence_strength}")
        print(textwrap.indent(textwrap.fill(r.explanation, 100), "     "))
        if r.missing_evidence:
            print(f"     Missing: {r.missing_evidence}")
        for e in r.evidence[:3]:
            snippet = textwrap.shorten(e.text, 260)
            print(f"       [{e.evidence_id}] p.{e.page} {e.section} ({e.chunk_type.value}): {snippet}")
        print(f"     Open: {r.pdf_url}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
