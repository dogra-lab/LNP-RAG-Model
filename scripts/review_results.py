#!/usr/bin/env python
"""Manually validate search results, and turn your judgements into eval labels.

Step 1 - generate a review pack for a query:

    python scripts/review_results.py review "quantitative LNP uptake over time" -k 15 \
        --out reports/review_uptake

  Writes:
    review.html  - each result with its excerpts and a rendering of the PDF page
                   with the excerpt highlighted, so you can check the claim
                   against the actual figure/table without hunting for it.
    review.csv   - one row per result; fill in the judgement columns:
                   relevant (0/1/2), has_quantitative_uptake (y/n),
                   multiple_timepoints (y/n), evidence_pages, notes.

Step 2 - convert the filled CSV into an evaluation entry:

    python scripts/review_results.py labels reports/review_uptake/review.csv \
        --append eval/labeled_queries.jsonl --query-id uptake-kinetics
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lnp_rag.config import get_settings  # noqa: E402
from lnp_rag.logging_setup import setup_logging  # noqa: E402
from lnp_rag.models import SearchRequest  # noqa: E402
from lnp_rag.services import Services  # noqa: E402

JUDGEMENT_COLS = ["relevant", "has_quantitative_uptake", "multiple_timepoints", "evidence_pages", "notes"]


def highlight_page(pdf_path: Path, page_no: int, excerpt: str, out_png: Path, zoom: float = 1.4) -> bool:
    """Render ``page_no`` with phrases from ``excerpt`` highlighted. Returns True if any phrase was found."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_no - 1]
        found = False
        sentences = re.split(r"(?<=[.;])\s+", excerpt)
        for sent in sentences[:12]:
            words = sent.split()
            for i in range(0, max(1, len(words) - 5), 6):
                phrase = " ".join(words[i : i + 6])
                if len(phrase) < 15:
                    continue
                for quad in page.search_for(phrase, quads=True)[:3]:
                    page.add_highlight_annot(quad)
                    found = True
        page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True).save(out_png)
        return found
    finally:
        doc.close()  # annotations are never saved to the original PDF


def cmd_review(args) -> int:
    settings = get_settings()
    setup_logging("WARNING", settings.log_format)
    services = Services(settings)
    resp = services.searcher.search(SearchRequest(query=args.query, top_k=args.k, synthesize=not args.no_synthesis))
    out: Path = args.out
    (out / "img").mkdir(parents=True, exist_ok=True)
    root = Path(services.catalog.get_kv("papers_dir") or settings.papers_dir)

    rows, parts = [], [
        "<!doctype html><meta charset='utf-8'><title>Result review</title>",
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;"
        "color:#1d1d1f;background:#fff}.r{border:1px solid #ddd;border-radius:8px;padding:16px;margin:20px 0}"
        "blockquote{border-left:3px solid #c8a200;margin:8px 0;padding:4px 12px;background:#fffbea}"
        "img{max-width:100%;border:1px solid #ccc;margin:8px 0}.m{color:#555;font-size:13px}"
        "@media (prefers-color-scheme:dark){body{background:#111;color:#eee}.r{border-color:#333}"
        "blockquote{background:#2a2614}.m{color:#aaa}}</style>",
        f"<h1>Review: {html.escape(args.query)}</h1>",
        f"<p class=m>{len(resp.results)} results. Check each highlighted page against the claim, "
        "then fill in review.csv.</p>",
    ]
    for w in resp.warnings:
        parts.append(f"<p class=m>⚠ {html.escape(w)}</p>")
    for r in resp.results:
        meta = services.catalog.get_paper(r.paper_id)
        pdf = next((root / p for p in meta.paths if (root / p).is_file()), None) if meta else None
        parts.append(f"<div class=r><h2>{r.rank}. {html.escape(r.title)}</h2>")
        parts.append(
            f"<p class=m>{html.escape(r.filename)} · pages {r.pages} · {r.study_type.value} · "
            f"evidence: {r.evidence_strength} · score {r.relevance_score:.3f}"
            f"{' · REVIEW ARTICLE' if r.is_review else ''}</p>"
        )
        parts.append(f"<p><b>Explanation ({r.explanation_source}):</b> {html.escape(r.explanation)}</p>")
        for e in r.evidence[: args.evidence_per_paper]:
            parts.append(
                f"<p class=m>[{e.evidence_id}] p.{e.page} · {html.escape(e.section)} · {e.chunk_type.value}"
                f"{' · time points: ' + ', '.join(e.features.timepoints) if e.features.timepoints else ''}</p>"
                f"<blockquote>{html.escape(e.text)}</blockquote>"
            )
            if pdf:
                png = out / "img" / f"{r.rank:02d}_{e.evidence_id}.png"
                ok = highlight_page(pdf, e.page, e.text, png)
                note = "" if ok else " (excerpt text not found on page - check manually)"
                parts.append(f"<p class=m>Page {e.page}{note}</p><img src='img/{png.name}' loading=lazy>")
        parts.append("</div>")
        rows.append({
            "rank": r.rank, "paper_id": r.paper_id, "filename": r.filename, "title": r.title,
            "doi": r.doi or "", "pages": " ".join(map(str, r.pages)), "study_type": r.study_type.value,
            "evidence_strength": r.evidence_strength, "explanation": r.explanation,
            "top_excerpt": r.evidence[0].text[:500] if r.evidence else "",
            **{c: "" for c in JUDGEMENT_COLS},
        })
    (out / "review.html").write_text("\n".join(parts))
    with open(out / "review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["rank"])
        w.writeheader()
        w.writerows(rows)
    (out / "query.json").write_text(json.dumps({"query": args.query}))
    print(f"Wrote {out / 'review.html'} and {out / 'review.csv'}")
    return 0


def cmd_labels(args) -> int:
    csv_path: Path = args.csv
    query = args.query or json.loads((csv_path.parent / "query.json").read_text())["query"]
    relevant = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            grade = (row.get("relevant") or "").strip()
            if not grade or int(grade) <= 0:
                continue
            pages = [int(p) for p in re.findall(r"\d+", row.get("evidence_pages") or "")]
            relevant.append({
                "paper": row["paper_id"], "grade": int(grade), "pages": pages,
                "evidence": (row.get("notes") or "").strip(),
            })
    entry = {"query_id": args.query_id or csv_path.parent.name, "query": query, "relevant": relevant}
    if args.append:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with open(args.append, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(f"Appended {len(relevant)} relevant papers for '{entry['query_id']}' to {args.append}")
    else:
        print(json.dumps(entry, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review", help="build an HTML/CSV review pack for a query")
    r.add_argument("query")
    r.add_argument("-k", type=int, default=15)
    r.add_argument("--out", type=Path, default=Path("reports/review"))
    r.add_argument("--evidence-per-paper", type=int, default=2)
    r.add_argument("--no-synthesis", action="store_true")
    r.set_defaults(func=cmd_review)
    lab = sub.add_parser("labels", help="convert a filled review.csv into an eval JSONL entry")
    lab.add_argument("csv", type=Path)
    lab.add_argument("--append", type=Path)
    lab.add_argument("--query-id")
    lab.add_argument("--query")
    lab.set_defaults(func=cmd_labels)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
