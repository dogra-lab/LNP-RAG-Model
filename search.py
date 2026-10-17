"""Retrieve the top-N papers for a query: hybrid retrieval, then LLM screening.

Examples:
  python search.py --step uptake_pkpd --candidates 500    # PK/PD uptake data, learned from 'paper used/'
  python search.py "LNP uptake kinetics in hepatocytes" --step cellular_uptake --strict-kinetic
  python search.py "ApoE adsorption on ionizable LNPs" --step-name "Protein corona" \
      --step-description "plasma protein adsorption onto LNPs"
  python search.py "endosomal escape efficiency" --no-llm      # retrieval only, no API calls
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from lnp_rag.index import PaperIndex
from lnp_rag.pipeline import SearchParams, export_included, run_search, save_results
from lnp_rag.profile import load_criteria
from lnp_rag.steps import STEPS


def main() -> None:
    load_dotenv()  # reads OPENAI_API_KEY / ANTHROPIC_API_KEY from .env in the project folder
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", default="", help="free-text search query (optional with --step)")
    ap.add_argument("--step", choices=sorted(STEPS), help="preset LNP-journey step (see lnp_rag/steps.py)")
    ap.add_argument("--step-name", help="override/define the step name")
    ap.add_argument("--step-description", help="override/define the step description")
    ap.add_argument("--guidance", default=None, help="step-specific guidance for the screener")
    ap.add_argument("--strict-kinetic", action="store_true", help="require time-resolved/kinetic data")
    ap.add_argument("--top", type=int, default=100, help="number of papers to return (default 100)")
    ap.add_argument("--candidates", type=int, default=200, help="papers retrieved for LLM screening (default 200)")
    ap.add_argument("--only-relevant", action="store_true", help="drop papers the screener marks 'exclude'")
    ap.add_argument("--no-llm", action="store_true", help="skip LLM screening; rank by retrieval only")
    ap.add_argument("--provider", choices=["openai", "anthropic"], default=None,
                    help="LLM for screening (default: openai if OPENAI_API_KEY is set, else anthropic)")
    ap.add_argument("--model", default=None, help="model id (default: gpt-5-mini / claude-opus-5-5)")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--workers", type=int, default=16, help="parallel API requests")
    ap.add_argument("--no-validated-similarity", dest="use_validated", action="store_false",
                    help="don't learn from the validated set (criteria, facets, evidence passages)")
    ap.add_argument("--index", type=Path, default=Path("index"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = vars(ap.parse_args())

    index_dir, out_dir = args.pop("index"), args.pop("out")
    params = SearchParams(**args)
    validated = PaperIndex(index_dir / "validated") if (index_dir / "validated" / "papers.jsonl").exists() else None
    criteria = load_criteria(index_dir / "validated")
    output = run_search(PaperIndex(index_dir), params, index_dir / "screen_cache.jsonl", validated=validated,
                        criteria=criteria)
    rows, meta = output["results"], output["meta"]
    if meta["learned_criteria"]:
        print("Using criteria learned from the validated papers (index/validated/criteria.json)")
    if meta["model"]:
        print(f"Screened with {meta['model']}")
    if meta["errors"]:
        print(f"WARNING: {meta['errors']} papers failed screening (see 'rationale' column).")

    if output["validated"]:
        print(f"Validated papers ({len(output['validated'])}), by relevance:")
        for r in output["validated"]:
            print(f"  V{r['relevance_rank']:<3} {r['label'][:28]:28} {r['title'][:80]}")
        print("\nAdditional papers:")
    for i, r in enumerate(rows, 1):
        tag = f"[{r['verdict']:<10} {r['score']:.2f} {r['study_type']:<12}] " if "verdict" in r else ""
        data = f"  <{', '.join(r['data_types'])}>" if r.get("data_types") else ""
        review = " (review)" if r["is_review"] else ""
        print(f"{i:3}. {tag}{r['title'][:110]}{review}{data}")
    if rows and "verdict" in rows[0]:
        counts = {v: sum(r["verdict"] == v for r in rows) for v in ("include", "borderline", "exclude")}
        print(f"\nTop {len(rows)}: {counts['include']} include, {counts['borderline']} borderline, "
              f"{counts['exclude']} exclude")
    stem = save_results(output, out_dir)
    print(f"Saved {stem}.csv and {stem}.json")
    if meta["model"]:
        folder, n = export_included(output, Path("included_papers") / stem.name)
        print(f"Copied the PDFs of {n} included papers to {folder}/")


if __name__ == "__main__":
    main()
