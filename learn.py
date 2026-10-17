"""Learn the selection logic of the validated papers ('paper used/') for later searches.

  python learn.py                    # uses the research question below
  python learn.py --question "..."   # or your own

Re-run after changing 'paper used/' (and after `python ingest.py --only-validated`).
Writes index/validated/profiles.jsonl (one evidence profile per paper) and index/validated/criteria.json.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from lnp_rag.index import PaperIndex
from lnp_rag.profile import learn_from_validated

QUESTION = (
    "Find research papers reporting quantitative, time-course or dose-dependent experimental data on LNP "
    "cellular uptake for PK/PD modeling, including uptake concentrations, intracellular accumulation, and "
    "endosomal escape, with accessible figures, tables, or supplementary datasets."
)

if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--index", type=Path, default=Path("index"))
    ap.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    provider = args.provider or ("openai" if os.environ.get("OPENAI_API_KEY") else "anthropic")
    vdir = args.index / "validated"
    criteria = learn_from_validated(PaperIndex(vdir), vdir, args.question, provider=provider, model=args.model)
    print(f"Learned from {criteria['n_papers']} validated papers:\n\n{criteria['summary']}\n")
    print("Inclusion criteria:\n" + "\n".join(f"  - {c}" for c in criteria["criteria"]))
    print("Exclusions:\n" + "\n".join(f"  - {c}" for c in criteria["exclusions"]))
    print("Search facets:\n" + "\n".join(f"  - {c}" for c in criteria["facets"]))
    print(f"\nSaved {vdir / 'criteria.json'} and {vdir / 'profiles.jsonl'}")
