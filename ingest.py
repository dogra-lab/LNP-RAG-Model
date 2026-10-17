"""Build the paper indexes (run once, re-run when papers change).

  python ingest.py                 # library (./papers) + validated set (./paper used)
  python ingest.py --only-validated
"""

import argparse
from pathlib import Path

from lnp_rag.index import build_index, build_validated_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", type=Path, default=Path("papers"))
    ap.add_argument("--validated", type=Path, default=Path("paper used"))
    ap.add_argument("--index", type=Path, default=Path("index"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-validated", action="store_true", help="rebuild only the validated-papers index")
    args = ap.parse_args()
    if not args.only_validated:
        build_index(args.papers, args.index, args.workers)
    if args.validated.exists():
        build_validated_index(args.validated, args.index / "validated")
