#!/usr/bin/env python
"""Index (or incrementally re-index) the PDF folder.

    python scripts/index_papers.py                    # uses PAPERS_DIR from .env
    python scripts/index_papers.py --papers-dir ~/pdfs
    python scripts/index_papers.py --rebuild          # wipe and rebuild everything
    python scripts/index_papers.py --stats            # show index statistics
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lnp_rag.config import get_settings  # noqa: E402
from lnp_rag.logging_setup import setup_logging  # noqa: E402
from lnp_rag.services import Services  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers-dir", type=Path, default=None)
    ap.add_argument("--rebuild", action="store_true", help="clear the index and re-process every PDF")
    ap.add_argument("--stats", action="store_true", help="print index statistics and exit")
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    services = Services(settings)

    if args.stats:
        print(json.dumps(services.catalog.stats(), indent=2))
        return 0

    if args.rebuild:
        services.prepare_rebuild()
    indexer = services.indexer
    done = threading.Event()

    def progress():
        while not done.wait(5):
            st = indexer.status
            print(
                f"  [{st.state}] {st.processed_files}/{st.total_files} PDFs, "
                f"{st.chunks_indexed} chunks - {st.message}",
                file=sys.stderr,
            )

    threading.Thread(target=progress, daemon=True).start()
    t0 = time.time()
    status = indexer.run(args.papers_dir, rebuild=args.rebuild)
    done.set()
    print(json.dumps(
        {
            "state": status.state,
            "message": status.message,
            "new_files": status.new,
            "modified_files": status.modified,
            "deleted_files": status.deleted,
            "unchanged_files": status.unchanged,
            "papers_processed": status.processed_files,
            "chunks_indexed": status.chunks_indexed,
            "failed": status.failed,
            "seconds": round(time.time() - t0, 1),
        },
        indent=2,
    ))
    return 0 if status.state == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
