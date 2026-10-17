"""Web API for the LNP paper search. Run: .venv/bin/uvicorn app:app --port 8000"""

from __future__ import annotations

import os
import threading
import traceback
import uuid
from dataclasses import fields
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lnp_rag.index import PaperIndex
from lnp_rag.pipeline import SearchParams, export_included, run_search, save_results
from lnp_rag.profile import load_criteria
from lnp_rag.screen import DEFAULT_MODELS
from lnp_rag.steps import STEPS

load_dotenv()
INDEX_DIR = Path("index")
RESULTS_DIR = Path("results")
INCLUDED_DIR = Path("included_papers")
FRONTEND_DIST = Path("frontend/dist")

app = FastAPI(title="LNP Paper Search")
index = PaperIndex(INDEX_DIR)
validated = PaperIndex(INDEX_DIR / "validated") if (INDEX_DIR / "validated" / "papers.jsonl").exists() else None
criteria = load_criteria(INDEX_DIR / "validated")  # written by learn.py
papers_by_id = {p["id"]: p for p in index.papers + (validated.papers if validated else [])}
search_lock = threading.Lock()  # the embedding model is not safe to call from several threads
jobs: dict[str, dict] = {}


class SearchRequest(BaseModel):
    query: str
    step: str | None = None
    step_name: str | None = None
    step_description: str | None = None
    guidance: str | None = None
    strict_kinetic: bool = False
    top: int = 100
    candidates: int = 500
    only_relevant: bool = False
    no_llm: bool = False
    provider: str | None = None
    model: str | None = None
    effort: str = "low"
    use_validated: bool = True


def _run_job(job_id: str, params: SearchParams) -> None:
    job = jobs[job_id]

    def on_progress(done: int, total: int) -> None:
        job.update(status="screening", done=done, total=total)

    try:
        job["status"] = "retrieving"
        with search_lock:
            output = run_search(index, params, INDEX_DIR / "screen_cache.jsonl", on_progress=on_progress,
                                validated=validated, criteria=criteria)
        stem = save_results(output, RESULTS_DIR)
        folder, n_included = export_included(output, INCLUDED_DIR / stem.name)
        job.update(status="done", result=output, saved=f"{stem}.csv", included_dir=str(folder), n_included=n_included)
    except Exception as e:  # surface any failure (bad key, unknown model...) to the UI
        traceback.print_exc()
        job.update(status="error", error=f"{type(e).__name__}: {e}")


@app.get("/api/config")
def config() -> dict:
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {
        "steps": [{"id": k, **v} for k, v in STEPS.items()],
        "paper_count": len(index.papers),
        "validated_count": len(validated.papers) if validated else 0,
        "learned_criteria": criteria,
        "providers": {"openai": has_openai, "anthropic": has_anthropic},
        "default_provider": "openai" if has_openai else ("anthropic" if has_anthropic else None),
        "default_models": DEFAULT_MODELS,
    }


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    if not req.query.strip() and not (req.step or req.step_name):
        raise HTTPException(400, "Enter a query or choose a step")
    data = req.model_dump()
    params = SearchParams(**{f.name: data[f.name] for f in fields(SearchParams) if f.name in data})
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "done": 0, "total": 0}
    threading.Thread(target=_run_job, args=(job_id, params), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[job_id]


@app.get("/api/papers/{paper_id}/pdf")
def paper_pdf(paper_id: str, file: int = 0):
    """file=0 is the paper itself; file=1.. are its supplements (validated papers only)."""
    paper = papers_by_id.get(paper_id)
    paths = paper["all_paths"] if paper else []
    if not 0 <= file < len(paths) or not Path(paths[file]).exists():
        raise HTTPException(404, "Paper not found")
    return FileResponse(paths[file], media_type="application/pdf", content_disposition_type="inline",
                        filename=Path(paths[file]).name)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
