"""FastAPI backend.

    uvicorn lnp_rag.api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from lnp_rag import __version__
from lnp_rag.config import get_settings
from lnp_rag.logging_setup import setup_logging
from lnp_rag.models import IndexRequest, IndexStatus, PaperMeta, SearchRequest, SearchResponse
from lnp_rag.services import Services

log = logging.getLogger("lnp_rag.api")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class AppState:
    def __init__(self, services: Services):
        self.services = services
        self.index_thread: threading.Thread | None = None
        self.index_lock = threading.Lock()

    @property
    def indexing(self) -> bool:
        return self.index_thread is not None and self.index_thread.is_alive()


def create_app(services: Services | None = None, warmup: bool = True) -> FastAPI:
    settings = services.settings if services else get_settings()
    setup_logging(settings.log_level, settings.log_format)
    state = AppState(services or Services(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warmup:
            # Load models in the background so the first search is fast.
            threading.Thread(target=_warmup, args=(state.services,), daemon=True).start()
        yield

    app = FastAPI(title="LNP Paper Search", version=__version__, lifespan=lifespan)
    app.state.app_state = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled_error", extra={"path": request.url.path})
        return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}"})

    # --- health / stats -------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        s = state.services
        return {
            "status": "ok",
            "version": __version__,
            "indexed_papers": len(s.catalog.paper_ids()),
            "indexing": state.indexing,
            "llm_enabled": s.llm is not None and s.settings.synthesis_enabled,
            "llm_provider": s.llm.name if s.llm else None,
            "llm_model": s.llm.model if s.llm else None,
            "embedding_model": s.embedding_model_id,
            "reranker": s.settings.reranker_model if s.settings.reranker_enabled else None,
        }

    @app.get("/stats")
    def stats() -> dict:
        out = state.services.catalog.stats()
        out["papers_dir"] = state.services.catalog.get_kv("papers_dir") or str(state.services.settings.papers_dir)
        return out

    # --- indexing ---------------------------------------------------------------
    @app.post("/index", response_model=IndexStatus, status_code=202)
    def start_index(req: IndexRequest) -> IndexStatus:
        with state.index_lock:
            if state.indexing:
                raise HTTPException(409, "Indexing is already running")
            papers_dir = Path(req.papers_dir).expanduser() if req.papers_dir else None
            if papers_dir is not None and not papers_dir.is_dir():
                raise HTTPException(400, f"Folder not found: {papers_dir}")
            if req.rebuild:
                state.services.prepare_rebuild()
            indexer = state.services.indexer
            indexer.status = IndexStatus(state="scanning", started_at=time.time(), message="Starting")
            state.index_thread = threading.Thread(
                target=indexer.run, kwargs={"papers_dir": papers_dir, "rebuild": req.rebuild}, daemon=True
            )
            state.index_thread.start()
            return indexer.status

    @app.get("/index/status", response_model=IndexStatus)
    def index_status() -> IndexStatus:
        return state.services.indexer.status

    @app.delete("/index")
    def clear_index() -> dict:
        with state.index_lock:
            if state.indexing:
                raise HTTPException(409, "Cannot clear the index while indexing is running")
            s = state.services
            s.catalog.clear()
            s.vectors.reset()
            s.bm25.build(s.catalog)
            return {"cleared": True}

    @app.post("/upload")
    async def upload(files: list[UploadFile] = File(...)) -> dict:
        target = state.services.settings.uploads_dir
        target.mkdir(parents=True, exist_ok=True)
        saved, rejected = [], []
        for f in files:
            name = _safe_filename(f.filename or "upload.pdf")
            data = await f.read(MAX_UPLOAD_BYTES + 1)
            if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
                rejected.append({"file": f.filename, "reason": "not a PDF"})
                continue
            if len(data) > MAX_UPLOAD_BYTES:
                rejected.append({"file": f.filename, "reason": "file too large"})
                continue
            (target / name).write_bytes(data)
            saved.append(name)
        return {"saved": saved, "rejected": rejected, "folder": str(target),
                "note": "Run indexing to add uploaded papers to the index."}

    # --- search -----------------------------------------------------------------
    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        try:
            return await run_in_threadpool(state.services.searcher.search, req)
        except RuntimeError as exc:  # e.g. embedding-model mismatch
            raise HTTPException(409, str(exc)) from exc

    # --- papers -----------------------------------------------------------------
    @app.get("/papers/{paper_id}", response_model=PaperMeta)
    def paper(paper_id: str) -> PaperMeta:
        meta = state.services.catalog.get_paper(paper_id)
        if not meta:
            raise HTTPException(404, "Paper not found")
        return meta

    @app.get("/papers/{paper_id}/pdf")
    def paper_pdf(paper_id: str) -> FileResponse:
        s = state.services
        meta = s.catalog.get_paper(paper_id)
        if not meta or not meta.paths:
            raise HTTPException(404, "Paper not found")
        root = Path(os.path.abspath(s.catalog.get_kv("papers_dir") or s.settings.papers_dir))
        for rel in meta.paths:
            # Never serve files outside the indexed folder (".." is normalised away first;
            # symlinks inside the folder are allowed, as the scanner follows them too).
            path = Path(os.path.normpath(root / rel))
            if path.is_relative_to(root) and path.is_file():
                return FileResponse(
                    path,
                    media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{_safe_filename(path.name)}"'},
                )
        raise HTTPException(404, "PDF file is no longer on disk; re-index the folder")

    return app


def _safe_filename(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^\w.\- ()]+", "_", name)[:200] or "upload.pdf"


def _warmup(services: Services) -> None:
    try:
        _ = services.searcher
        services.embedder.embed_query("warmup")
        if services.reranker:
            services.reranker.score("warmup", ["warmup"])
        log.info("warmup_complete")
    except Exception:
        log.exception("warmup_failed")


app = create_app()
