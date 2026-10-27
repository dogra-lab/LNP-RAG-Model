"""Incremental indexing pipeline.

scan -> diff against catalog -> parse new papers in worker processes ->
embed in batches -> write vectors + catalog -> rebuild BM25.

Papers are identified by a hash of their *content*, so renaming or copying a PDF
into another folder never triggers re-embedding, and duplicate copies collapse
into one paper that remembers all its paths / folder labels.
"""
from __future__ import annotations

import hashlib
import logging
import os
import signal
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from lnp_rag.config import Settings
from lnp_rag.embeddings import Embedder
from lnp_rag.ingest.chunker import chunk_document, embedding_text
from lnp_rag.ingest.features import classify_study_type, looks_like_review
from lnp_rag.ingest.metadata import (
    CrossrefClient,
    extract_authors,
    extract_doi,
    extract_title,
    extract_year,
)
from lnp_rag.ingest.pdf_parser import parse_pdf
from lnp_rag.models import Chunk, ChunkType, IndexStatus, PaperMeta
from lnp_rag.store.bm25_index import BM25Index
from lnp_rag.store.catalog import Catalog
from lnp_rag.store.vector_store import VectorStore

log = logging.getLogger(__name__)

EMBED_BATCH_CHUNKS = 512


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def paper_id_for(sha: str) -> str:
    return sha[:16]


@dataclass
class ScanResult:
    files: dict[str, tuple[int, float, str]]  # rel path -> (size, mtime, sha)
    paths_by_paper: dict[str, list[str]]
    new_papers: set[str]
    deleted_papers: set[str]
    kept_papers: set[str]
    new_paths: int
    modified_paths: int
    deleted_paths: int
    unchanged_paths: int


def scan(papers_dir: Path, catalog: Catalog) -> ScanResult:
    known = catalog.files()
    files: dict[str, tuple[int, float, str]] = {}
    new_paths = modified = unchanged = 0
    for path in sorted(papers_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        if any(part.startswith(".") for part in path.relative_to(papers_dir).parts):
            continue
        rel = path.relative_to(papers_dir).as_posix()
        st = path.stat()
        prev = known.get(rel)
        if prev and prev[0] == st.st_size and abs(prev[1] - st.st_mtime) < 1e-3:
            files[rel] = prev  # size+mtime unchanged: trust the stored hash
            unchanged += 1
            continue
        sha = sha256_file(path)
        files[rel] = (st.st_size, st.st_mtime, sha)
        if prev is None:
            new_paths += 1
        elif prev[2] != sha:
            modified += 1
        else:
            unchanged += 1  # touched but identical content
    paths_by_paper: dict[str, list[str]] = defaultdict(list)
    for rel, (_, _, sha) in files.items():
        paths_by_paper[paper_id_for(sha)].append(rel)
    current = set(paths_by_paper)
    indexed = catalog.paper_ids()
    return ScanResult(
        files=files,
        paths_by_paper=dict(paths_by_paper),
        new_papers=current - indexed,
        deleted_papers=indexed - current,
        kept_papers=current & indexed,
        new_paths=new_paths,
        modified_paths=modified,
        deleted_paths=len(set(known) - set(files)),
        unchanged_paths=unchanged,
    )


def folder_labels(paths: list[str]) -> list[str]:
    labels = {p.split("/")[0] for p in paths if "/" in p}
    return sorted(labels)


# --- worker-process side --------------------------------------------------------


class PdfTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise PdfTimeout("PDF processing exceeded the time limit")


def process_pdf(abs_path: str, paper_id: str, rel_paths: list[str], opts: dict) -> dict:
    """Parse one PDF into paper metadata + chunks. Runs in a worker process.

    A per-PDF alarm guarantees one pathological file cannot stall the whole run.
    """
    timeout = int(opts.get("timeout_s") or 0)
    use_alarm = timeout > 0 and hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(timeout)
    try:
        return _process_pdf(abs_path, paper_id, rel_paths, opts)
    finally:
        if use_alarm:
            signal.alarm(0)


def _process_pdf(abs_path: str, paper_id: str, rel_paths: list[str], opts: dict) -> dict:
    path = Path(abs_path)
    doc = parse_pdf(
        path,
        ocr_enabled=opts["ocr_enabled"],
        ocr_min_chars=opts["ocr_min_chars"],
        ocr_dpi=opts["ocr_dpi"],
        ocr_lang=opts["ocr_lang"],
    )
    title, title_src = extract_title(doc)
    chunks, abstract = chunk_document(
        paper_id,
        doc.blocks,
        target_words=opts["target_words"],
        overlap_words=opts["overlap_words"],
        min_words=opts["min_words"],
    )
    section_names = sorted({b.section for b in doc.blocks if b.kind == "heading"})
    study_type, _ = classify_study_type([(c.section_category, c.features) for c in chunks])
    captions = [c for c in chunks if c.chunk_type in (ChunkType.FIGURE_CAPTION, ChunkType.TABLE_CAPTION)]
    evidential = [c for c in chunks if c.section_category in ("results", "methods", "figure", "table")]
    meta = PaperMeta(
        paper_id=paper_id,
        title=title,
        authors=extract_authors(doc, title),
        doi=(doi := extract_doi(doc)),
        year=extract_year(doc, doi),
        abstract=abstract[:4000],
        filename=path.name,
        paths=sorted(rel_paths),
        folder_labels=folder_labels(rel_paths),
        n_pages=doc.n_pages,
        n_figures=len({c.label for c in captions if c.chunk_type == ChunkType.FIGURE_CAPTION}),
        n_tables=len({c.label for c in captions if c.chunk_type == ChunkType.TABLE_CAPTION}),
        has_quantitative=sum(c.features.has_quantitative for c in evidential) >= 2,
        study_type=study_type,
        is_review=looks_like_review(title, abstract, section_names),
        ocr_pages=doc.ocr_pages,
        metadata_source=title_src,
    )
    return {
        "meta": meta.model_dump(mode="json"),
        "chunks": [c.model_dump(mode="json") for c in chunks],
        "warnings": doc.warnings,
    }


# --- orchestrator ---------------------------------------------------------------


class Indexer:
    def __init__(self, settings: Settings, catalog: Catalog, vectors: VectorStore, bm25: BM25Index,
                 embedder_factory: Callable[[], Embedder]):
        self.settings = settings
        self.catalog = catalog
        self.vectors = vectors
        self.bm25 = bm25
        self._embedder_factory = embedder_factory
        self.status = IndexStatus()

    def _set(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.status, k, v)

    def run(self, papers_dir: Path | None = None, rebuild: bool = False) -> IndexStatus:
        papers_dir = (papers_dir or self.settings.papers_dir).expanduser().resolve()
        self.status = IndexStatus(state="scanning", started_at=time.time(), message=f"Scanning {papers_dir}")
        try:
            self._run(papers_dir, rebuild)
            self._set(state="done", finished_at=time.time(),
                      message=f"Indexed {len(self.catalog.paper_ids())} papers")
        except Exception as exc:
            log.exception("indexing_failed")
            self._set(state="error", message=str(exc), finished_at=time.time())
        return self.status

    def _run(self, papers_dir: Path, rebuild: bool) -> None:
        if not papers_dir.is_dir():
            raise FileNotFoundError(f"PDF folder not found: {papers_dir}")
        stored_dir = self.catalog.get_kv("papers_dir")
        model_name = self.catalog.get_kv("embedding_model")
        if rebuild or (stored_dir and stored_dir != str(papers_dir)):
            # Relative paths are only meaningful for one root; switching roots rebuilds.
            log.info("index_reset", extra={"rebuild": rebuild, "old_dir": stored_dir})
            self.catalog.clear()
            self.vectors.reset()
        elif model_name and model_name != self.vectors.embedding_model and self.catalog.paper_ids():
            raise RuntimeError(
                f"Index was built with embedding model '{model_name}'; current model is "
                f"'{self.vectors.embedding_model}'. Run a rebuild."
            )
        self.catalog.set_kv("papers_dir", str(papers_dir))

        result = scan(papers_dir, self.catalog)
        self._set(
            total_files=len(result.new_papers),
            new=result.new_paths,
            modified=result.modified_paths,
            deleted=result.deleted_paths,
            unchanged=result.unchanged_paths,
        )
        log.info(
            "scan_complete",
            extra={
                "files": len(result.files),
                "unique_papers": len(result.paths_by_paper),
                "to_process": len(result.new_papers),
                "to_delete": len(result.deleted_papers),
            },
        )

        # 1. deletions
        for pid in result.deleted_papers:
            self.vectors.delete_paper(pid)
            self.catalog.delete_paper(pid)

        # 2. papers whose content is unchanged but whose copies/paths changed
        for pid in result.kept_papers:
            meta = self.catalog.get_paper(pid)
            paths = sorted(result.paths_by_paper[pid])
            if meta and meta.paths != paths:
                meta.paths, meta.folder_labels = paths, folder_labels(paths)
                meta.filename = Path(paths[0]).name
                self.catalog.update_paper_meta(meta)

        # 3. new / modified content
        if result.new_papers:
            self._process_new(papers_dir, result)

        # 4. file table + keyword index
        self.catalog.delete_files(set(self.catalog.files()) - set(result.files))
        indexed = self.catalog.paper_ids()
        for rel, (size, mtime, sha) in result.files.items():
            if paper_id_for(sha) in indexed:
                self.catalog.upsert_file(rel, size, mtime, sha)
        self._set(state="indexing", message="Building BM25 keyword index")
        self.bm25.build(self.catalog)
        self.catalog.set_kv("embedding_model", self.vectors.embedding_model)
        self._set(chunks_indexed=self.catalog.count_chunks())

    def _process_new(self, papers_dir: Path, result: ScanResult) -> None:
        s = self.settings
        opts = {
            "ocr_enabled": s.ocr_enabled,
            "ocr_min_chars": s.ocr_min_chars_per_page,
            "ocr_dpi": s.ocr_dpi,
            "ocr_lang": s.ocr_language,
            "target_words": s.chunk_target_words,
            "overlap_words": s.chunk_overlap_words,
            "min_words": s.chunk_min_words,
            "timeout_s": s.ingest_timeout_s,
        }
        crossref = (
            CrossrefClient(s.data_dir / "cache" / "crossref.sqlite", s.crossref_mailto)
            if s.crossref_enabled
            else None
        )
        embedder = self._embedder_factory()
        # Embedding, not parsing, is the bottleneck; a few parser processes keep it fed
        # without starving the embedding model of CPU.
        workers = s.ingest_workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        self._set(state="parsing", message=f"Parsing {len(result.new_papers)} PDFs with {workers} workers")

        pending: list[tuple[PaperMeta, list[Chunk]]] = []
        pending_chunks = 0

        def flush():
            nonlocal pending_chunks
            if not pending:
                return
            self._set(state="embedding", message=f"Embedding {pending_chunks} chunks")
            self._store(pending, embedder)
            pending.clear()
            pending_chunks = 0
            self._set(state="parsing", chunks_indexed=self.catalog.count_chunks())

        def completed(pool: ProcessPoolExecutor):
            """Yield finished parse jobs, keeping a bounded number in flight so parsed
            documents never pile up in memory while embedding catches up."""
            queue = sorted(result.new_papers)
            in_flight: dict = {}
            max_in_flight = workers * 2
            while queue or in_flight:
                while queue and len(in_flight) < max_in_flight:
                    pid = queue.pop(0)
                    rel_paths = result.paths_by_paper[pid]
                    abs_path = str(papers_dir / sorted(rel_paths)[0])
                    in_flight[pool.submit(process_pdf, abs_path, pid, rel_paths, opts)] = (pid, rel_paths)
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    yield fut, in_flight.pop(fut)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut, (pid, rel_paths) in completed(pool):
                self.status.processed_files += 1
                try:
                    out = fut.result()
                except Exception as exc:
                    log.warning("parse_failed", extra={"paper": rel_paths[0], "error": repr(exc)})
                    self.status.failed.append(f"{rel_paths[0]}: {exc!r}"[:300])
                    continue
                meta = PaperMeta.model_validate(out["meta"])
                chunks = [Chunk.model_validate(c) for c in out["chunks"]]
                if not chunks:
                    self.status.failed.append(f"{rel_paths[0]}: no extractable text")
                    continue
                if crossref and meta.doi:
                    _apply_crossref(meta, crossref.lookup(meta.doi))
                pending.append((meta, chunks))
                pending_chunks += len(chunks)
                if pending_chunks >= EMBED_BATCH_CHUNKS:
                    flush()
            flush()

    def _store(self, items: list[tuple[PaperMeta, list[Chunk]]], embedder: Embedder) -> None:
        texts, ids, metas = [], [], []
        for meta, chunks in items:
            for c in chunks:
                texts.append(embedding_text(c, meta.title))
                ids.append(c.chunk_id)
                metas.append(
                    {
                        "paper_id": c.paper_id,
                        "chunk_type": c.chunk_type.value,
                        "section_category": c.section_category,
                        "page": c.page_start,
                    }
                )
        vectors = embedder.embed_documents(texts)
        self.vectors.upsert(ids, vectors, metas)
        # Catalog rows are written only after vectors exist, so an interrupted run
        # leaves papers "not indexed" and they are retried next time.
        offset = 0
        now = time.time()
        for meta, chunks in items:
            n = len(chunks)
            self.catalog.upsert_paper(meta, chunks, texts[offset : offset + n], now)
            offset += n


def _apply_crossref(meta: PaperMeta, data: dict | None) -> None:
    if not data:
        return
    if data.get("title"):
        meta.title = data["title"]
        meta.metadata_source = "crossref"
    if data.get("authors"):
        meta.authors = data["authors"]
    if data.get("year"):
        meta.year = data["year"]
    if data.get("journal"):
        meta.journal = data["journal"]
