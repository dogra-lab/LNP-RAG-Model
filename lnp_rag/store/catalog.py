"""SQLite catalog: source of truth for files, papers and chunk text/metadata.

The vector store and BM25 index are derived from this catalog and can always be
rebuilt from it. Change detection for re-indexing also lives here.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

from lnp_rag.models import Chunk, PaperMeta

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,      -- relative to papers_dir
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    meta TEXT NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    data TEXT NOT NULL,
    embed_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_paper ON chunks(paper_id);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Catalog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    # --- key/value -------------------------------------------------------------
    def get_kv(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_kv(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, value))

    # --- files -----------------------------------------------------------------
    def files(self) -> dict[str, tuple[int, float, str]]:
        rows = self._conn.execute("SELECT path, size, mtime, sha256 FROM files").fetchall()
        return {p: (s, m, h) for p, s, m, h in rows}

    def upsert_file(self, path: str, size: int, mtime: float, sha256: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)", (path, size, mtime, sha256)
            )

    def delete_files(self, paths: Iterable[str]) -> None:
        with self._lock, self._conn:
            self._conn.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in paths])

    # --- papers ----------------------------------------------------------------
    def upsert_paper(self, meta: PaperMeta, chunks: list[Chunk], embed_texts: list[str], ts: float) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE paper_id = ?", (meta.paper_id,))
            self._conn.execute(
                "INSERT OR REPLACE INTO papers VALUES (?, ?, ?)",
                (meta.paper_id, meta.model_dump_json(), ts),
            )
            self._conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (c.chunk_id, c.paper_id, c.ordinal, c.model_dump_json(), t)
                    for c, t in zip(chunks, embed_texts)
                ],
            )

    def update_paper_meta(self, meta: PaperMeta) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE papers SET meta = ? WHERE paper_id = ?", (meta.model_dump_json(), meta.paper_id)
            )

    def delete_paper(self, paper_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
            self._conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))

    def paper_ids(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT paper_id FROM papers")}

    def get_paper(self, paper_id: str) -> PaperMeta | None:
        row = self._conn.execute("SELECT meta FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return PaperMeta.model_validate_json(row[0]) if row else None

    def all_papers(self) -> dict[str, PaperMeta]:
        rows = self._conn.execute("SELECT paper_id, meta FROM papers").fetchall()
        return {pid: PaperMeta.model_validate_json(m) for pid, m in rows}

    # --- chunks ----------------------------------------------------------------
    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        out: dict[str, Chunk] = {}
        for i in range(0, len(chunk_ids), 500):
            batch = chunk_ids[i : i + 500]
            q = f"SELECT chunk_id, data FROM chunks WHERE chunk_id IN ({','.join('?' * len(batch))})"
            for cid, data in self._conn.execute(q, batch):
                out[cid] = Chunk.model_validate_json(data)
        return out

    def chunks_for_paper(self, paper_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT data FROM chunks WHERE paper_id = ? ORDER BY ordinal", (paper_id,)
        ).fetchall()
        return [Chunk.model_validate_json(r[0]) for r in rows]

    def iter_embed_texts(self) -> Iterator[tuple[str, str, str]]:
        """(chunk_id, paper_id, embed_text) in a stable order."""
        yield from self._conn.execute(
            "SELECT chunk_id, paper_id, embed_text FROM chunks ORDER BY paper_id, ordinal"
        )

    def count_chunks(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def clear(self) -> None:
        with self._lock, self._conn:
            for table in ("files", "papers", "chunks", "kv"):
                self._conn.execute(f"DELETE FROM {table}")

    def stats(self) -> dict:
        papers = self.all_papers()
        return {
            "papers": len(papers),
            "chunks": self.count_chunks(),
            "files": len(self.files()),
            "reviews": sum(p.is_review for p in papers.values()),
            "study_types": _count(p.study_type.value for p in papers.values()),
            "folders": _count(lbl for p in papers.values() for lbl in p.folder_labels),
            "embedding_model": self.get_kv("embedding_model"),
        }


def _count(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
