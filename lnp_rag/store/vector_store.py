"""Persistent vector store backed by ChromaDB (embedded mode).

Chroma runs in-process for local use; for deployment the same code can point at
a Chroma server by swapping ``PersistentClient`` for ``HttpClient``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
import numpy as np

log = logging.getLogger(__name__)

COLLECTION = "paper_chunks"


class EmbeddingModelMismatch(RuntimeError):
    pass


class VectorStore:
    def __init__(self, path: Path, embedding_model: str, reset: bool = False):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self.embedding_model = embedding_model
        if reset and COLLECTION in [c.name for c in self._client.list_collections()]:
            self._client.delete_collection(COLLECTION)
        self._col = self._open()
        stored = (self._col.metadata or {}).get("embedding_model")
        if stored and stored != embedding_model and self._col.count() > 0:
            raise EmbeddingModelMismatch(
                f"Vector index was built with '{stored}' but EMBEDDING_MODEL is "
                f"'{embedding_model}'. Rebuild the index (scripts/index_papers.py --rebuild)."
            )
        self._max_batch = self._client.get_max_batch_size()

    def count(self) -> int:
        return self._col.count()

    def upsert(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict]) -> None:
        for i in range(0, len(ids), self._max_batch):
            self._col.upsert(
                ids=ids[i : i + self._max_batch],
                embeddings=embeddings[i : i + self._max_batch].tolist(),
                metadatas=metadatas[i : i + self._max_batch],
            )

    def delete_paper(self, paper_id: str) -> None:
        self._col.delete(where={"paper_id": paper_id})

    def query(
        self, embedding: np.ndarray, k: int, paper_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine_similarity)] best first."""
        n = self._col.count()
        if n == 0:
            return []
        where = {"paper_id": {"$in": paper_ids}} if paper_ids is not None else None
        if paper_ids is not None and not paper_ids:
            return []
        res = self._col.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(k, n),
            where=where,
            include=["distances"],
        )
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def get_vectors(self, chunk_ids: list[str], batch: int = 5000) -> np.ndarray:
        """Stored embeddings for ``chunk_ids`` in the same order (zeros if missing)."""
        out: np.ndarray | None = None
        for i in range(0, len(chunk_ids), batch):
            ids = chunk_ids[i : i + batch]
            res = self._col.get(ids=ids, include=["embeddings"])
            pos = {cid: j for j, cid in enumerate(ids)}
            for cid, emb in zip(res["ids"], res["embeddings"]):
                if out is None:
                    out = np.zeros((len(chunk_ids), len(emb)), dtype=np.float32)
                out[i + pos[cid]] = emb
        return out if out is not None else np.zeros((len(chunk_ids), 0), dtype=np.float32)

    def _open(self):
        return self._client.get_or_create_collection(
            COLLECTION,
            configuration={"hnsw": {"space": "cosine", "ef_construction": 200, "ef_search": 200}},
            metadata={"embedding_model": self.embedding_model},
            embedding_function=None,
        )

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._col = self._open()
