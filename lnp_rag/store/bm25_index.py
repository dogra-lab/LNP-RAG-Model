"""BM25 keyword index (bm25s: sparse-matrix BM25, fast and persistable).

The tokenizer is tuned for scientific text: hyphenated lipid / drug names
("DLin-MC3-DMA", "SM-102") are kept whole *and* split into parts, numbers with
units stay searchable ("24h" -> "24", "h"), and plain words are stemmed.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from lnp_rag.store.catalog import Catalog

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9α-ω]+(?:[-'/.][a-z0-9α-ω]+)*")
_STOP = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how in into "
    "is it its may more most no not of on or other our such than that the their them then there "
    "these they this those to was we were what when where which while who will with within would "
    "also both each however using used use showed shown study studies paper papers find research "
    "including include especially particularly containing contain".split()
)
_stemmer = Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        tok = tok.strip(".'/")
        if not tok or tok in _STOP:
            continue
        if re.search(r"[-/.]", tok):
            if not re.fullmatch(r"\d+\.\d+", tok):  # keep decimals as one token
                tokens.append(tok)  # whole compound term
                parts = [p for p in re.split(r"[-/.]", tok) if p and p not in _STOP]
                tokens.extend(_stem(p) for p in parts)
                continue
        tokens.append(_stem(tok))
    return tokens


def _stem(tok: str) -> str:
    if tok.isalpha() and len(tok) > 3:
        return _stemmer.stemWord(tok)
    return tok


class BM25Index:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._model: bm25s.BM25 | None = None
        self._ids: list[str] = []
        self._paper_of: np.ndarray = np.array([], dtype=object)
        self.load()

    def load(self) -> None:
        ids_file = self.path / "ids.json"
        if not ids_file.exists():
            return
        try:
            with self._lock:
                self._model = bm25s.BM25.load(str(self.path), show_progress=False)
                data = json.loads(ids_file.read_text())
                self._ids = data["chunk_ids"]
                self._paper_of = np.array(data["paper_ids"], dtype=object)
        except Exception:
            log.exception("bm25_load_failed")
            self._model = None

    def build(self, catalog: Catalog) -> int:
        """Rebuild from the catalog (seconds for ~100k chunks)."""
        ids, papers, corpus = [], [], []
        for cid, pid, text in catalog.iter_embed_texts():
            ids.append(cid)
            papers.append(pid)
            corpus.append(tokenize(text))
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)
            if not ids:
                self._model, self._ids, self._paper_of = None, [], np.array([], dtype=object)
                (self.path / "ids.json").unlink(missing_ok=True)
                return 0
            model = bm25s.BM25(k1=1.2, b=0.75)
            model.index(corpus, show_progress=False)
            model.save(str(self.path), show_progress=False)
            (self.path / "ids.json").write_text(json.dumps({"chunk_ids": ids, "paper_ids": papers}))
            self._model, self._ids = model, ids
            self._paper_of = np.array(papers, dtype=object)
        return len(ids)

    def search(self, query: str, k: int, paper_ids: set[str] | None = None) -> list[tuple[str, float]]:
        with self._lock:
            if self._model is None or not self._ids:
                return []
            tokens = [t for t in tokenize(query) if t in self._model.vocab_dict]
            if not tokens:
                return []
            mask = None
            if paper_ids is not None:
                if not paper_ids:
                    return []
                mask = np.isin(self._paper_of, list(paper_ids)).astype(np.float32)
            docs, scores = self._model.retrieve(
                [tokens], k=min(k, len(self._ids)), show_progress=False, weight_mask=mask
            )
        return [
            (self._ids[int(d)], float(s)) for d, s in zip(docs[0], scores[0]) if s > 0
        ]

    def rows_for_papers(self, paper_ids: set[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(chunk_ids, paper_id per chunk, index rows) of every indexed chunk of ``paper_ids``."""
        with self._lock:
            rows = np.flatnonzero(np.isin(self._paper_of, list(paper_ids))) if self._ids else np.array([], int)
            return [self._ids[int(r)] for r in rows], self._paper_of[rows], rows

    def score_rows(self, query: str, rows: np.ndarray) -> np.ndarray:
        """Exhaustive BM25 scores of ``query`` for the given index rows (no top-k)."""
        with self._lock:
            if self._model is None or not len(rows):
                return np.zeros(len(rows), dtype=np.float32)
            tokens = [t for t in tokenize(query) if t in self._model.vocab_dict]
            if not tokens:
                return np.zeros(len(rows), dtype=np.float32)
            return np.asarray(self._model.get_scores(tokens), dtype=np.float32)[rows]

    def __len__(self) -> int:
        return len(self._ids)
