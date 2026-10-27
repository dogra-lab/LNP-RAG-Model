"""Embedding providers (configurable via EMBEDDING_PROVIDER / EMBEDDING_MODEL).

* ``sentence_transformers`` - any SentenceTransformers model; default
  ``BAAI/bge-base-en-v1.5`` (local, 768-d, 512-token context).
* ``medcpt`` - NCBI MedCPT asymmetric query/article encoders trained on PubMed
  search logs (local, biomedical-domain).
* ``openai`` - ``text-embedding-3-large`` etc. via the OpenAI REST API (paid,
  8k-token context, no local GPU needed).

All providers return L2-normalised float32 vectors, and document embeddings are
cached on disk keyed by (model, text) so rebuilds do not re-embed.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from lnp_rag.config import Settings

log = logging.getLogger(__name__)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder(ABC):
    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str, device: str, batch_size: int, query_instruction: str = ""):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._model = SentenceTransformer(model, device=resolve_device(device))
        self._model.max_seq_length = min(self._model.max_seq_length or 512, 512)
        self._batch = batch_size
        self._instruction = query_instruction if "bge" in model.lower() or "e5" in model.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(
            self._model.encode(texts, batch_size=self._batch, normalize_embeddings=True, show_progress_bar=False)
        )

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._model.encode([self._instruction + text], normalize_embeddings=True))[0]


class MedCPTEmbedder(Embedder):
    """MedCPT: separate encoders for queries and (title, text) articles; CLS pooling."""

    def __init__(self, device: str, batch_size: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = "ncbi/MedCPT"
        self._torch = torch
        self._device = resolve_device(device)
        self._q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self._q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(self._device).eval()
        self._a_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self._a_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(self._device).eval()
        self._batch = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i : i + self._batch]
            pairs = [t.split("\n", 1) if "\n" in t else ["", t] for t in batch]
            with self._torch.no_grad():
                enc = self._a_tok(pairs, truncation=True, padding=True, max_length=512, return_tensors="pt")
                enc = {k: v.to(self._device) for k, v in enc.items()}
                out.append(self._a_model(**enc).last_hidden_state[:, 0, :].cpu().numpy())
        return _normalize(np.vstack(out))

    def embed_query(self, text: str) -> np.ndarray:
        with self._torch.no_grad():
            enc = self._q_tok([text], truncation=True, padding=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(self._device) for k, v in enc.items()}
            return _normalize(self._q_model(**enc).last_hidden_state[:, 0, :].cpu().numpy())[0]


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, api_key: str, batch_size: int = 128):
        self.name = model
        self._batch = max(1, min(batch_size, 512))
        self._http = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _call(self, inputs: list[str]) -> np.ndarray:
        r = self._http.post("/embeddings", json={"model": self.name, "input": inputs})
        if r.status_code == 429 or r.status_code >= 500:
            r.raise_for_status()
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI embeddings error {r.status_code}: {r.text[:300]}")
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return np.array([d["embedding"] for d in data], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.vstack([self._call(texts[i : i + self._batch]) for i in range(0, len(texts), self._batch)]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._call([text]))[0]


class CachedEmbedder(Embedder):
    """Wraps an embedder with a persistent (model, text) -> vector cache."""

    def __init__(self, inner: Embedder, cache_path: Path):
        self.inner = inner
        self.name = inner.name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self._lock = threading.Lock()
        self._query_cache = lru_cache(maxsize=512)(self.inner.embed_query)

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.name}\x00{text}".encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 900):
                batch = keys[i : i + 900]
                q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
                for k, blob in self._db.execute(q, batch):
                    found[k] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            vecs = self.inner.embed_documents([texts[i] for i in missing])
            with self._lock, self._db:
                self._db.executemany(
                    "INSERT OR REPLACE INTO emb VALUES (?, ?)",
                    [(keys[i], v.astype(np.float16).tobytes()) for i, v in zip(missing, vecs)],
                )
            for i, v in zip(missing, vecs):
                found[keys[i]] = v
        return _normalize(np.vstack([found[k] for k in keys]))

    def embed_query(self, text: str) -> np.ndarray:
        return self._query_cache(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "medcpt":
        inner: Embedder = MedCPTEmbedder(settings.embedding_device, settings.embedding_batch_size)
    elif settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        inner = OpenAIEmbedder(settings.embedding_model, settings.openai_api_key.get_secret_value(),
                               settings.embedding_batch_size)
    else:
        inner = SentenceTransformerEmbedder(
            settings.embedding_model,
            settings.embedding_device,
            settings.embedding_batch_size,
            settings.embedding_query_instruction,
        )
    log.info("embedder_ready", extra={"provider": settings.embedding_provider, "model": inner.name})
    return CachedEmbedder(inner, settings.embedding_cache_path)
