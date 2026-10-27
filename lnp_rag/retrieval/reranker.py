"""Cross-encoder reranking (default BAAI/bge-reranker-base, runs locally).

A cross-encoder reads the query and the passage together, so it judges
relevance far better than embedding similarity, at the cost of one forward pass
per candidate - which is why it only sees the fused top-N candidates.
"""
from __future__ import annotations

import logging

import numpy as np

from lnp_rag.embeddings import resolve_device

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model: str, device: str = "auto", batch_size: int = 32):
        import torch
        from sentence_transformers import CrossEncoder

        self.name = model
        self._identity = torch.nn.Identity()  # get raw logits; we apply the sigmoid
        self._model = CrossEncoder(model, device=resolve_device(device), max_length=512)
        self._batch = batch_size

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance probabilities in [0, 1], one per passage."""
        if not passages:
            return np.array([], dtype=np.float32)
        logits = self._model.predict(
            [(query, p) for p in passages],
            batch_size=self._batch,
            show_progress_bar=False,
            activation_fn=self._identity,
        )
        return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
