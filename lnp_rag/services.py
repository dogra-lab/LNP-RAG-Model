"""Lazily-constructed shared components (models are loaded once per process)."""
from __future__ import annotations

import logging
import os
import threading
from functools import cached_property

from lnp_rag.agents.orchestrator import AgentOrchestrator
from lnp_rag.config import RetrievalConfig, Settings, get_retrieval_config, get_settings
from lnp_rag.embeddings import Embedder, build_embedder
from lnp_rag.ingest.pipeline import Indexer
from lnp_rag.retrieval.query_expansion import QueryExpander
from lnp_rag.retrieval.reranker import CrossEncoderReranker
from lnp_rag.retrieval.searcher import Searcher
from lnp_rag.store.bm25_index import BM25Index
from lnp_rag.store.catalog import Catalog
from lnp_rag.store.vector_store import VectorStore
from lnp_rag.synthesis.prompts import LLMClient

log = logging.getLogger(__name__)


class Services:
    def __init__(self, settings: Settings | None = None, cfg: RetrievalConfig | None = None):
        self.settings = settings or get_settings()
        self.cfg = cfg or get_retrieval_config()
        self._lock = threading.Lock()

    @cached_property
    def catalog(self) -> Catalog:
        return Catalog(self.settings.catalog_path)

    @cached_property
    def vectors(self) -> VectorStore:
        return VectorStore(self.settings.chroma_dir, self.embedding_model_id)

    @cached_property
    def bm25(self) -> BM25Index:
        return BM25Index(self.settings.bm25_dir)

    @property
    def embedding_model_id(self) -> str:
        s = self.settings
        return "ncbi/MedCPT" if s.embedding_provider == "medcpt" else s.embedding_model

    @cached_property
    def embedder(self) -> Embedder:
        with self._lock:
            return build_embedder(self.settings)

    @cached_property
    def reranker(self) -> CrossEncoderReranker | None:
        if not self.settings.reranker_enabled:
            return None
        return CrossEncoderReranker(
            self.settings.reranker_model, self.settings.embedding_device, self.settings.reranker_batch_size
        )

    @property
    def llm_provider(self) -> str | None:
        """Which LLM provider will be used, or None if no key is configured."""
        s = self.settings
        has_anthropic = bool(s.anthropic_key() or os.environ.get("ANTHROPIC_API_KEY")
                             or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        has_openai = bool(s.openai_api_key and s.openai_api_key.get_secret_value())
        if s.llm_provider == "anthropic":
            return "anthropic" if has_anthropic else None
        if s.llm_provider == "openai":
            return "openai" if has_openai else None
        return "anthropic" if has_anthropic else "openai" if has_openai else None

    @cached_property
    def llm(self) -> LLMClient | None:
        s = self.settings
        if not s.synthesis_enabled and s.query_expansion_mode not in ("llm", "claude"):
            return None
        return self._build_llm()

    @cached_property
    def critic_llm(self) -> LLMClient | None:
        """LLM for the folder critic agents (used even when synthesis is disabled)."""
        return self.__dict__.get("llm") or self._build_llm()

    def _build_llm(self) -> LLMClient | None:
        s = self.settings
        provider = self.llm_provider
        if provider == "openai":
            from lnp_rag.synthesis.openai_llm import OpenAIClient

            client: LLMClient = OpenAIClient(s)
        elif provider == "anthropic":
            from lnp_rag.synthesis.claude import ClaudeClient

            client = ClaudeClient(s)
        else:
            log.warning("llm_disabled", extra={"reason": "no OPENAI_API_KEY or ANTHROPIC_API_KEY"})
            return None
        log.info("llm_ready", extra={"provider": client.name, "model": client.model})
        return client

    @cached_property
    def expander(self) -> QueryExpander:
        return QueryExpander(self.settings.query_expansion_mode, self.settings.synonyms_path, self.llm)

    @cached_property
    def agents(self) -> AgentOrchestrator:
        return AgentOrchestrator(self.searcher, self.critic_llm)

    @cached_property
    def searcher(self) -> Searcher:
        return Searcher(
            self.settings,
            self.cfg,
            self.catalog,
            self.vectors,
            self.bm25,
            self.embedder,
            self.reranker,
            self.expander,
            self.llm if self.settings.synthesis_enabled else None,
        )

    def prepare_rebuild(self) -> None:
        """Drop the vector collection (even if built with another embedding model) and
        forget components that hold a reference to it."""
        for name in ("vectors", "searcher", "agents", "indexer"):
            self.__dict__.pop(name, None)
        self.__dict__["vectors"] = VectorStore(self.settings.chroma_dir, self.embedding_model_id, reset=True)
        self.catalog.clear()

    @cached_property
    def indexer(self) -> Indexer:
        return Indexer(self.settings, self.catalog, self.vectors, self.bm25, lambda: self.embedder)
