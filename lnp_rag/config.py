"""Application configuration.

Two layers:

* ``Settings`` - deployment settings (paths, providers, model names, API keys).
  Read from environment variables / ``.env``. Secrets live only here.
* ``RetrievalConfig`` - retrieval tuning knobs (candidate sizes, fusion weights,
  evidence boosts). Read from ``config/retrieval.yaml`` so they can be tuned and
  version-controlled without touching code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- paths ---------------------------------------------------------------
    papers_dir: Path = PROJECT_ROOT / "papers"
    data_dir: Path = PROJECT_ROOT / "data"
    retrieval_config_path: Path = PROJECT_ROOT / "config" / "retrieval.yaml"
    synonyms_path: Path = PROJECT_ROOT / "config" / "synonyms.yaml"

    # --- ingestion -----------------------------------------------------------
    ocr_enabled: bool = True
    ocr_min_chars_per_page: int = 80  # pages with less extractable text get OCR'd
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    ingest_workers: int = 0  # 0 = min(4, cpu_count - 1)
    ingest_timeout_s: int = 300  # per-PDF limit; a file that exceeds it is reported as failed
    chunk_target_words: int = 220
    chunk_overlap_words: int = 50
    chunk_min_words: int = 25
    crossref_enabled: bool = False  # look up title/authors/year by DOI (network)
    crossref_mailto: str = ""  # polite-pool contact for Crossref

    # --- embeddings ----------------------------------------------------------
    embedding_provider: Literal["sentence_transformers", "medcpt", "openai"] = (
        "sentence_transformers"
    )
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    # Instruction prepended to queries (BGE v1.5 recommends this for retrieval).
    embedding_query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )
    embedding_batch_size: int = 64
    embedding_device: str = "auto"  # auto | cpu | cuda | mps
    openai_api_key: SecretStr | None = None

    # --- reranking -----------------------------------------------------------
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = 32

    # --- LLM (evidence synthesis + optional query paraphrasing) ----------------
    # auto = Claude if ANTHROPIC_API_KEY is set, else OpenAI if OPENAI_API_KEY is set.
    llm_provider: Literal["auto", "openai", "anthropic"] = "auto"
    llm_max_retries: int = 3
    llm_timeout_s: float = 180.0
    openai_model: str = "gpt-5.4-mini"
    openai_reasoning_effort: str = ""  # e.g. "low"/"medium" for reasoning models; empty = model default
    anthropic_api_key: SecretStr | None = None
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    claude_max_retries: int = 3
    claude_timeout_s: float = 180.0
    # "default" opts into server-side refusal fallbacks; "" disables them.
    claude_fallbacks: str = "default"
    synthesis_enabled: bool = True
    # "llm" adds paraphrases from the configured LLM ("claude" is accepted as an alias).
    query_expansion_mode: Literal["none", "rules", "llm", "claude"] = "rules"

    # --- service -------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Public URL of the API, used to build "open PDF" links in the UI.
    api_public_url: str = "http://localhost:8000"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    # --- derived paths -------------------------------------------------------
    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.sqlite"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def bm25_dir(self) -> Path:
        return self.data_dir / "bm25"

    @property
    def embedding_cache_path(self) -> Path:
        return self.data_dir / "cache" / "embeddings.sqlite"

    @property
    def uploads_dir(self) -> Path:
        return self.papers_dir / "uploads"

    def anthropic_key(self) -> str | None:
        return self.anthropic_api_key.get_secret_value() if self.anthropic_api_key else None


class EvidenceBoosts(BaseModel):
    """Multiplicative boosts applied to a chunk's reranker score.

    ``base`` boosts always apply. ``intent`` boosts apply only when the query
    signals that intent (e.g. it mentions time courses), so that a query about
    formulation chemistry is not dragged toward kinetic-uptake figures.
    """

    figure_caption: float = 0.15
    table_caption: float = 0.12
    results_section: float = 0.10
    methods_section: float = 0.04
    abstract: float = 0.02
    quantitative: float = 0.06
    in_vitro_or_vivo_label: float = 0.03
    # intent-gated
    intent_quantitative: float = 0.10
    intent_multi_timepoint: float = 0.15
    intent_dose: float = 0.08
    intent_figure: float = 0.08
    # paper-level multiplier for review articles (<1 demotes)
    review_paper_multiplier: float = 0.80


class RetrievalConfig(BaseModel):
    dense_top_k: int = 120  # per query variant
    bm25_top_k: int = 120  # per query variant
    rrf_k: int = 60
    original_query_weight: float = 1.0
    expanded_query_weight: float = 0.6
    dense_weight: float = 1.0
    bm25_weight: float = 0.8
    rerank_candidates: int = 150  # chunks sent to the cross-encoder
    max_evidence_per_paper: int = 5
    # paper score = sum_i chunk_score_i * decay^i over the paper's best chunks
    paper_aggregation_decay: float = 0.5
    paper_aggregation_top_n: int = 4
    evidence_type_coverage_bonus: float = 0.05  # per distinct evidence type, capped
    min_chunk_relevance: float = 0.05  # drop chunks the reranker considers irrelevant
    synthesis_top_papers: int = 10
    synthesis_score_weight: float = 0.25  # blend of LLM relevance into final score
    boosts: EvidenceBoosts = Field(default_factory=EvidenceBoosts)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_retrieval_config() -> RetrievalConfig:
    path = get_settings().retrieval_config_path
    if path.exists():
        return RetrievalConfig(**(yaml.safe_load(path.read_text()) or {}))
    return RetrievalConfig()
