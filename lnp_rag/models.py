"""Shared data models for ingestion, retrieval and the HTTP API."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    PARAGRAPH = "paragraph"
    FIGURE_CAPTION = "figure_caption"
    TABLE_CAPTION = "table_caption"


class StudyType(str, Enum):
    IN_VITRO = "in vitro"
    IN_VIVO = "in vivo"
    BOTH = "both"
    UNKNOWN = "unknown"


class ChunkFeatures(BaseModel):
    """Evidence signals detected in a chunk's text by ``ingest.features``."""

    has_quantitative: bool = False
    n_timepoints: int = 0
    timepoints: list[str] = Field(default_factory=list)
    has_dose: bool = False
    has_uptake_terms: bool = False
    mentions_in_vitro: bool = False
    mentions_in_vivo: bool = False


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str  # cleaned text shown to users and given to the LLM
    chunk_type: ChunkType
    section: str
    section_category: str  # normalised: abstract/introduction/methods/results/discussion/...
    page_start: int  # 1-indexed
    page_end: int
    label: str = ""  # e.g. "Figure 3" for captions
    ordinal: int = 0  # position within the paper
    features: ChunkFeatures = Field(default_factory=ChunkFeatures)


class PaperMeta(BaseModel):
    paper_id: str  # sha256 prefix of file content: stable across renames/copies
    title: str
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    year: int | None = None
    journal: str | None = None
    abstract: str = ""
    filename: str
    paths: list[str] = Field(default_factory=list)  # relative to papers_dir
    folder_labels: list[str] = Field(default_factory=list)
    n_pages: int = 0
    n_figures: int = 0
    n_tables: int = 0
    has_quantitative: bool = False
    study_type: StudyType = StudyType.UNKNOWN
    is_review: bool = False
    ocr_pages: int = 0
    metadata_source: str = "pdf"  # pdf | heuristic | crossref


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class SearchFilters(BaseModel):
    in_vitro: bool | None = None  # True = require in vitro evidence
    in_vivo: bool | None = None
    year_min: int | None = None
    year_max: int | None = None
    title_contains: str | None = None
    has_figures: bool | None = None
    has_quantitative: bool | None = None
    exclude_reviews: bool = False
    folder: str | None = None  # restrict to a sub-folder label


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    synthesize: bool = True  # call the LLM for grounded explanations


class Evidence(BaseModel):
    evidence_id: str  # e.g. "P1-E2", referenced by explanations
    chunk_id: str
    text: str
    chunk_type: ChunkType
    section: str
    page: int
    label: str = ""
    score: float
    rerank_score: float
    features: ChunkFeatures
    pdf_url: str


class PaperResult(BaseModel):
    rank: int
    paper_id: str
    title: str
    authors: list[str]
    filename: str
    doi: str | None
    year: int | None
    relevance_score: float
    retrieval_score: float
    pages: list[int]
    sections: list[str]
    study_type: StudyType
    study_type_basis: str  # what text the label was derived from
    is_review: bool
    explanation: str
    explanation_source: str  # "openai" | "claude" | "extractive"
    evidence_strength: str  # strong | partial | insufficient
    missing_evidence: str = ""
    cited_evidence_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence]
    captions: list[Evidence]
    pdf_url: str


class SearchResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    total_candidates: int
    results: list[PaperResult]
    timings_ms: dict[str, float]
    warnings: list[str] = Field(default_factory=list)


class IndexRequest(BaseModel):
    papers_dir: str | None = None
    rebuild: bool = False


class IndexStatus(BaseModel):
    state: str = "idle"  # idle | scanning | parsing | embedding | indexing | done | error
    message: str = ""
    total_files: int = 0
    processed_files: int = 0
    new: int = 0
    modified: int = 0
    deleted: int = 0
    unchanged: int = 0
    failed: list[str] = Field(default_factory=list)
    chunks_indexed: int = 0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def progress(self) -> float:
        return self.processed_files / self.total_files if self.total_files else 0.0
