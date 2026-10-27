"""Structure-aware chunking.

Rules:

* Chunks never cross a section boundary, so every chunk has one section name.
* Figure and table captions are standalone chunks (split only if very long),
  because captions are often where quantitative evidence is summarised.
* The abstract is its own chunk type.
* Long sections are windowed at sentence boundaries with word-level overlap.
* Paragraphs in references / back matter are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from lnp_rag.ingest.cleaning import word_count
from lnp_rag.ingest.features import detect_features
from lnp_rag.ingest.pdf_parser import SKIP_CATEGORIES, Block
from lnp_rag.models import Chunk, ChunkType

_ABBREV = re.compile(r"\b(Fig|Figs|et al|e\.g|i\.e|approx|vs|ca|cf|resp|Ref|Refs|No|Eq|Dr|Inc|Ltd|St)\.", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


def split_sentences(text: str) -> list[str]:
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    # Decimal numbers never end sentences: "0.5 h" is protected by the lookahead
    # because the split requires whitespace after the period.
    return [s.replace("<DOT>", ".").strip() for s in _SENT_SPLIT.split(protected) if s.strip()]


@dataclass
class _Piece:
    text: str
    page: int


def _window(pieces: list[_Piece], target: int, overlap: int) -> list[tuple[str, int, int]]:
    """Group sentence pieces into ~target-word windows with ~overlap-word overlap."""
    out: list[tuple[str, int, int]] = []
    buf: list[_Piece] = []
    words = 0
    for piece in pieces:
        n = word_count(piece.text)
        if buf and words + n > target:
            out.append((" ".join(p.text for p in buf), buf[0].page, buf[-1].page))
            # carry trailing sentences as overlap
            carry: list[_Piece] = []
            carried = 0
            for p in reversed(buf):
                if carried >= overlap:
                    break
                carry.insert(0, p)
                carried += word_count(p.text)
            if carried >= words:  # a single huge sentence; avoid infinite overlap
                carry = []
                carried = 0
            buf, words = carry, carried
        buf.append(piece)
        words += n
    if buf:
        out.append((" ".join(p.text for p in buf), buf[0].page, buf[-1].page))
    return out


def _sentence_pieces(text: str, page: int, max_words: int) -> list[_Piece]:
    pieces = []
    for sent in split_sentences(text):
        words = sent.split()
        # Hard-wrap run-on "sentences" (tables, lists) so windows stay bounded.
        for i in range(0, len(words), max_words):
            pieces.append(_Piece(" ".join(words[i : i + max_words]), page))
    return pieces


def chunk_document(
    paper_id: str,
    blocks: list[Block],
    target_words: int = 220,
    overlap_words: int = 50,
    min_words: int = 25,
) -> tuple[list[Chunk], str]:
    """Return (chunks, abstract_text)."""
    chunks: list[Chunk] = []
    abstract_parts: list[tuple[str, int]] = []

    def add(text: str, ctype: ChunkType, section: str, category: str, p0: int, p1: int, label: str = ""):
        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}:{len(chunks):04d}",
                paper_id=paper_id,
                text=text,
                chunk_type=ctype,
                section=section,
                section_category=category,
                page_start=p0,
                page_end=p1,
                label=label,
                ordinal=len(chunks),
                features=detect_features(text),
            )
        )

    # Group consecutive paragraph blocks by section.
    group: list[Block] = []

    def flush_group():
        if not group:
            return
        first = group[0]
        pieces: list[_Piece] = []
        for b in group:
            pieces += _sentence_pieces(b.text, b.page, target_words)
        windows = _window(pieces, target_words, overlap_words)
        # Merge a short trailing window into the previous one.
        if len(windows) > 1 and word_count(windows[-1][0]) < min_words:
            tail = windows.pop()
            prev = windows.pop()
            windows.append((prev[0] + " " + tail[0], prev[1], tail[2]))
        for text, p0, p1 in windows:
            if word_count(text) >= min(min_words, 12):
                add(text, ChunkType.PARAGRAPH, first.section, first.category, p0, p1)
        group.clear()

    for b in blocks:
        if b.kind == "heading":
            continue
        if b.kind in ("figure_caption", "table_caption"):
            # Captions interrupt the flow of paragraphs but not the section.
            ctype = ChunkType.FIGURE_CAPTION if b.kind == "figure_caption" else ChunkType.TABLE_CAPTION
            windows = (
                _window(_sentence_pieces(b.text, b.page, target_words), int(target_words * 1.5), overlap_words)
                if word_count(b.text) > target_words * 1.6
                else [(b.text, b.page, b.page)]
            )
            for text, p0, p1 in windows:
                add(text, ctype, b.section, b.category, p0, p1, label=b.label)
            continue
        if b.category in SKIP_CATEGORIES:
            flush_group()
            continue
        if b.category == "abstract":
            flush_group()
            abstract_parts.append((b.text, b.page))
            continue
        if group and (group[-1].section != b.section or group[-1].category != b.category):
            flush_group()
        group.append(b)
    flush_group()

    abstract = " ".join(t for t, _ in abstract_parts).strip()
    if abstract:
        page = abstract_parts[0][1]
        pieces = _sentence_pieces(abstract, page, target_words)
        abstract_chunks = []
        for text, p0, p1 in _window(pieces, int(target_words * 1.5), overlap_words):
            abstract_chunks.append((text, p0, p1))
        # Abstract chunks go first; renumber everything to keep ordinals sequential.
        existing = chunks[:]
        chunks.clear()
        for text, p0, p1 in abstract_chunks:
            add(text, ChunkType.ABSTRACT, "Abstract", "abstract", p0, p1)
        for c in existing:
            c.chunk_id = f"{paper_id}:{len(chunks):04d}"
            c.ordinal = len(chunks)
            chunks.append(c)
    else:
        # No explicit abstract: use the first substantial front/introduction paragraph.
        for c in chunks:
            if c.chunk_type == ChunkType.PARAGRAPH and c.section_category in ("front", "introduction") \
                    and word_count(c.text) >= 60:
                abstract = c.text
                break
    return chunks, abstract


def embedding_text(chunk: Chunk, title: str) -> str:
    """Text actually embedded / BM25-indexed: a contextual header plus the chunk.

    Prefixing title and section lets an isolated paragraph ("uptake increased at
    4 h") carry its paper's topic (e.g. "lipid nanoparticle") into retrieval.
    """
    section = chunk.label or chunk.section
    return f"{title}\n{section}: {chunk.text}"
