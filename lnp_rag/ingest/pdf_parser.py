"""Structure-aware PDF parsing with PyMuPDF.

Produces an ordered list of typed blocks (heading / paragraph / figure caption /
table caption) with page numbers and the section each block belongs to.

Heuristics, in order:

1. Per page, read text blocks with font information (``page.get_text("dict")``).
   Pages with almost no extractable text are OCR'd instead (``ingest.ocr``).
2. Remove running headers/footers and watermarks: lines whose digit-normalised
   text repeats on many pages.
3. Restore reading order on two-column pages (left column, then right column,
   with full-width blocks such as titles and wide figures acting as separators).
4. Classify blocks: captions start with "Figure N." / "Fig. N" / "Table N";
   headings are short lines that are bold or larger than the body font, or that
   match a known section name.
5. Track the current section and map it to a coarse category
   (abstract / introduction / methods / results / discussion / ...).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from lnp_rag.ingest.cleaning import boilerplate_key, is_page_number, join_lines, normalize_text
from lnp_rag.ingest.ocr import ocr_page

log = logging.getLogger(__name__)

# Coarse section categories, matched against heading text (numbering stripped).
SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^(abstract|summary|conspectus|graphical abstract|significance)\b", re.I)),
    ("front", re.compile(r"^(keywords?|key words|classification|in brief|highlights|authors?|correspondence)\b", re.I)),
    ("introduction", re.compile(r"^(introduction|background)\b", re.I)),
    (
        "methods",
        re.compile(
            r"^(materials? and methods?|methods?|experimental( section| procedures?| methods?)?|"
            r"methodology|star\W*methods|materials?|method details|online methods)\b",
            re.I,
        ),
    ),
    ("results", re.compile(r"^(results( and discussion)?|findings)\b", re.I)),
    ("discussion", re.compile(r"^(discussion|general discussion)\b", re.I)),
    ("conclusion", re.compile(r"^(conclusions?|concluding remarks|outlook|future perspectives?)\b", re.I)),
    (
        "references",
        re.compile(r"^(references|bibliography|literature cited|works cited|notes and references)\b", re.I),
    ),
    (
        "backmatter",
        re.compile(
            r"^(acknowledge?ments?|funding|conflicts? of interest|competing interests?|"
            r"declaration of competing interest|author contributions?|credit authorship|"
            r"data availability|ethics statement|abbreviations|supplementary (material|information)|"
            r"supporting information|disclosure)\b",
            re.I,
        ),
    ),
]
# Paragraphs in these sections are not indexed (captions still are, because many
# author manuscripts place figures after the reference list).
SKIP_CATEGORIES = {"references", "backmatter"}

# Case-insensitive only for the prefix: the "next word is capitalised" test must
# stay case-sensitive so in-text references ("Figure 2 shows...") are not captions.
_FIG_CAPTION = re.compile(
    r"^(?i:supplementary\s+)?((?i:fig(?:ure)?s?\.?|scheme))\s*([Ss]?\d+[A-Za-z]?)"
    r"\s*(?:[.:|\-]|\s{2,}|\s(?=[A-Z(Ͱ-Ͽ]))"
)
_TABLE_CAPTION = re.compile(
    r"^(?i:supplementary\s+)?((?i:table))\s*([Ss]?\d+|[IVX]+)\s*(?:[.:|\-]|\s(?=[A-Z(Ͱ-Ͽ]))"
)
_NUMBERING = re.compile(r"^((\d+(\.\d+)*\.?)|([IVX]+\.)|([A-Z]\.))\s+")
_ABSTRACT_PREFIX = re.compile(r"^(abstract|summary)\s*[:.\-]?\s*", re.I)


@dataclass
class Block:
    text: str
    page: int  # 1-indexed
    kind: str  # heading | paragraph | figure_caption | table_caption
    size: float
    bold: bool
    bbox: tuple[float, float, float, float]
    section: str = ""
    category: str = "body"
    label: str = ""  # "Figure 2" / "Table 1" for captions


@dataclass
class ParsedDocument:
    path: Path
    n_pages: int
    blocks: list[Block]
    pdf_metadata: dict
    first_page_lines: list[tuple[str, float, float]]  # (text, size, y0) on page 1
    early_text: str  # raw text of the first two pages (DOI/year detection)
    body_size: float
    ocr_pages: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _RawBlock:
    lines: list[tuple[str, float, bool]]  # (text, size, bold)
    bbox: tuple[float, float, float, float]
    page: int

    @property
    def text(self) -> str:
        return " ".join(t for t, _, _ in self.lines)

    @property
    def size(self) -> float:
        weights = Counter()
        for t, s, _ in self.lines:
            weights[round(s, 1)] += len(t)
        return weights.most_common(1)[0][0] if weights else 0.0

    @property
    def bold(self) -> bool:
        chars = sum(len(t) for t, _, _ in self.lines)
        bold_chars = sum(len(t) for t, _, b in self.lines if b)
        return chars > 0 and bold_chars / chars > 0.6


def section_category(heading: str) -> str | None:
    clean = _NUMBERING.sub("", heading.strip()).strip(" .:")
    if re.fullmatch(r"(\w\s)+\w", clean):  # letter-spaced, e.g. "a b s t r a c t"
        clean = clean.replace(" ", "")
    for category, pattern in SECTION_PATTERNS:
        if pattern.match(clean):
            return category
    return None


def caption_label(text: str) -> tuple[str, str] | None:
    """Return (kind, label) if ``text`` starts like a figure/table caption."""
    if m := _FIG_CAPTION.match(text):
        prefix = "Scheme" if m.group(1).lower().startswith("scheme") else "Figure"
        return "figure_caption", f"{prefix} {m.group(2)}"
    if m := _TABLE_CAPTION.match(text):
        return "table_caption", f"Table {m.group(2)}"
    return None


def _span_is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or bool(re.search(r"bold|heavy|black|semibold", span["font"], re.I))


def _read_page_blocks(page: pymupdf.Page, page_no: int) -> list[_RawBlock]:
    blocks: list[_RawBlock] = []
    data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_DEHYPHENATE)
    for b in data["blocks"]:
        if b.get("type") != 0:
            continue
        lines = []
        for line in b["lines"]:
            # Keep whitespace-only spans when joining: some PDFs encode every
            # inter-word space as its own span.
            text = normalize_text("".join(s["text"] for s in line["spans"]))
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            if not text:
                continue
            size = max(spans, key=lambda s: len(s["text"]))["size"]
            bold_chars = sum(len(s["text"]) for s in spans if _span_is_bold(s))
            lines.append((text, size, bold_chars > 0.6 * sum(len(s["text"]) for s in spans)))
        if lines:
            blocks.append(_RawBlock(lines=lines, bbox=tuple(b["bbox"]), page=page_no))
    return blocks


def _reading_order(blocks: list[_RawBlock], page_width: float) -> list[_RawBlock]:
    """Order blocks column-wise on two-column pages."""
    mid = page_width / 2
    tol = page_width * 0.04

    def column(b: _RawBlock) -> str:
        x0, _, x1, _ = b.bbox
        if x1 <= mid + tol:
            return "L"
        if x0 >= mid - tol:
            return "R"
        return "F"

    cols = [column(b) for b in blocks]
    left_chars = sum(len(b.text) for b, c in zip(blocks, cols) if c == "L")
    right_chars = sum(len(b.text) for b, c in zip(blocks, cols) if c == "R")
    if min(left_chars, right_chars) < 200:  # single-column page
        return sorted(blocks, key=lambda b: (round(b.bbox[1]), b.bbox[0]))

    ordered: list[_RawBlock] = []
    left: list[_RawBlock] = []
    right: list[_RawBlock] = []
    for b, c in sorted(zip(blocks, cols), key=lambda bc: bc[0].bbox[1]):
        if c == "F":
            ordered += sorted(left, key=lambda x: x.bbox[1]) + sorted(right, key=lambda x: x.bbox[1])
            left, right = [], []
            ordered.append(b)
        else:
            (left if c == "L" else right).append(b)
    ordered += sorted(left, key=lambda x: x.bbox[1]) + sorted(right, key=lambda x: x.bbox[1])
    return ordered


def _find_boilerplate(pages: list[list[_RawBlock]], heights: list[float]) -> set[str]:
    """Keys of lines repeated across pages near the top/bottom edge (or anywhere, if very frequent)."""
    edge_counts: Counter[str] = Counter()
    any_counts: Counter[str] = Counter()
    for blocks, height in zip(pages, heights):
        seen_edge, seen_any = set(), set()
        for b in blocks:
            near_edge = b.bbox[1] < height * 0.09 or b.bbox[3] > height * 0.91
            for text, _, _ in b.lines:
                key = boilerplate_key(text)
                if len(key) < 3:
                    continue
                seen_any.add(key)
                if near_edge:
                    seen_edge.add(key)
        edge_counts.update(seen_edge)
        any_counts.update(seen_any)
    n = len(pages)
    if n < 3:
        return set()
    edge = {k for k, c in edge_counts.items() if c >= max(3, 0.3 * n)}
    anywhere = {k for k, c in any_counts.items() if c >= max(4, 0.6 * n) and len(k) < 80}
    return edge | anywhere


def _is_figure_label(lines: list[tuple[str, float, bool]], body_size: float) -> bool:
    """Short small-font fragments are usually text inside figures (axis labels, legends)."""
    text = " ".join(t for t, _, _ in lines)
    sizes = [s for _, s, _ in lines if s]
    if not sizes or caption_label(text):
        return False
    return max(sizes) < body_size - 1.0 and len(text.split()) < 8


def _is_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    words = len(text.split())
    if not text or words > 16 or len(text) > 140 or caption_label(text):
        return False
    if not re.search(r"[A-Za-z]{3}", text) or text.startswith(("•", "●", "▪", "-", "*")):
        return False
    if "@" in text or text.count(",") >= 3:  # e-mail lines, author lists
        return False
    if section_category(text):
        return words <= 8
    if text.rstrip().endswith((",", ";")) or re.match(r"^[a-z]", text):
        return False
    larger = size >= body_size + 0.9
    if (bold and size >= body_size - 0.3) or larger:
        # Sentences ending with a period are usually emphasised text, unless numbered.
        return not text.endswith(".") or bool(_NUMBERING.match(text))
    return False


def parse_pdf(path: Path, ocr_enabled: bool = True, ocr_min_chars: int = 80, ocr_dpi: int = 300,
              ocr_lang: str = "eng") -> ParsedDocument:
    path = Path(path)
    doc = pymupdf.open(path)
    try:
        return _parse(doc, path, ocr_enabled, ocr_min_chars, ocr_dpi, ocr_lang)
    finally:
        doc.close()


def _parse(doc: pymupdf.Document, path: Path, ocr_enabled: bool, ocr_min_chars: int,
           ocr_dpi: int, ocr_lang: str) -> ParsedDocument:
    pages: list[list[_RawBlock]] = []
    heights: list[float] = []
    ocr_pages = 0
    warnings: list[str] = []
    early_text_parts: list[str] = []

    for page in doc:
        page_no = page.number + 1
        raw = _read_page_blocks(page, page_no)
        n_chars = sum(len(b.text) for b in raw)
        if n_chars < ocr_min_chars and ocr_enabled:
            text = ocr_page(page, dpi=ocr_dpi, lang=ocr_lang)
            if text.strip():
                ocr_pages += 1
                raw = [
                    _RawBlock(lines=[(normalize_text(p), 0.0, False)], bbox=(0, i, page.rect.width, i + 1), page=page_no)
                    for i, p in enumerate(re.split(r"\n\s*\n", text))
                    if p.strip()
                ]
            elif n_chars == 0 and len(page.get_images()) > 0:
                warnings.append(f"page {page_no}: image-only page and OCR unavailable")
        if page_no <= 2:
            early_text_parts.append(page.get_text())
        pages.append(_reading_order(raw, page.rect.width) if raw and raw[0].lines[0][1] else raw)
        heights.append(page.rect.height)

    # Body font size: the size covering the most characters.
    size_weights: Counter[float] = Counter()
    for blocks in pages:
        for b in blocks:
            for text, size, _ in b.lines:
                if size:
                    size_weights[round(size, 1)] += len(text)
    body_size = size_weights.most_common(1)[0][0] if size_weights else 10.0

    boilerplate = _find_boilerplate(pages, heights)

    first_page_lines = [
        (t, s, b.bbox[1]) for b in (pages[0] if pages else []) for t, s, _ in b.lines
    ]

    blocks: list[Block] = []
    for raw_blocks in pages:
        for rb in raw_blocks:
            lines = [
                ln for ln in rb.lines
                if boilerplate_key(ln[0]) not in boilerplate and not is_page_number(ln[0])
            ]
            if not lines or _is_figure_label(lines, body_size):
                continue
            blocks.extend(_split_block(_RawBlock(lines, rb.bbox, rb.page), body_size))

    _merge_caption_continuations(blocks, body_size)
    _assign_sections(blocks)
    return ParsedDocument(
        path=path,
        n_pages=doc.page_count,
        blocks=blocks,
        pdf_metadata=dict(doc.metadata or {}),
        first_page_lines=first_page_lines,
        early_text=normalize_text("\n".join(early_text_parts)),
        body_size=body_size,
        ocr_pages=ocr_pages,
        warnings=warnings,
    )


def _split_block(rb: _RawBlock, body_size: float) -> list[Block]:
    """Split leading heading line(s) off a block; classify the remainder."""
    out: list[Block] = []
    lines = list(rb.lines)
    # Leading heading lines (a heading can wrap onto two lines).
    head: list[str] = []
    while lines and len(head) < 3:
        text, size, bold = lines[0]
        candidate = " ".join(head + [text])
        if _is_heading(candidate, size, bold, body_size):
            head.append(text)
            lines.pop(0)
        else:
            break
    if head and lines and not section_category(" ".join(head)):
        # A bold first line followed by more lines in the same style is a bold
        # paragraph (e.g. a highlighted summary), not a heading.
        _, h_size, h_bold = rb.lines[len(head) - 1]
        _, n_size, n_bold = lines[0]
        if h_bold == n_bold and abs(h_size - n_size) <= 0.5:
            lines = list(rb.lines)
            head = []
    if head:
        heading_text = join_lines(" ".join(head))
        out.append(Block(heading_text, rb.page, "heading", rb.size, True, rb.bbox))
    if lines:
        text = join_lines(" ".join(t for t, _, _ in lines))
        if not text:
            return out
        sizes = [s for _, s, _ in lines]
        size = max(set(sizes), key=sizes.count)
        bold = sum(1 for _, _, b in lines if b) > len(lines) / 2
        kind, label = "paragraph", ""
        if cap := caption_label(text):
            kind, label = cap
        out.append(Block(text, rb.page, kind, size, bold, rb.bbox, label=label))
    return out


def _merge_caption_continuations(blocks: list[Block], body_size: float) -> None:
    """Captions often span several blocks set in a smaller font; merge them."""
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind in ("figure_caption", "table_caption"):
            j = i + 1
            while j < len(blocks):
                nxt = blocks[j]
                same_style = abs(nxt.size - b.size) <= 0.3 and nxt.size < body_size - 0.2
                if nxt.kind == "paragraph" and nxt.page == b.page and same_style:
                    b.text = f"{b.text} {nxt.text}"
                    blocks.pop(j)
                else:
                    break
            b.text = re.sub(r"\(legend continued on next page\)", "", b.text, flags=re.I).strip()
        i += 1


def _assign_sections(blocks: list[Block]) -> None:
    category, section = "front", ""
    for b in blocks:
        if b.kind == "heading":
            cat = section_category(b.text)
            if cat:
                category = cat
            section = b.text
        elif b.kind == "paragraph" and _ABSTRACT_PREFIX.match(b.text) and len(b.text.split()) > 40 \
                and category in ("front", "abstract"):
            category, section = "abstract", "Abstract"
            b.text = _ABSTRACT_PREFIX.sub("", b.text, count=1)
        b.section = section or ("Front matter" if category == "front" else category.title())
        b.category = category
        if b.kind == "figure_caption":
            b.category = "figure"
        elif b.kind == "table_caption":
            b.category = "table"
        if b.kind in ("figure_caption", "table_caption"):
            # Figure placement is a layout artefact (often after the references),
            # so the caption's own label is the meaningful section name.
            b.section = b.label
