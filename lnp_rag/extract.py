"""PDF -> structured paper record (title, abstract, captions, results, chunks)."""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import pymupdf

MAX_PAGES = 40  # skip giant supplements / theses beyond this
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
MAX_CHUNKS = 80

_CAPTION_RE = re.compile(
    r"^\s*(supplementary\s+)?(figure|fig\.?|table|scheme)\s*S?\d+[a-z]?\s*[.:|\-–—]",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*\.?\s*)?([A-Z][A-Za-z &]{2,40})\s*$")
_REFS_RE = re.compile(r"^\s*(references|bibliography|literature cited)\s*$", re.IGNORECASE | re.MULTILINE)
_ABSTRACT_RE = re.compile(r"(^|\n)\s*(a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|summary)\s*[:.]?\s*\n?", re.IGNORECASE)
_ABSTRACT_END_RE = re.compile(
    r"\n\s*(\d+\.?\s*)?(introduction|background|keywords?|key\s*words|graphical abstract|"
    r"abbreviations|significance)\b",
    re.IGNORECASE,
)
_REVIEW_TITLE_RE = re.compile(
    r"\b(review|overview|perspective|advances in|progress in|recent developments|"
    r"state of the art|landscape|trends in|primer)\b",
    re.IGNORECASE,
)
_REVIEW_BODY_RE = re.compile(
    r"\b(in this review|this review|we review|here,? we review|this article reviews|"
    r"review article|mini[- ]review|systematic review)\b",
    re.IGNORECASE,
)
_JUNK_TITLES = re.compile(r"(microsoft word|untitled|^\s*$|\.docx?$|\.pdf$|^doi|^pii)", re.IGNORECASE)


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _title_from_filename(path: Path) -> str:
    stem = re.sub(r"^CU_\d+_", "", path.stem)
    return stem.replace("_", " ").strip()


def _title_from_fonts(page: pymupdf.Page) -> str:
    """Longest text run among the three largest font sizes (skips banners like 'NIH Public Access')."""
    by_size: dict[float, list[str]] = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip():
                    by_size.setdefault(round(span["size"]), []).append(span["text"])
    candidates = [_flat(" ".join(by_size[size])) for size in sorted(by_size, reverse=True)[:5]]
    candidates = [c for c in candidates if 15 <= len(c) <= 300 and not _looks_like_junk(c)]
    return max(candidates[:3], key=lambda c: len(c.split()), default="")


def _looks_like_junk(text: str) -> bool:
    """Author lists, letter-spaced journal banners ('S C I E N C E'), affiliation digits, preprint notices."""
    digits = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    return (
        text.count(",") >= 3
        or bool(re.search(r"\b(\w\s){4,}", text))
        or digits > 0.15
        or bool(re.search(r"peer review|journals —|copyright|downloaded from", text, re.IGNORECASE))
    )


def _pick_title(doc: pymupdf.Document, path: Path) -> str:
    meta = _flat(doc.metadata.get("title") or "")
    font_title = _title_from_fonts(doc[0]) if doc.page_count else ""
    filename_title = _title_from_filename(path)
    if (
        len(meta.split()) >= 4
        and not _JUNK_TITLES.search(meta)
        and not re.search(r"\d+\.\.\d+|\\|et al|journals —", meta, re.IGNORECASE)
        and _norm(meta) != _norm(filename_title)
    ):
        return meta
    if re.search(r"et al|^P\d+", filename_title):  # curated filenames ('P009 Kim et al 2025')
        if font_title:
            return font_title
        for para in re.split(r"\n\s*\n", doc[0].get_text()) if doc.page_count else []:
            text = _flat(para)
            if len(text.split()) >= 6 and len(text) <= 300 and not _looks_like_junk(text):
                return text
    # CU_* filenames are truncated, so prefer the font heuristic when it agrees on the opening words.
    if font_title and _norm(font_title)[:12] == _norm(filename_title)[:12]:
        return font_title
    return filename_title or font_title


def _abstract(front: str) -> str:
    m = _ABSTRACT_RE.search(front)
    if m:
        rest = front[m.end():]
        end = _ABSTRACT_END_RE.search(rest)
        abstract = rest[: end.start()] if end and end.start() > 200 else rest[:3000]
        return _flat(abstract)[:3000]
    # No explicit heading: take the first long paragraph (abstracts precede the introduction).
    paras = [
        p for p in re.split(r"\n\s*\n", front)
        if len(p) > 500 and not re.search(r"corresponding author|reprints|permissions|@|copyright", p, re.I)
    ]
    return _flat(paras[0])[:3000] if paras else ""


def _section(lines: list[str], names: tuple[str, ...], stops: tuple[str, ...]) -> str:
    start = None
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if not m:
            continue
        heading = m.group(3).strip().lower()
        if start is None and heading.startswith(names):
            start = i + 1
        elif start is not None and heading.startswith(stops):
            return "\n".join(lines[start:i])
    return "\n".join(lines[start:]) if start is not None else ""


def _chunks(text: str) -> list[str]:
    words = _flat(text).split(" ")
    if not words or words == [""]:
        return []
    size = CHUNK_CHARS // 6  # ~6 chars per word
    step = size - CHUNK_OVERLAP // 6
    out = [" ".join(words[i : i + size]) for i in range(0, len(words), step)]
    return [c for c in out if len(c) > 200][:MAX_CHUNKS]


def extract_paper(path: str) -> dict | None:
    """Parse one PDF. Returns None if the file can't be read or has no text."""
    p = Path(path)
    try:
        doc = pymupdf.open(p)
    except Exception:
        return None
    with doc:
        pages = [doc[i].get_text() for i in range(min(doc.page_count, MAX_PAGES))]
        blocks = []
        for i in range(min(doc.page_count, MAX_PAGES)):
            blocks.extend(b[4] for b in doc[i].get_text("blocks") if b[6] == 0)
        title = re.sub(r"\s*author manuscript\s*$", "", _pick_title(doc, p), flags=re.I)

    full = _clean("\n".join(pages))
    if len(full) < 500:
        return None  # scanned/image-only PDF

    refs = _REFS_RE.search(full)
    body = full[: refs.start()] if refs and refs.start() > len(full) * 0.3 else full
    front = _clean("\n".join(pages[:2]))
    abstract = _abstract(front)

    captions = []
    for b in blocks:
        if _CAPTION_RE.match(b):
            cap = _flat(b)[:600]
            if cap not in captions:
                captions.append(cap)

    lines = body.split("\n")
    results = _section(
        lines,
        names=("results",),
        stops=("discussion", "conclusion", "materials", "methods", "experimental", "acknowledg"),
    )

    head = f"{title}\n{front[:3000]}".lower()
    is_review = bool(_REVIEW_TITLE_RE.search(title) or _REVIEW_BODY_RE.search(head)) and not results

    chunks = [f"{title}. {abstract}"] + _chunks(body)
    if captions:
        chunks += _chunks(" ".join(captions))

    return {
        "path": str(p),
        "folder": p.parent.name,
        "title": title,
        "abstract": abstract,
        "captions": captions[:40],
        "results": _flat(results)[:8000],
        "is_review": is_review,
        "chunks": chunks,
    }
