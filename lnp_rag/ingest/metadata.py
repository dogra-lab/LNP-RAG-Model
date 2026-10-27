"""Paper-level metadata: title, authors, DOI, year (+ optional Crossref lookup)."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from lnp_rag.ingest.cleaning import join_lines
from lnp_rag.ingest.pdf_parser import ParsedDocument

log = logging.getLogger(__name__)

_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]+)", re.I)
_YEAR_CONTEXT = re.compile(
    r"(?:received|accepted|published|available online|final edited form as:|©|copyright|\(c\))"
    r"[\s\S]{0,80}?\b((?:19|20)\d{2})\b",
    re.I,
)
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_JOURNAL_YEAR = re.compile(
    rf"\b((?:19|20)\d{{2}})\s*(?:;|,)?\s*(?:\d{{1,4}}\s*\(\d+\)|{_MONTHS}\s*\d{{0,2}}\s*;\s*\d+)",
    re.I,
)  # "2020;12(3)" and PMC style "2015 December ; 28: 1-9"
_DOI_YEAR = re.compile(r"\.((?:19|20)\d{2})\.")  # Elsevier-style DOIs: 10.1016/j.actbio.2022.09.065
_BAD_TITLE = re.compile(
    r"^(untitled|microsoft word|document\d*|\d+$|title$|manuscript|pii:|doi:|s\d{4}-|"
    r"[\w-]+\.(pdf|docx?|tex)$|article$|research article$|review article$|original article$|"
    r"hhs public access|nih public access|author manuscript|(the |international )?journal of|"
    r"proceedings of|annals of)"
    r"|author manuscript|nih-pa|available in pmc"  # PMC manuscript banners
    r"|_|\.\.|\b\d{5,}\b",  # file-system style ids: "IJN_A_517385 8085..8099"
    re.I,
)
_BAD_AUTHOR = re.compile(r"^(author|user|admin|unknown|microsoft|\w+\.\w+@|elsevier|springer)", re.I)


def clean_doi(raw: str) -> str:
    doi = raw.strip().rstrip(".)]}'\"").lower()
    doi = re.sub(r"(\.|/)(pdf|full|abstract|html|epub)$", "", doi)
    return doi


def extract_doi(doc: ParsedDocument) -> str | None:
    meta_text = " ".join(str(v) for v in doc.pdf_metadata.values() if v)
    for source in (meta_text, doc.early_text):
        dois = [clean_doi(m) for m in _DOI.findall(source)]
        # Ignore DOIs of cited works that appear in reference-like lines; the paper's
        # own DOI is usually the first one on page 1.
        dois = [d for d in dois if len(d) > 10]
        if dois:
            return dois[0]
    return None


def extract_title(doc: ParsedDocument) -> tuple[str, str]:
    """Returns (title, source)."""
    meta_title = (doc.pdf_metadata.get("title") or "").strip()
    if 15 <= len(meta_title) <= 300 and not _BAD_TITLE.search(meta_title) and " " in meta_title:
        return join_lines(meta_title), "pdf"

    # Heuristic: the largest-font run of consecutive lines in the top 60% of page 1.
    bottom = max((y for _, _, y in doc.first_page_lines), default=0.0)
    lines = [ln for ln in doc.first_page_lines if ln[2] < 0.6 * bottom and len(ln[0]) > 2]
    runs: list[tuple[float, list[str]]] = []
    for text, size, _ in lines:
        if runs and abs(runs[-1][0] - size) < 0.2:
            runs[-1][1].append(text)
        else:
            runs.append((size, [text]))
    for _, parts in sorted(runs, key=lambda r: r[0], reverse=True):
        candidate = join_lines(" ".join(parts))
        if 4 <= len(candidate.split()) <= 40 and not _BAD_TITLE.search(candidate):
            return candidate, "heuristic"
    # Filenames often carry the title, sometimes behind a catalogue code ("CU_306_...").
    stem = re.sub(r"^[A-Z]{1,4}[_ -]?\d{1,5}[_ -]+", "", doc.path.stem)
    return stem.replace("_", " ").strip(), "filename"


def extract_authors(doc: ParsedDocument, title: str) -> list[str]:
    raw = (doc.pdf_metadata.get("author") or "").strip()
    if raw and not _BAD_AUTHOR.match(raw):
        parts = re.split(r"\s*(?:;|,|\band\b|&)\s*", raw)
        authors = [p.strip() for p in parts if len(p.strip().split()) >= 2 and len(p) < 60]
        # Some PDFs put title fragments in the author field ("In Vitro, In Vivo");
        # if any "name" is part of the title, the whole field is untrustworthy.
        if any(a.lower() in title.lower() for a in authors):
            authors = []
        if authors:
            return authors[:50]
    # Heuristic: the line block right after the title that looks like a name list.
    texts = [t for t, _, _ in doc.first_page_lines]
    joined_title = title.lower()[:40]
    for i, t in enumerate(texts):
        if joined_title and joined_title[:25] in t.lower():
            for cand in texts[i + 1 : i + 5]:
                cand = re.sub(r"[\d*†‡§¶#,]+(?=\s|,|$)", ",", cand)
                names = [n.strip(" ,") for n in re.split(r",|\band\b|&", cand)]
                names = [n for n in names if re.fullmatch(r"[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]*){1,3}", n)]
                if len(names) >= 2 and not any(n.lower() in title.lower() for n in names):
                    return names[:50]
            break
    return []


def extract_year(doc: ParsedDocument, doi: str | None = None) -> int | None:
    years = [int(y) for y in _YEAR_CONTEXT.findall(doc.early_text)]
    years += [int(y) for y in _JOURNAL_YEAR.findall(doc.early_text)]
    years = [y for y in years if 1950 <= y <= 2100]
    if years:
        return Counter(years).most_common(1)[0][0]
    if doi and (m := _DOI_YEAR.search(doi)):
        return int(m.group(1))
    return None


class CrossrefClient:
    """Minimal Crossref works lookup with on-disk caching and retries."""

    def __init__(self, cache_path: Path, mailto: str = ""):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS crossref (doi TEXT PRIMARY KEY, body TEXT)")
        ua = "lnp-rag/1.0" + (f" (mailto:{mailto})" if mailto else "")
        self._http = httpx.Client(timeout=15, headers={"User-Agent": ua})

    def lookup(self, doi: str) -> dict | None:
        row = self._db.execute("SELECT body FROM crossref WHERE doi = ?", (doi,)).fetchone()
        if row:
            return json.loads(row[0]) or None
        try:
            body = self._fetch(doi)
        except Exception as exc:
            log.warning("crossref_failed", extra={"doi": doi, "error": str(exc)})
            return None
        self._db.execute("INSERT OR REPLACE INTO crossref VALUES (?, ?)", (doi, json.dumps(body or {})))
        self._db.commit()
        return body

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _fetch(self, doi: str) -> dict | None:
        r = self._http.get(f"https://api.crossref.org/works/{doi}")
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            r.raise_for_status()
        if r.status_code != 200:
            return None
        msg = r.json().get("message", {})
        year = None
        for key in ("published-print", "published-online", "issued", "created"):
            parts = (msg.get(key) or {}).get("date-parts") or [[None]]
            if parts[0] and parts[0][0]:
                year = int(parts[0][0])
                break
        return {
            "title": " ".join(msg.get("title") or []),
            "authors": [
                " ".join(filter(None, [a.get("given"), a.get("family")]))
                for a in msg.get("author") or []
            ],
            "year": year,
            "journal": " ".join(msg.get("container-title") or []) or None,
        }
