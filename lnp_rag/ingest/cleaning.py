"""Text cleaning and normalisation for PDF-extracted text."""
from __future__ import annotations

import re
import unicodedata

# Ligatures and typographic characters PDF extractors commonly emit.
_CHAR_MAP = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": " - ",
        "−": "-",  # minus sign
        " ": " ",
        " ": " ",
        " ": " ",
        "​": "",
        "­": "",  # soft hyphen
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "µ": "μ",  # micro sign -> greek mu, so "µg" matches one form
    }
)

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WS = re.compile(r"[ \t\f\v]+")
_MULTI_NL = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """Normalise unicode, ligatures, whitespace and line-break hyphenation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    text = _CONTROL.sub(" ", text)
    # "nano-\nparticle" -> "nanoparticle" only when the next line starts lowercase,
    # so genuine hyphenated names ("DLin-\nMC3") keep their hyphen.
    text = _HYPHEN_BREAK.sub(
        lambda m: m.group(1) + ("" if m.group(2).islower() else "-") + m.group(2), text
    )
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def join_lines(text: str) -> str:
    """Collapse a block's soft line breaks into a single paragraph."""
    return _WS.sub(" ", normalize_text(text).replace("\n", " ")).strip()


_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
_DIGITS = re.compile(r"\d+")


def boilerplate_key(line: str) -> str:
    """Key used to detect running headers/footers repeated across pages."""
    return _DIGITS.sub("#", line.strip().lower())[:80]


def is_page_number(line: str) -> bool:
    return bool(_PAGE_NUM.match(line))


def word_count(text: str) -> int:
    return len(text.split())
