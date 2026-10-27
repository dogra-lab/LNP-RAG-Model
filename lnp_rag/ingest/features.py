"""Rule-based detection of evidence signals in chunk text.

These signals feed metadata filters (``has_quantitative``, in vitro / in vivo)
and evidence boosts during ranking. They are deliberately transparent regexes so
that a reviewer can see *why* a chunk was flagged.
"""
from __future__ import annotations

import re

from lnp_rag.models import ChunkFeatures, StudyType

# Numbers never start mid-number: without the look-behind, a long digit run is
# re-scanned from every position (quadratic).
_NUM = r"(?<![\d.])\d+(?:\.\d+)?"

# Time points: "24 h", "0.5 hr", "30 min", "day 7", "1, 4 and 24 h", "2-8 h".
_TIME_UNIT = r"(?:h|hr|hrs|hours?|min|mins|minutes?|days?|d|weeks?|wk)"
# Each whitespace character has exactly one way to match. An ambiguous separator
# such as r"\s*(?:,\s*(?:and)?)\s*" backtracks exponentially on long number lists.
_LIST_SEP = r"\s*(?:,(?:\s*(?:and|or)\b)?|and\b|or\b|&)\s*"
_TIME_LIST = re.compile(
    rf"\b(?P<nums>{_NUM}(?:{_LIST_SEP}{_NUM})*)\s*(?:-\s*)?(?P<unit>{_TIME_UNIT})\b", re.I
)
_TIME_RANGE = re.compile(rf"\b{_NUM}\s*(?:-|to)\s*{_NUM}\s*{_TIME_UNIT}\b", re.I)
_DAY_N = re.compile(
    rf"\b(?P<unit>day|week)s?\s+(?P<nums>{_NUM}(?:{_LIST_SEP}{_NUM})*)\b", re.I
)
_UNIT_NORMAL = {"h": "h", "m": "min", "d": "d", "w": "wk"}
_TIME_WORDS = re.compile(
    r"\b(time[- ]dependent|time[- ]course|kinetic(?:s)?|over time|at different time points|"
    r"time points?)\b",
    re.I,
)

# Quantities with scientific units, percentages and statistics.
_QUANT = re.compile(
    rf"(?:{_NUM}\s*(?:%|±|\+/-)|"
    rf"{_NUM}\s*(?:nm|μm|mm|nM|μM|mM|pM|M|ng|μg|mg|g|kg|mL|μL|L|kDa|mV|°C|fold|-fold)\b|"
    rf"\bp\s*[<=>]\s*0?\.\d+|\bMFI\b|\bIC50\b|\bEC50\b|\bn\s*=\s*\d+|"
    rf"\bmean\s*(?:±|\+/-)|\bstandard deviation\b|\bs\.?d\.?\b(?=\s*[\(\),.;]))",
    re.I,
)
_DOSE = re.compile(
    rf"(?:{_NUM}\s*(?:mg|μg|ng)\s*/\s*(?:kg|mL|well)|{_NUM}\s*(?:nM|μM|mM|pM)\b|"
    rf"{_NUM}\s*(?:μg|ng|mg)\s+(?:of\s+)?(?:mRNA|siRNA|DNA|RNA|lipid)|"
    rf"\b(?:dose|doses|dosage|concentration[s]?)\b)",
    re.I,
)
_UPTAKE = re.compile(
    r"\b(uptake|internali[sz](?:ation|ed)|endocytos[ie]s|cellular entry|cell[- ]associated|"
    r"flow cytometry|FACS|confocal|fluorescence intensity|MFI|mean fluorescence|"
    r"DiD|DiI|DiR|Cy5|Cy3|rhodamine|FITC|transfection efficiency)\b",
    re.I,
)
_IN_VITRO = re.compile(
    r"\b(in vitro|cell line[s]?|cells were (?:seeded|cultured|incubated|treated)|"
    r"cultured cells|cell culture|HeLa|HEK[- ]?293T?|HepG2|Huh-?7|RAW ?264\.7|"
    r"A549|MCF-?7|Caco-?2|bEnd\.3|DC2\.4|BMDCs?|primary (?:human|murine|mouse) \w+ cells)\b",
    re.I,
)
_IN_VIVO = re.compile(
    r"\b(in vivo|mice|mouse|murine model|rats?\b|rabbits?|non-human primates?|NHPs?|"
    r"zebrafish|animals? (?:were|was)|intravenous(?:ly)? (?:injected|administered|injection)|"
    r"i\.v\.|intramuscular(?:ly)?|subcutaneous(?:ly)?|biodistribution|IACUC|"
    r"animal (?:experiments?|studies|care))\b",
    re.I,
)
_REVIEW_TITLE = re.compile(
    r"\b(review|overview|perspective|advances in|recent advances|recent progress|"
    r"progress in|state of the art|prospects|challenges and opportunities|mini[- ]review|"
    r"a decade of|landscape|roadmap|insights? into)\b",
    re.I,
)
_REVIEW_ABSTRACT = re.compile(
    r"\b(this review|in this review|we review|here,? we review|this article reviews|"
    r"we summarize|we summarise|this perspective|systematic review|meta-analysis|"
    r"we discuss recent|this mini-review)\b",
    re.I,
)
_EXPERIMENTAL_CUES = re.compile(
    r"\b(we (?:synthesi[sz]ed|prepared|formulated|developed|measured|evaluated|treated|injected|found|"
    r"observed|demonstrate[ds]?|introduce[ds]?|employed|report|show(?:ed)?|used|tested|investigated|"
    r"designed|engineered|screened|identified|performed|compared|generated|established)|"
    r"were incubated|were injected|statistical analysis)\b",
    re.I,
)


def extract_timepoints(text: str) -> list[str]:
    """Return distinct time points (normalised, e.g. ``"24 h"``) mentioned in text."""
    found: set[str] = set()
    for pattern in (_TIME_LIST, _DAY_N):
        for m in pattern.finditer(text):
            unit = _UNIT_NORMAL[m.group("unit")[0].lower()]
            for n in re.findall(_NUM, m.group("nums")):
                found.add(f"{float(n):g} {unit}")
    return sorted(found, key=_time_sort_key)


def _time_sort_key(tp: str) -> float:
    value, unit = tp.split()
    scale = {"min": 1 / 60, "h": 1, "d": 24, "wk": 168}.get(unit, 1)
    return float(value) * scale


def detect_features(text: str) -> ChunkFeatures:
    timepoints = extract_timepoints(text)
    n_tp = len(timepoints)
    if n_tp < 2 and _TIME_WORDS.search(text) and _TIME_RANGE.search(text):
        n_tp = 2  # "2-24 h time course": at least two points implied
    return ChunkFeatures(
        has_quantitative=len(_QUANT.findall(text)) >= 2,
        n_timepoints=n_tp,
        timepoints=timepoints[:12],
        has_dose=bool(_DOSE.search(text)),
        has_uptake_terms=bool(_UPTAKE.search(text)),
        mentions_in_vitro=bool(_IN_VITRO.search(text)),
        mentions_in_vivo=bool(_IN_VIVO.search(text)),
    )


def classify_study_type(
    chunk_features: list[tuple[str, ChunkFeatures]],
) -> tuple[StudyType, str]:
    """Paper-level in vitro / in vivo label from methods/results/caption chunks.

    ``chunk_features`` is a list of (section_category, features). Introduction and
    discussion mentions are ignored because they often describe *other* work.
    Returns the label and a short human-readable basis.
    """
    evidential = {"methods", "results", "figure", "table", "abstract"}
    vitro = sum(1 for cat, f in chunk_features if cat in evidential and f.mentions_in_vitro)
    vivo = sum(1 for cat, f in chunk_features if cat in evidential and f.mentions_in_vivo)
    basis = f"in vitro cues in {vitro} methods/results/caption chunks; in vivo cues in {vivo}"
    if vitro >= 2 and vivo >= 2:
        return StudyType.BOTH, basis
    if vitro >= 2:
        return StudyType.IN_VITRO, basis
    if vivo >= 2:
        return StudyType.IN_VIVO, basis
    return StudyType.UNKNOWN, basis


def looks_like_review(title: str, abstract: str, section_names: list[str]) -> bool:
    """Heuristic review-article detector (review articles rarely contain primary data)."""
    if _REVIEW_ABSTRACT.search(abstract):
        return True
    # Methods are often only in the supplement, so a Results section or first-person
    # experimental language also marks primary research.
    has_experimental_sections = any(
        re.search(r"method|experimental|materials|^\W*(\d+\.?\s*)?results\b", s, re.I) for s in section_names
    )
    if has_experimental_sections or _EXPERIMENTAL_CUES.search(abstract):
        return False
    if _REVIEW_TITLE.search(title):
        return True
    return len(section_names) > 6


def query_intents(query: str) -> dict[str, bool]:
    """Which evidence types the query asks for; gates the intent boosts."""
    q = query.lower()
    return {
        "quantitative": bool(
            re.search(r"quantitativ|measure|data|intensity|percent|%|level|amount|mfi|fold", q)
        ),
        "multi_timepoint": bool(
            re.search(
                r"over time|time[- ]?(?:point|course|dependent)|kinetic|multiple time|"
                r"\b\d+\s*h\b|hours|curve",
                q,
            )
        ),
        "dose": bool(re.search(r"dose|concentration|dosage|mg/kg|μg|ug/|nm\b|\bμm\b", q)),
        "figure": bool(re.search(r"figure|fig\.|histogram|curve|plot|graph|table|image", q)),
    }
