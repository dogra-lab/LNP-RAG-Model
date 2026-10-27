"""Grounding checks for LLM output and the extractive (no-LLM) fallback."""
from __future__ import annotations

import re

from lnp_rag.models import ChunkType, Evidence, StudyType

_CITE = re.compile(r"\[([A-Z]\d+-E\d+(?:\s*,\s*[A-Z]\d+-E\d+)*)\]")


def validate_synthesis(item: dict, evidence: list[Evidence]) -> dict:
    """Drop citations to excerpts that were not provided for this paper.

    Returns a cleaned copy with ``grounded`` = whether at least one valid citation remains.
    """
    valid = {e.evidence_id for e in evidence}
    cited = [c for c in item.get("cited_evidence_ids", []) if c in valid]

    def fix_marker(m: re.Match) -> str:
        ids = [i.strip() for i in m.group(1).split(",") if i.strip() in valid]
        cited.extend(i for i in ids if i not in cited)
        return f"[{', '.join(ids)}]" if ids else ""

    explanation = _CITE.sub(fix_marker, item.get("explanation", "")).strip()
    explanation = re.sub(r"\s{2,}", " ", explanation)
    strength = item.get("evidence_strength", "insufficient")
    grounded = bool(cited)
    if not grounded and strength != "insufficient":
        strength = "insufficient"
        explanation += " (No retrieved excerpt was cited for this assessment.)"
    relevance = max(0, min(100, int(item.get("relevance", 0))))
    return {
        **item,
        "explanation": explanation,
        "cited_evidence_ids": cited,
        "evidence_strength": strength,
        "relevance": relevance if grounded else min(relevance, 20),
        "grounded": grounded,
    }


def remap_citations(text: str, mapping: dict[str, str]) -> str:
    """Rename evidence IDs (e.g. after re-ranking P3 becomes P1)."""
    return re.sub(r"\b([A-Z]\d+)-E(\d+)\b", lambda m: f"{mapping.get(m.group(1), m.group(1))}-E{m.group(2)}", text)


def extractive_explanation(
    evidence: list[Evidence], intents: dict[str, bool], top_relevance: float
) -> tuple[str, str, str]:
    """Explanation built only from detected signals. Returns (text, strength, missing)."""
    if not evidence:
        return "No supporting excerpts were retrieved.", "insufficient", "all requested evidence"
    parts = []
    for e in evidence[:3]:
        where = e.label or e.section
        signals = []
        if e.features.has_uptake_terms:
            signals.append("uptake measurement terms")
        if e.features.n_timepoints >= 2:
            tps = ", ".join(e.features.timepoints[:6])
            signals.append(f"{e.features.n_timepoints} time points ({tps})" if tps else "multiple time points")
        if e.features.has_quantitative:
            signals.append("numeric values")
        if e.features.has_dose:
            signals.append("dose/concentration")
        desc = ", ".join(signals) if signals else "query-relevant text"
        parts.append(f"{where} (p. {e.page}) contains {desc} [{e.evidence_id}]")
    missing = []
    any_tp = any(e.features.n_timepoints >= 2 for e in evidence)
    any_q = any(e.features.has_quantitative for e in evidence)
    any_cap = any(e.chunk_type in (ChunkType.FIGURE_CAPTION, ChunkType.TABLE_CAPTION) for e in evidence)
    if intents.get("multi_timepoint") and not any_tp:
        missing.append("multiple time points")
    if intents.get("quantitative") and not any_q:
        missing.append("quantitative values")
    if intents.get("figure") and not any_cap:
        missing.append("a matching figure/table caption")
    if top_relevance >= 0.5 and not missing:
        strength = "strong"
    elif top_relevance >= 0.15:
        strength = "partial"
    else:
        strength = "insufficient"
    text = "Matched by keyword/semantic retrieval (automatic summary, not an LLM judgement): " + "; ".join(parts) + "."
    return text, strength, ", ".join(missing)


def study_type_from_evidence(evidence: list[Evidence], paper_level: StudyType) -> StudyType:
    if paper_level != StudyType.UNKNOWN:
        return paper_level
    vitro = any(e.features.mentions_in_vitro for e in evidence)
    vivo = any(e.features.mentions_in_vivo for e in evidence)
    if vitro and vivo:
        return StudyType.BOTH
    if vitro:
        return StudyType.IN_VITRO
    if vivo:
        return StudyType.IN_VIVO
    return StudyType.UNKNOWN
