"""Provider-independent prompts, output schemas and request builders for LLM synthesis.

The LLM only ever sees excerpts retrieved from the user's PDFs, each tagged with an
ID. It must cite those IDs; ``synthesis.grounding`` then discards any citation that
does not exist, so explanations stay tied to real PDF text.
"""
from __future__ import annotations

from typing import Protocol
from xml.sax.saxutils import escape, quoteattr


class SynthesisError(RuntimeError):
    pass


SYNTHESIS_SYSTEM = """\
You assess whether scientific papers contain evidence for a researcher's literature query.

You receive the query and, for each candidate paper, numbered excerpts retrieved from that \
paper's PDF (paragraphs, figure captions, table captions). These excerpts are the only \
information you have about each paper.

For every paper:
- relevance: 0-100, how well the excerpts show the paper contains what the query asks for. \
Judge each requested element separately (e.g. the measurement type, multiple time points, \
experimental conditions) and score lower when elements are missing.
- evidence_strength: "strong" if the excerpts directly show the requested evidence, "partial" \
if they show some elements, "insufficient" if they do not support a match.
- explanation: 1-3 sentences on why the paper does or does not match. Every factual statement \
must be supported by an excerpt and followed by its ID in brackets, e.g. [P2-E1]. Quote numbers \
and time points exactly as written in the excerpt. Do not use outside knowledge about the paper.
- cited_evidence_ids: the excerpt IDs you relied on.
- study_type: in vitro / in vivo / both, only as stated or clearly described in the excerpts; \
otherwise "unknown".
- missing_evidence: which requested elements the excerpts do not show (empty string if none).

If the excerpts are not enough to tell, say so in the explanation and mark the evidence \
insufficient rather than inferring what the paper probably contains. A review article that \
only summarises other studies does not itself contain primary data; say so when that is the case.\
"""

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_key": {"type": "string"},
                    "relevance": {"type": "integer"},
                    "evidence_strength": {"type": "string", "enum": ["strong", "partial", "insufficient"]},
                    "study_type": {"type": "string", "enum": ["in vitro", "in vivo", "both", "unknown"]},
                    "explanation": {"type": "string"},
                    "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "missing_evidence": {"type": "string"},
                },
                "required": [
                    "paper_key", "relevance", "evidence_strength", "study_type",
                    "explanation", "cited_evidence_ids", "missing_evidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["papers"],
    "additionalProperties": False,
}

PARAPHRASE_SYSTEM = """\
You rewrite literature-search queries for a semantic search engine over scientific PDFs. \
Write alternative phrasings a paper's own text might use (synonyms, abbreviations, \
related measurement terms). Each rewrite must ask for exactly the same thing as the \
original: do not broaden, narrow, or add new requirements.\
"""

PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {"rewrites": {"type": "array", "items": {"type": "string"}}},
    "required": ["rewrites"],
    "additionalProperties": False,
}

CRITIC_SYSTEM = """\
You are the critic agent for one folder of a scientific paper collection. A retriever agent \
searched that folder and proposed candidate papers for a researcher's query. Your job is to \
check each candidate and decide whether it really answers the query.

For each paper you receive numbered excerpts retrieved from its PDF: paragraphs, figure \
captions and table captions. These excerpts are the only information you have about the paper. \
Figure and table captions are full evidence: a paper whose matching data is described only in \
a caption (e.g. "Figure 3. Uptake of DiD-LNPs at 1, 4 and 24 h") answers the query.

For every paper:
- verdict: "ACCEPT" if the excerpts show the paper contains what the query asks for, i.e. \
its core request (the measurement, material and system asked about). Anything introduced by \
"especially", "prioritize", "preferably", "ideally" or "such as" is a preference, not a \
requirement: a paper that meets the core request but lacks a preferred element (e.g. it \
has uptake data at a single time point when several are preferred) is still ACCEPT, and \
the reason says which preferred element is missing. "REJECT" if the excerpts do not show the core request, or only mention it in passing \
(e.g. the introduction citing other work, or a review summarising other studies).
- reason: for ACCEPT, 1-3 sentences explaining WHY the paper is a correct answer: which \
requested element each excerpt shows, citing excerpt IDs in brackets, e.g. [P2-E1]. For \
REJECT, one sentence on what is missing. Quote numbers and time points exactly as written. \
Do not use outside knowledge about the paper.
- cited_evidence_ids: the excerpt IDs the verdict relies on.
- quotes: for ACCEPT, 1-3 short verbatim quotes (copied exactly, character for character) from \
the cited excerpts that prove the match, each with its excerpt ID. Empty for REJECT.

Before judging any paper, fill in core_request (the elements a paper MUST show to answer \
the query) and preferences (elements that only make a paper a better match). Then judge \
every paper against core_request alone.

Papers marked as near-miss scored just under the retriever's threshold; judge them by the \
same standard, since the retriever may have under-scored them.\
"""

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "core_request": {"type": "string"},
        "preferences": {"type": "string"},
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_key": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
                    "reason": {"type": "string"},
                    "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "quotes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["evidence_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["paper_key", "verdict", "reason", "cited_evidence_ids", "quotes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["core_request", "preferences", "papers"],
    "additionalProperties": False,
}


def critic_user_message(query: str, folder: str, papers: list[dict]) -> str:
    """``papers``: [{key, title, is_review, near_miss, evidence: [{id, type, section, page, text}]}]."""
    parts = [f"<query>{escape(query)}</query>", f"<folder>{escape(folder)}</folder>", "<papers>"]
    for p in papers:
        parts.append(f"<paper key={quoteattr(p['key'])}>")
        parts.append(f"<title>{escape(p['title'])}</title>")
        if p.get("is_review"):
            parts.append("<note>Detected as a review article.</note>")
        if p.get("near_miss"):
            parts.append("<note>Near-miss: scored just under the retriever threshold.</note>")
        for e in p["evidence"]:
            parts.append(
                f"<excerpt id={quoteattr(e['id'])} type={quoteattr(e['type'])} "
                f"section={quoteattr(e['section'])} page=\"{e['page']}\">{escape(e['text'])}</excerpt>"
            )
        parts.append("</paper>")
    parts.append("</papers>")
    return "\n".join(parts)


def synthesis_user_message(query: str, papers: list[dict]) -> str:
    """``papers``: [{key, title, is_review, evidence: [{id, type, section, page, text}]}]."""
    parts = [f"<query>{escape(query)}</query>", "<papers>"]
    for p in papers:
        parts.append(f"<paper key={quoteattr(p['key'])}>")
        parts.append(f"<title>{escape(p['title'])}</title>")
        if p.get("is_review"):
            parts.append("<note>Detected as a review article.</note>")
        for e in p["evidence"]:
            parts.append(
                f"<excerpt id={quoteattr(e['id'])} type={quoteattr(e['type'])} "
                f"section={quoteattr(e['section'])} page=\"{e['page']}\">{escape(e['text'])}</excerpt>"
            )
        parts.append("</paper>")
    parts.append("</papers>")
    return "\n".join(parts)


def paraphrase_user_message(query: str, n: int) -> str:
    return f"Write {n} rewrites of this query.\n<query>{escape(query)}</query>"


class LLMClient(Protocol):
    """What the searcher and query expander need from an LLM provider."""

    name: str  # "openai" | "claude"
    model: str

    def synthesize(self, query: str, papers: list[dict]) -> dict[str, dict]: ...

    def paraphrase_query(self, query: str, n: int = 3) -> list[str]: ...

    def review(self, query: str, folder: str, papers: list[dict]) -> dict[str, dict]: ...
