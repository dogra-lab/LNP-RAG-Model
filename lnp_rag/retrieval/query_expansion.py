"""Query expansion / multi-query generation.

Modes (QUERY_EXPANSION_MODE):

* ``none``   - original query only.
* ``rules``  - curated synonym groups from ``config/synonyms.yaml``.
* ``llm``    - rules + up to three paraphrases written by the configured LLM
  (OpenAI or Claude), constrained to preserve the original intent; falls back
  to rules on any error. ``claude`` is accepted as an alias.

The original query is always kept and always weighted highest in fusion.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    original: str
    variants: list[str] = field(default_factory=list)  # paraphrases for dense retrieval
    keywords: list[str] = field(default_factory=list)  # extra BM25 terms
    source: str = "none"

    @property
    def all_dense(self) -> list[str]:
        return [self.original, *self.variants]

    @property
    def bm25_query(self) -> str:
        return " ".join([self.original, *self.keywords])


@dataclass
class SynonymTable:
    equivalent: list[list[str]]
    related: dict[str, list[str]]

    @classmethod
    def load(cls, path: Path) -> SynonymTable:
        if not path.exists():
            return cls([], {})
        data = yaml.safe_load(path.read_text()) or {}
        return cls(
            equivalent=[[str(t) for t in g] for g in data.get("equivalent", [])],
            related={str(k): [str(t) for t in v] for k, v in (data.get("related") or {}).items()},
        )


def _term_regex(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])", re.I)


def expand_with_rules(query: str, table: SynonymTable, max_variants: int = 3) -> ExpandedQuery:
    keywords: list[str] = []
    variants: list[str] = []
    matched_groups: list[tuple[str, list[str]]] = []
    for group in table.equivalent:
        # Longest matching term first so "lipid nanoparticles" beats "LNP" inside it.
        for term in sorted(group, key=len, reverse=True):
            if _term_regex(term).search(query):
                others = [t for t in group if t.lower() != term.lower()]
                matched_groups.append((term, others))
                break
    seen = {query.lower()}
    for term, others in matched_groups:
        for other in others:
            if not _term_regex(other).search(query) and other.lower() not in (k.lower() for k in keywords):
                keywords.append(other)
        # One paraphrase per matched group, substituting the first multi-word alternative
        # (multi-word alternatives carry more meaning for the embedding model than acronyms).
        alt = next((o for o in others if " " in o or "-" in o), others[0] if others else None)
        if alt and len(variants) < max_variants:
            variant = _term_regex(term).sub(alt, query, count=1)
            if variant.lower() not in seen:
                seen.add(variant.lower())
                variants.append(variant)
    # Related terms fire when the anchor - or any synonym of it - is in the query.
    matched_terms = {t.lower() for term, others in matched_groups for t in [term, *others]}
    for anchor, terms in table.related.items():
        if anchor.lower() in matched_terms or _term_regex(anchor).search(query):
            keywords.extend(t for t in terms if t not in keywords)
    return ExpandedQuery(query, variants, keywords, source="rules" if (variants or keywords) else "none")


class QueryExpander:
    def __init__(self, mode: str, synonyms_path: Path, llm=None):
        self.mode = "llm" if mode == "claude" else mode
        self.table = SynonymTable.load(synonyms_path)
        self._llm = llm  # synthesis.prompts.LLMClient | None

    @lru_cache(maxsize=256)
    def expand(self, query: str) -> ExpandedQuery:
        if self.mode == "none":
            return ExpandedQuery(query)
        expanded = expand_with_rules(query, self.table)
        if self.mode == "llm" and self._llm is not None:
            try:
                paraphrases = self._llm.paraphrase_query(query)
                existing = {v.lower() for v in expanded.all_dense}
                extra = [p for p in paraphrases if p.lower() not in existing]
                expanded.variants = (extra + expanded.variants)[:4]
                expanded.source = f"{self._llm.name}+rules"
            except Exception as exc:  # never fail a search because expansion failed
                log.warning("llm_expansion_failed", extra={"error": str(exc)})
        return expanded
