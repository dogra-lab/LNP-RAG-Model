"""Learn why the validated papers ('paper used/') were included.

1. Each validated paper is read by the LLM (captions, Results excerpts, supplement) and summarized as an
   evidence profile: which uptake measurements it reports, at which time points / doses, and where the
   data lives (figures, tables, supplement).
2. The profiles are synthesized into general inclusion criteria (screening guidance) and a set of search
   queries (retrieval facets). Nothing here names individual papers, so the criteria transfer to the library.

Output: <validated index>/profiles.jsonl and <validated index>/criteria.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .index import QUERY_PREFIX, PaperIndex
from .screen import Screener, build_user_message

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "uptake_assays": {"type": "array", "items": {"type": "string"}},
        "time_points": {"type": "string"},
        "doses": {"type": "string"},
        "intracellular_accumulation": {"type": "string"},
        "endosomal_escape": {"type": "string"},
        "cell_models": {"type": "array", "items": {"type": "string"}},
        "cargo": {"type": "string"},
        "data_access": {"type": "array", "items": {"type": "string", "enum": ["figures", "tables", "supplementary"]}},
        "pkpd_value": {"type": "string"},
    },
    "required": ["uptake_assays", "time_points", "doses", "intracellular_accumulation", "endosomal_escape",
                 "cell_models", "cargo", "data_access", "pkpd_value"],
    "additionalProperties": False,
}

PROFILE_SYSTEM = (
    "You are curating evidence for pharmacokinetic/pharmacodynamic (PK/PD) modeling of lipid nanoparticle "
    "(LNP) delivery. The paper below was hand-picked by a domain expert as a source of cellular uptake data. "
    "Extract what quantitative LNP cellular-uptake evidence it reports, using the abstract, figure/table "
    "captions, Results excerpts and supplementary material.\n"
    "- uptake_assays: methods used to measure uptake/internalization (e.g. 'flow cytometry of DiD-labeled "
    "LNPs (MFI)', 'confocal microscopy of Cy5-mRNA'). Empty if none.\n"
    "- time_points: the time points of uptake/delivery measurements in cells, or 'none'.\n"
    "- doses: the LNP or cargo doses/concentrations tested in cells, or 'none'.\n"
    "- intracellular_accumulation: how intracellular amount/localization was quantified, or 'none'.\n"
    "- endosomal_escape: how endosomal escape/disruption was measured, or 'none'.\n"
    "- cell_models: cells or tissues used.\n"
    "- cargo: delivered cargo (mRNA, siRNA, pDNA, small molecule...).\n"
    "- data_access: where the quantitative data are presented.\n"
    "- pkpd_value: one sentence on what this paper contributes to a PK/PD model of LNP uptake.\n"
    "Be specific and terse; quote numbers when given."
)

CRITERIA_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "criteria": {"type": "array", "items": {"type": "string"}},
        "exclusions": {"type": "array", "items": {"type": "string"}},
        "facets": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "criteria", "exclusions", "facets"],
    "additionalProperties": False,
}

CRITERIA_SYSTEM = (
    "You are given evidence profiles of papers that a domain expert selected as sources of quantitative "
    "LNP cellular uptake data for PK/PD modeling, plus the expert's research question. Infer the expert's "
    "selection logic so it can be applied to thousands of unseen papers.\n"
    "- summary: 3-5 sentences on what the selected papers have in common (and what varies, e.g. topic, "
    "cargo, therapeutic area), written for a screener.\n"
    "- criteria: 5-8 concrete, checkable inclusion criteria, from most to least important.\n"
    "- exclusions: 3-6 kinds of papers that should be excluded.\n"
    "- facets: 8-12 diverse search queries (one sentence each, written like a Results passage) that would "
    "retrieve passages with this kind of evidence, covering the different assay types, time-course and "
    "dose-dependent designs, intracellular accumulation and endosomal escape.\n"
    "Generalize: never name specific papers, authors, lipids brands or disease applications as requirements."
)


def _llm(system: str, cache: Path, schema: dict, provider: str, model: str | None, effort: str) -> Screener:
    return Screener(system, cache, provider=provider, model=model, effort=effort, schema=schema)


def _evidence_chunks(index: PaperIndex, query_vecs: np.ndarray, pid: int, k: int = 4) -> list[str]:
    ids = np.flatnonzero(index.owner == pid)
    sims = (index.emb[ids] @ query_vecs.T).max(1)
    return [index.chunks[i] for i in ids[np.argsort(-sims)[:k]]]


def learn_from_validated(
    validated: PaperIndex, out_dir: Path, research_question: str,
    provider: str = "openai", model: str | None = None, workers: int = 8,
) -> dict:
    if validated._model is None:
        from .index import load_embedder

        validated._model = load_embedder()
    qv = validated._model.encode([QUERY_PREFIX + research_question], normalize_embeddings=True).astype(np.float32)

    profiler = _llm(PROFILE_SYSTEM, out_dir / "profile_cache.jsonl", PROFILE_SCHEMA, provider, model, "low")
    msgs = [build_user_message(p, _evidence_chunks(validated, qv, i)) for i, p in enumerate(validated.papers)]
    profiles = profiler.screen_many(msgs, workers=workers)
    with open(out_dir / "profiles.jsonl", "w") as f:
        for paper, prof in zip(validated.papers, profiles):
            f.write(json.dumps({"id": paper["id"], "label": paper["label"], "title": paper["title"], **prof}) + "\n")

    ok = [p for p in profiles if "error" not in p]
    body = "\n\n".join(
        f"Paper {i}:\n" + "\n".join(f"{k}: {v}" for k, v in p.items()) for i, p in enumerate(ok, 1)
    )
    synth = _llm(CRITERIA_SYSTEM, out_dir / "profile_cache.jsonl", CRITERIA_SCHEMA, provider, model, "medium")
    criteria = synth.screen_one(f"Research question: {research_question}\n\n{body}")
    if "error" in criteria:
        raise RuntimeError(f"Could not derive criteria: {criteria['error']}")
    criteria.update(research_question=research_question, n_papers=len(ok))
    (out_dir / "criteria.json").write_text(json.dumps(criteria, indent=2))
    return criteria


def load_criteria(validated_dir: Path) -> dict | None:
    path = validated_dir / "criteria.json"
    return json.loads(path.read_text()) if path.exists() else None


def criteria_guidance(criteria: dict) -> str:
    return (
        "The expert's reference set of included papers shares this profile: "
        f"{criteria['summary']}\nInclude when: " + "; ".join(criteria["criteria"])
        + ".\nExclude: " + "; ".join(criteria["exclusions"]) + "."
    )
