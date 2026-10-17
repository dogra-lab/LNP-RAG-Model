"""Search pipeline shared by the CLI (search.py) and the web app (app.py)."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .index import PaperIndex
from .profile import criteria_guidance
from .screen import VERDICT_RANK, Screener, _system_prompt, build_user_message
from .steps import STEPS


@dataclass
class SearchParams:
    query: str
    step: str | None = None
    step_name: str | None = None
    step_description: str | None = None
    guidance: str | None = None
    strict_kinetic: bool = False
    top: int = 100
    candidates: int = 200
    only_relevant: bool = False
    no_llm: bool = False
    provider: str | None = None
    model: str | None = None
    effort: str = "low"
    workers: int = 16
    use_validated: bool = True  # learn from the validated set: its criteria, search facets and evidence passages


MATCH_THRESHOLD = 0.94  # title+abstract cosine above which two PDFs are the same paper


def validated_links(index: PaperIndex, validated: PaperIndex) -> dict[int, int]:
    """Library paper idx -> validated paper idx, for library PDFs that are copies of a validated paper."""
    if getattr(index, "_validated_links", None) is None:
        sim = index.head_vectors() @ validated.head_vectors().T
        best = sim.argmax(1)
        index._validated_links = {int(i): int(best[i]) for i in np.flatnonzero(sim.max(1) >= MATCH_THRESHOLD)}
    return index._validated_links


def facet_scores(index: PaperIndex, facet_vecs: np.ndarray) -> np.ndarray:
    """Per library paper: mean dense relevance to the learned search facets."""
    return np.mean([index._paper_scores(index.emb @ f) for f in facet_vecs], axis=0)


def evidence_similarity(
    index: PaperIndex, validated: PaperIndex, facet_vecs: np.ndarray, per_paper: int = 8
) -> np.ndarray:
    """Per library paper: how closely its passages resemble the validated papers' evidence passages.

    Each validated paper contributes the passages most aligned with the facets (the reason it was
    included), so this matches on kind of evidence rather than on topic.
    """
    key = facet_vecs.tobytes()
    if getattr(index, "_evidence_key", None) != key:
        aligned = (validated.emb @ facet_vecs.T).max(1)
        exemplars = []
        for pid in range(len(validated.papers)):
            ids = np.flatnonzero(validated.owner == pid)
            exemplars += ids[np.argsort(-aligned[ids])[:per_paper]].tolist()
        sims = index.emb @ validated.emb[exemplars].T
        index._evidence_sim = index._paper_scores(np.sort(sims, axis=1)[:, -3:].mean(1))
        index._evidence_key = key
    return index._evidence_sim


def run_search(
    index: PaperIndex,
    params: SearchParams,
    cache_path: Path,
    on_progress: Callable[[int, int], None] | None = None,
    validated: PaperIndex | None = None,
    criteria: dict | None = None,
) -> dict:
    """Retrieve, optionally screen, and rank papers.

    Returns {"meta": {...}, "validated": [...], "results": [...]}: every validated paper (ranked by
    relevance to the query) plus the top additional library papers, excluding copies of validated ones.
    `criteria` (from learn.py) adds the selection logic learned from the validated set to retrieval and screening.
    """
    preset = STEPS.get(params.step or "", {})
    query = params.query.strip()
    step_name = params.step_name or preset.get("name") or query
    step_desc = params.step_description or preset.get("description") or query
    if not step_name:
        raise ValueError("Enter a query or choose a step")
    guidance = params.guidance if params.guidance else preset.get("guidance", "")
    learned = criteria if (criteria and validated is not None and params.use_validated) else None
    if learned:
        guidance = f"{guidance}\n{criteria_guidance(learned)}".strip()

    retrieval_query = query if step_desc == query else f"{query}. {step_name}: {step_desc}".lstrip(". ")
    n_candidates = params.top if params.no_llm else max(params.candidates, params.top)

    validated_rows, exclude, extra = [], set(), []
    if validated is not None:
        links = validated_links(index, validated)
        exclude = set(links)
        in_library = set(links.values())
        for rank, (vid, _, _) in enumerate(validated.search(retrieval_query, len(validated.papers)), 1):
            paper = validated.papers[vid]
            validated_rows.append(
                {
                    "relevance_rank": rank,
                    "id": paper["id"],
                    "label": paper["label"],
                    "title": paper["title"],
                    "is_review": paper["is_review"],
                    "in_library": vid in in_library,
                    "files": len(paper["all_paths"]),
                    "path": paper["path"],
                    "all_paths": paper["all_paths"],
                    "abstract": paper["abstract"][:1200],
                }
            )

    facet_vecs = None
    if validated is not None and params.use_validated:
        # Learned facets when learn.py has run, else the query itself defines which evidence to look for.
        facet_vecs = index.encode(learned["facets"] if learned else [retrieval_query])
        extra = [facet_scores(index, facet_vecs), evidence_similarity(index, validated, facet_vecs)]
    hits = index.search(retrieval_query, n_candidates + 30, extra_scores=extra, exclude=exclude)
    if facet_vecs is not None:  # show the screener the passages most like the evidence it is looking for
        evidence_vecs = np.vstack([index.encode([retrieval_query]), facet_vecs])
        hits = [(pid, fused, index.best_chunks(pid, evidence_vecs)) for pid, fused, _ in hits]

    # The same paper can exist as two different PDF files (e.g. preprint + publisher copy).
    title_key = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())[:80]
    seen_titles = {title_key(r["title"]) for r in validated_rows}  # also catches validated copies
    unique_hits = []
    for hit in hits:
        key = title_key(index.papers[hit[0]]["title"])
        if key not in seen_titles:
            seen_titles.add(key)
            unique_hits.append(hit)

    rows = []
    for rank, (pid, fused, chunk_ids) in enumerate(unique_hits[:n_candidates], 1):
        paper = index.papers[pid]
        rows.append(
            {
                "retrieval_rank": rank,
                "retrieval_score": round(fused, 5),
                "id": paper["id"],
                "title": paper["title"],
                "is_review": paper["is_review"],
                "folders": "; ".join(paper["folders"]),
                "path": paper["path"],
                "abstract": paper["abstract"][:1200],
                "_msg": build_user_message(paper, [index.chunks[c] for c in chunk_ids]),
            }
        )

    model = None
    errors = 0
    if not params.no_llm:
        system = _system_prompt(step_name, step_desc, params.strict_kinetic, guidance)
        provider = params.provider or ("openai" if os.environ.get("OPENAI_API_KEY") else "anthropic")
        screener = Screener(system, cache_path, provider=provider, model=params.model, effort=params.effort)
        model = f"{provider}:{screener.model}"
        results = screener.screen_many([r["_msg"] for r in rows], workers=params.workers, on_progress=on_progress)
        for row, res in zip(rows, results):
            row["verdict"] = res.get("verdict", "error")
            row["score"] = res.get("score", 0.0)
            row["study_type"] = res.get("study_type", "")
            row["data_types"] = res.get("data_types", [])
            row["rationale"] = res.get("rationale") or res.get("error", "")
        rows.sort(key=lambda r: (-VERDICT_RANK.get(r["verdict"], -1), -r["score"], r["retrieval_rank"]))
        if params.only_relevant:
            rows = [r for r in rows if r["verdict"] in ("include", "borderline")]
        errors = sum(r["verdict"] == "error" for r in rows)

    # Keep every included paper, then fill up to `top` with the rest.
    n_include = sum(r.get("verdict") == "include" for r in rows)
    rows = rows[: max(params.top, n_include)]
    for r in rows:
        del r["_msg"]

    meta = {
        **asdict(params),
        "validated_count": len(validated_rows),
        "step_name": step_name,
        "step_description": step_desc,
        "guidance": guidance,
        "learned_criteria": bool(learned),
        "model": model,
        "errors": errors,
    }
    return {"meta": meta, "validated": validated_rows, "results": rows}


def export_included(output: dict, out_dir: Path) -> tuple[Path, int]:
    """Copy the PDFs of every included paper (validated + screened 'include') into out_dir/<name>/.

    Files are numbered in ranked order; supplements follow their paper. An included_papers.csv lists them.
    Returns (folder, number of papers).
    """
    papers = [{**r, "verdict": "include"} for r in output.get("validated", [])]
    papers += [r for r in output["results"] if r.get("verdict") == "include"]
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = lambda t: re.sub(r"\s+", " ", re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", t)).strip()[:90] or "untitled"
    rows = []
    for rank, paper in enumerate(papers, 1):
        name = f"{rank:03d} {safe(paper['title'])}"
        sources = paper.get("all_paths") or [paper["path"]]
        if "all_paths" not in paper:  # library paper: identical copies in several folders, keep one
            sources = [paper["path"]]
        files = []
        for i, src in enumerate(sources):
            dest = out_dir / (f"{name}.pdf" if i == 0 else f"{name} (supplement {i}).pdf")
            if Path(src).exists():
                shutil.copy2(src, dest)
                files.append(dest.name)
        rows.append({
            "rank": rank, "title": paper["title"], "score": paper.get("score", ""),
            "data_types": "; ".join(paper.get("data_types", [])), "rationale": paper.get("rationale", ""),
            "files": "; ".join(files), "source_path": paper["path"],
        })
    with open(out_dir / "included_papers.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["rank"])
        writer.writeheader()
        writer.writerows(rows)
    return out_dir, len(rows)


def save_results(output: dict, out_dir: Path) -> Path:
    """Write <timestamp>_<query>.csv and .json; returns the path stem."""
    out_dir.mkdir(exist_ok=True)
    rows = [{"section": "validated", **r} for r in output.get("validated", [])]
    rows += [{"section": "additional", **r} for r in output["results"]]
    name = output["meta"]["query"] or output["meta"]["step_name"]
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:50]
    stem = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}"
    if rows:
        with open(f"{stem}.csv", "w", newline="") as f:
            fieldnames = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
    Path(f"{stem}.json").write_text(json.dumps(output, indent=2))
    return stem
