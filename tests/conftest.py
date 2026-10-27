"""Shared fixtures: synthetic PDFs, a deterministic fake embedder, isolated settings."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from lnp_rag.config import RetrievalConfig, Settings
from lnp_rag.embeddings import Embedder
from lnp_rag.services import Services

UPTAKE_PAPER = {
    "title": "Time-resolved cellular uptake of ionizable lipid nanoparticles in hepatocytes",
    "authors": "Jane Smith, Rahul Mehta, Wei Chen",
    "sections": [
        ("Abstract", [
            "Lipid nanoparticles (LNPs) are the leading platform for mRNA delivery. Here we quantify the "
            "cellular uptake of DiD-labelled LNPs in HepG2 cells by flow cytometry at 0.5, 1, 2, 4 and 24 h "
            "and show that uptake increases with time and dose. In vivo biodistribution in mice confirms "
            "liver accumulation. doi:10.1234/jnano.2023.0042. Received 12 March 2023."
        ]),
        ("1. Introduction", [
            "Messenger RNA therapeutics require efficient delivery vehicles. Lipid nanoparticles composed of "
            "an ionizable lipid, DSPC, cholesterol and a PEG-lipid have enabled vaccines and protein "
            "replacement therapies. The kinetics of cellular entry are, however, poorly characterised."
        ]),
        ("2. Materials and Methods", [
            "HepG2 cells were seeded at 1 x 10^5 cells per well and incubated with DiD-labelled LNPs at "
            "0.5, 1 and 2 μg/mL mRNA. Cells were collected after 0.5, 1, 2, 4 and 24 h and analysed by flow "
            "cytometry. Mean fluorescence intensity (MFI) was recorded for 10,000 events per sample.",
            "Female C57BL/6 mice received 0.5 mg/kg mRNA intravenously. Animal experiments were approved "
            "by the IACUC. Organs were imaged 6 h after injection.",
        ]),
        ("3. Results", [
            "Uptake was time-dependent: the MFI increased from 1.8 ± 0.2 x 10^3 at 0.5 h to 2.4 ± 0.3 x 10^4 "
            "at 24 h (n = 3, p < 0.01). Uptake also increased with dose, reaching 85 ± 4% DiD-positive "
            "cells at 2 μg/mL after 4 h.",
        ]),
        ("Figure", [
            "Figure 2. Cellular uptake kinetics of DiD-LNPs in HepG2 cells. (A) Flow cytometry histograms "
            "at 0.5, 1, 2, 4 and 24 h. (B) Mean fluorescence intensity over time (mean ± SD, n = 3)."
        ]),
        ("4. Discussion", [
            "Our data show that LNP uptake in hepatocytes is rapid and saturates after 4 h, consistent "
            "with apolipoprotein E mediated endocytosis."
        ]),
        ("References", [
            "1. Doe J. et al. Some unrelated cited paper about uptake kinetics. Nature 2020; 12(3): 100-110.",
        ]),
    ],
}

FORMULATION_PAPER = {
    "title": "Microfluidic formulation parameters control lipid nanoparticle size and polydispersity",
    "authors": "Ana Lopez, Tom Baker",
    "sections": [
        ("Abstract", [
            "We studied how the flow rate ratio and total flow rate in a staggered herringbone mixer affect "
            "particle size, polydispersity index and encapsulation efficiency of lipid nanoparticles."
        ]),
        ("2. Methods", [
            "Lipids were dissolved in ethanol and mixed with mRNA in citrate buffer at flow rate ratios of "
            "3:1 and 5:1. Particle size was measured by dynamic light scattering."
        ]),
        ("3. Results", [
            "Increasing the total flow rate from 2 to 12 mL/min reduced the particle diameter from 120 nm to "
            "75 nm while the polydispersity index stayed below 0.2. Encapsulation efficiency exceeded 90%."
        ]),
        ("Table", ["Table 1. Size, PDI and encapsulation efficiency of LNPs prepared at different flow rates."]),
    ],
}


def make_pdf(path: Path, paper: dict) -> Path:
    """Write a simple single-column PDF with real font sizes/weights for headings."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60.0

    def ensure_space(height: float):
        nonlocal page, y
        if y + height > 780:
            page = doc.new_page()
            y = 60.0

    page.insert_textbox(pymupdf.Rect(50, y, 545, y + 60), paper["title"], fontsize=16, fontname="hebo")
    y += 60
    page.insert_textbox(pymupdf.Rect(50, y, 545, y + 30), paper["authors"], fontsize=10, fontname="helv")
    y += 30
    for heading, paragraphs in paper["sections"]:
        if heading not in ("Figure", "Table"):
            ensure_space(40)
            page.insert_textbox(pymupdf.Rect(50, y, 545, y + 30), heading, fontsize=12, fontname="hebo")
            y += 22
        for para in paragraphs:
            lines = max(1, len(para) // 95 + 1)
            height = lines * 13 + 8
            ensure_space(height)
            size = 9 if heading in ("Figure", "Table") else 10
            page.insert_textbox(pymupdf.Rect(50, y, 545, y + height + 20), para, fontsize=size, fontname="helv")
            y += height + 10
    doc.set_metadata({"title": paper["title"], "author": paper["authors"]})
    doc.save(path)
    doc.close()
    return path


class FakeEmbedder(Embedder):
    """Deterministic hashed bag-of-words embedding: texts sharing words are similar."""

    name = "fake-hash-embedder"

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls += len(texts)
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLLM:
    """Stands in for an LLM client; records calls and returns scripted output."""

    name = "fake"
    model = "fake-model"

    def __init__(self, response_fn=None):
        self.calls: list[tuple[str, list[dict]]] = []
        self._fn = response_fn

    def synthesize(self, query: str, papers: list[dict]) -> dict[str, dict]:
        self.calls.append((query, papers))
        if self._fn:
            return self._fn(query, papers)
        return {
            p["key"]: {
                "paper_key": p["key"],
                "relevance": 90 if "uptake" in p["title"].lower() else 10,
                "evidence_strength": "strong" if "uptake" in p["title"].lower() else "insufficient",
                "study_type": "both",
                "explanation": f"Uptake measured by flow cytometry [{p['evidence'][0]['id']}] [Z9-E9].",
                "cited_evidence_ids": [p["evidence"][0]["id"], "Z9-E9"],
                "missing_evidence": "",
            }
            for p in papers
        }

    def paraphrase_query(self, query: str, n: int = 3) -> list[str]:
        return [f"{query} (rephrased)"]


@pytest.fixture
def papers_dir(tmp_path: Path) -> Path:
    d = tmp_path / "papers"
    (d / "Cellular_Uptake").mkdir(parents=True)
    (d / "Formulation").mkdir(parents=True)
    make_pdf(d / "Cellular_Uptake" / "uptake_kinetics.pdf", UPTAKE_PAPER)
    make_pdf(d / "Formulation" / "microfluidics.pdf", FORMULATION_PAPER)
    return d


@pytest.fixture
def settings(tmp_path: Path, papers_dir: Path) -> Settings:
    return Settings(
        _env_file=None,
        papers_dir=papers_dir,
        data_dir=tmp_path / "data",
        reranker_enabled=False,
        synthesis_enabled=False,
        query_expansion_mode="rules",
        ingest_workers=1,
        anthropic_api_key=None,
        openai_api_key=None,
        api_public_url="http://testserver",
    )


@pytest.fixture
def services(settings: Settings) -> Services:
    svc = Services(settings, RetrievalConfig())
    fake = FakeEmbedder()
    svc.__dict__["embedder"] = fake  # override cached_property
    svc.__dict__["llm"] = None
    return svc
