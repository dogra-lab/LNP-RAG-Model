"""End-to-end: indexing (incremental), hybrid search, LLM grounding, HTTP API."""
import os
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from lnp_rag.api.main import create_app
from lnp_rag.models import SearchFilters, SearchRequest, StudyType

from .conftest import FakeLLM

QUERY = "quantitative cellular uptake of lipid nanoparticles over time by flow cytometry"


def _index(services, **kw):
    status = services.indexer.run(**kw)
    assert status.state == "done", status.message
    return status


def test_index_and_search(services):
    status = _index(services)
    assert status.processed_files == 2 and not status.failed
    papers = services.catalog.all_papers()
    assert len(papers) == 2
    uptake = next(p for p in papers.values() if p.filename == "uptake_kinetics.pdf")
    assert uptake.folder_labels == ["Cellular_Uptake"]
    assert uptake.doi == "10.1234/jnano.2023.0042" and uptake.year == 2023
    assert uptake.n_figures == 1 and uptake.has_quantitative
    assert uptake.study_type == StudyType.BOTH
    assert services.vectors.count() == services.catalog.count_chunks() == len(services.bm25)

    resp = services.searcher.search(SearchRequest(query=QUERY, top_k=5, synthesize=False))
    assert resp.results[0].filename == "uptake_kinetics.pdf"
    top = resp.results[0]
    assert top.rank == 1 and top.evidence and top.pages
    assert top.explanation_source == "extractive"
    assert any(e.chunk_type.value == "figure_caption" for e in top.evidence + top.captions)
    assert top.pdf_url.startswith("http://testserver/papers/") and "#page=" in top.pdf_url
    assert resp.expanded_queries  # rule-based expansion fired


def test_filters(services):
    _index(services)
    resp = services.searcher.search(
        SearchRequest(query="particle size flow rate", synthesize=False, filters=SearchFilters(folder="Formulation"))
    )
    assert {r.filename for r in resp.results} == {"microfluidics.pdf"}
    resp = services.searcher.search(
        SearchRequest(query="uptake", synthesize=False, filters=SearchFilters(year_min=2030))
    )
    assert resp.results == [] and resp.warnings


def test_incremental_reindex(services, papers_dir):
    _index(services)
    calls = services.embedder.calls

    # unchanged: nothing re-embedded
    st = _index(services)
    assert st.processed_files == 0 and services.embedder.calls == calls
    assert st.unchanged == 2

    # duplicate copy in another folder: same paper, extra path, no re-embedding
    (papers_dir / "Copies").mkdir()
    shutil.copy(papers_dir / "Cellular_Uptake" / "uptake_kinetics.pdf", papers_dir / "Copies" / "dup.pdf")
    st = _index(services)
    assert st.processed_files == 0 and st.new == 1
    uptake = next(p for p in services.catalog.all_papers().values() if "uptake" in p.title.lower())
    assert sorted(uptake.folder_labels) == ["Cellular_Uptake", "Copies"]

    # modified content: re-processed as a new paper, old version removed
    target = papers_dir / "Formulation" / "microfluidics.pdf"
    from .conftest import FORMULATION_PAPER, make_pdf

    changed = dict(FORMULATION_PAPER, title="Microfluidic mixing revisited: size control of lipid nanoparticles")
    make_pdf(target, changed)
    os.utime(target, (time.time() + 5, time.time() + 5))
    st = _index(services)
    assert st.modified == 1 and st.processed_files == 1
    titles = {p.title for p in services.catalog.all_papers().values()}
    assert changed["title"] in titles and FORMULATION_PAPER["title"] not in titles

    # deletion
    target.unlink()
    st = _index(services)
    assert st.deleted == 1
    assert len(services.catalog.all_papers()) == 1
    assert services.vectors.count() == services.catalog.count_chunks()


def test_rebuild(services):
    _index(services)
    n = services.catalog.count_chunks()
    st = _index(services, rebuild=True)
    assert st.processed_files == 2 and services.catalog.count_chunks() == n


def test_llm_synthesis_is_grounded(services):
    _index(services)
    fake = FakeLLM()
    services.searcher.llm = fake
    resp = services.searcher.search(SearchRequest(query=QUERY, top_k=2, synthesize=True))
    assert fake.calls, "the LLM should be called"
    # The LLM only receives retrieved excerpts with IDs
    sent = fake.calls[0][1]
    assert all(e["id"].startswith(p["key"] + "-E") for p in sent for e in p["evidence"])
    top = resp.results[0]
    assert top.explanation_source == "fake"
    assert "Z9-E9" not in top.explanation and "Z9-E9" not in top.cited_evidence_ids  # hallucinated id removed
    assert all(c in {e.evidence_id for e in top.evidence + top.captions} for c in top.cited_evidence_ids)
    assert top.filename == "uptake_kinetics.pdf" and top.evidence_strength == "strong"


def test_llm_failure_falls_back_to_extractive(services):
    from lnp_rag.synthesis.prompts import SynthesisError

    _index(services)

    def boom(q, p):
        raise SynthesisError("rate limited")

    services.searcher.llm = FakeLLM(boom)
    resp = services.searcher.search(SearchRequest(query=QUERY, top_k=2, synthesize=True))
    assert resp.results[0].explanation_source == "extractive"
    assert any("LLM synthesis unavailable" in w for w in resp.warnings)


@pytest.fixture
def client(services):
    app = create_app(services, warmup=False)
    with TestClient(app) as c:
        yield c


def test_api_end_to_end(client, services, papers_dir):
    assert client.get("/health").json()["indexed_papers"] == 0
    assert client.post("/search", json={"query": QUERY}).json()["warnings"][0].startswith("The index is empty")

    r = client.post("/index", json={})
    assert r.status_code == 202
    for _ in range(200):
        st = client.get("/index/status").json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert st["state"] == "done", st

    res = client.post("/search", json={"query": QUERY, "top_k": 3, "synthesize": False}).json()
    top = res["results"][0]
    assert top["filename"] == "uptake_kinetics.pdf"
    pid = top["paper_id"]
    assert client.get(f"/papers/{pid}").json()["doi"] == "10.1234/jnano.2023.0042"
    pdf = client.get(f"/papers/{pid}/pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert pdf.headers["content-type"] == "application/pdf"
    assert client.get("/papers/doesnotexist/pdf").status_code == 404

    # upload validation
    up = client.post("/upload", files=[("files", ("evil.pdf", b"not a pdf", "application/pdf"))]).json()
    assert up["saved"] == [] and up["rejected"]
    good = (papers_dir / "Formulation" / "microfluidics.pdf").read_bytes()
    up = client.post("/upload", files=[("files", ("../../x.pdf", good, "application/pdf"))]).json()
    assert up["saved"] == ["x.pdf"] and (papers_dir / "uploads" / "x.pdf").exists()

    assert client.delete("/index").json() == {"cleared": True}
    assert client.get("/stats").json()["papers"] == 0


def test_rebuild_after_embedding_model_change(services, settings):
    from lnp_rag.config import RetrievalConfig
    from lnp_rag.services import Services
    from lnp_rag.store.vector_store import EmbeddingModelMismatch

    from .conftest import FakeEmbedder

    _index(services)
    other = Services(settings.model_copy(update={"embedding_model": "another/model"}), RetrievalConfig())
    other.__dict__["embedder"] = FakeEmbedder()
    other.__dict__["llm"] = None
    with pytest.raises(EmbeddingModelMismatch):
        _ = other.vectors
    other.prepare_rebuild()
    st = other.indexer.run(rebuild=True)
    assert st.state == "done" and other.vectors.count() == other.catalog.count_chunks() > 0


def test_pdf_timeout_is_reported_not_fatal(services, monkeypatch):
    import lnp_rag.ingest.pipeline as pipeline

    def slow(*a, **k):
        time.sleep(5)

    monkeypatch.setattr(pipeline, "_process_pdf", slow)
    with pytest.raises(pipeline.PdfTimeout):
        pipeline.process_pdf("x.pdf", "pid", ["x.pdf"], {"timeout_s": 1})
