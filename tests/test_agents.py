"""Multi-agent folder search: folder isolation, cross-folder de-duplication,
no top-k truncation, critic verdict grounding, HTTP endpoint."""
import shutil

from fastapi.testclient import TestClient

from lnp_rag.agents.critic_agent import apply_verdict
from lnp_rag.api.main import create_app
from lnp_rag.config import AgentConfig, RetrievalConfig
from lnp_rag.models import AgentSearchRequest, Verdict
from lnp_rag.services import Services

from .conftest import UPTAKE_PAPER, FakeEmbedder, make_pdf

QUERY = "cellular uptake of lipid nanoparticles over time by flow cytometry"
FOLDERS = ["Cellular_Uptake", "Formulation"]


class FakeCritic:
    """Accepts papers whose title mentions uptake, quoting the first excerpt."""

    name = "fake-critic"
    model = "fake"

    def __init__(self):
        self.batches: list[list[dict]] = []

    def review(self, query, folder, papers):
        self.batches.append(papers)
        out = {}
        for p in papers:
            ev = p["evidence"][0]
            ok = "uptake" in p["title"].lower()
            out[p["key"]] = {
                "paper_key": p["key"],
                "verdict": "ACCEPT" if ok else "REJECT",
                "reason": f"Uptake measured [{ev['id']}]." if ok else "Not about uptake.",
                "cited_evidence_ids": [ev["id"]] if ok else [],
                "quotes": [{"evidence_id": ev["id"], "quote": ev["text"][:40]}] if ok else [],
            }
        return out


def _services(settings, agents: AgentConfig | None = None, critic=None) -> Services:
    cfg = RetrievalConfig(agents=agents or AgentConfig(folders=FOLDERS))
    svc = Services(settings, cfg)
    svc.__dict__["embedder"] = FakeEmbedder()
    svc.__dict__["llm"] = None
    svc.__dict__["critic_llm"] = critic
    status = svc.indexer.run()
    assert status.state == "done", status.message
    return svc


def test_each_agent_only_sees_its_folder(settings):
    svc = _services(settings, critic=FakeCritic())
    resp = svc.agents.search(AgentSearchRequest(query=QUERY))
    assert [r.folder for r in resp.folders] == FOLDERS
    for report in resp.folders:
        assert report.papers_in_folder == 1
        for p in report.accepted + report.rejected:
            assert report.folder in p.folder_labels
    uptake = next(r for r in resp.folders if r.folder == "Cellular_Uptake")
    assert [p.filename for p in uptake.accepted] == ["uptake_kinetics.pdf"]
    top = uptake.accepted[0]
    assert top.critic == "fake-critic" and top.reason and top.quotes and top.cited_evidence_ids == ["E1"]
    assert top.caption_match  # the Figure 2 caption reaches the critic
    assert all(p.filename != "uptake_kinetics.pdf" for p in resp.accepted[1:])


def test_paper_in_two_folders_is_merged(settings, papers_dir):
    shutil.copy(papers_dir / "Cellular_Uptake" / "uptake_kinetics.pdf", papers_dir / "Formulation" / "copy.pdf")
    svc = _services(settings, critic=FakeCritic())
    resp = svc.agents.search(AgentSearchRequest(query=QUERY))
    per_folder = [p.paper_id for r in resp.folders for p in r.accepted]
    assert len(per_folder) == 2 and len(set(per_folder)) == 1  # both agents found it
    assert len(resp.accepted) == 1
    merged = resp.accepted[0]
    assert sorted(merged.folder_labels) == sorted(FOLDERS)
    assert sorted(merged.found_by) == sorted(FOLDERS)


def test_no_top_k_truncation(settings, papers_dir):
    n = 27
    for i in range(n):
        paper = {**UPTAKE_PAPER, "title": f"Cellular uptake of lipid nanoparticles, cohort {i}"}
        make_pdf(papers_dir / "Cellular_Uptake" / f"uptake_{i:02d}.pdf", paper)
    critic = FakeCritic()
    svc = _services(settings, AgentConfig(folders=["Cellular_Uptake"], critic_batch_size=4), critic)
    resp = svc.agents.search(AgentSearchRequest(query=QUERY))
    report = resp.folders[0]
    assert report.papers_in_folder == n + 1
    assert len(report.accepted) == n + 1 == len(resp.accepted)
    assert sum(len(b) for b in critic.batches) == n + 1 and max(len(b) for b in critic.batches) <= 4


def test_heuristic_critic_without_llm(settings):
    svc = _services(settings)
    resp = svc.agents.search(AgentSearchRequest(query=QUERY))
    assert any("heuristic" in w for w in resp.warnings)
    assert {p.critic for r in resp.folders for p in r.accepted + r.rejected} == {"heuristic"}
    assert resp.accepted and resp.accepted[0].filename == "uptake_kinetics.pdf"


def test_verdict_keeps_only_real_evidence(settings):
    svc = _services(settings, critic=FakeCritic())
    paper = svc.agents.search(AgentSearchRequest(query=QUERY)).accepted[0].model_copy(deep=True)
    text = paper.evidence[0].text
    item = {
        "verdict": "ACCEPT",
        "reason": "Shown in [P1-E1] and [P1-E99].",
        "cited_evidence_ids": ["P1-E1", "P1-E99"],
        "quotes": [{"evidence_id": "P1-E1", "quote": " ".join(text.split()[:6])},
                   {"evidence_id": "P1-E1", "quote": "text the paper never says"}],
    }
    out = apply_verdict(paper.model_copy(deep=True), item, "P1", "x")
    assert out.verdict == Verdict.ACCEPT and out.cited_evidence_ids == ["E1"]
    assert len(out.quotes) == 1 and out.reason == "Shown in [E1] and [E99]."

    fabricated = {"verdict": "ACCEPT", "reason": "Trust me.", "cited_evidence_ids": ["P1-E99"], "quotes": []}
    out = apply_verdict(paper.model_copy(deep=True), fabricated, "P1", "x")
    assert out.verdict == Verdict.REJECT and "no evidence" in out.reason


def test_agents_endpoint(settings):
    svc = _services(settings, critic=FakeCritic())
    client = TestClient(create_app(svc, warmup=False))
    r = client.post("/agents/search", json={"query": QUERY})
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"][0]["verdict"] == "ACCEPT"
    assert {f["folder"] for f in body["folders"]} == set(FOLDERS)
    assert all("rejected" in f for f in body["folders"])
