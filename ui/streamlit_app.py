"""Streamlit front end for the paper-search API.

    streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import html
import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
EXAMPLE_QUERY = (
    "Find research papers containing quantitative cellular uptake data for lipid nanoparticles (LNPs), "
    "especially histograms or curves showing uptake over time. Prioritize studies with multiple time "
    "points and clear experimental conditions."
)
STRENGTH_COLORS = {"strong": "green", "partial": "orange", "insufficient": "red"}

st.set_page_config(page_title="LNP Paper Search", page_icon=":material/biotech:", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
    except httpx.TransportError:
        st.error(f"Cannot reach the API at {API_URL}. Start it with `uvicorn lnp_rag.api.main:app`.")
        st.stop()
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


# ---------------------------------------------------------------------------
# Sidebar: library management
# ---------------------------------------------------------------------------
health = api("GET", "/health")
stats = api("GET", "/stats")

with st.sidebar:
    st.header("Library")
    st.caption(
        f"{stats['papers']} papers · {stats['chunks']} chunks · embeddings: "
        f"`{health['embedding_model']}` · LLM: "
        f"{health['llm_provider'] + ' `' + health['llm_model'] + '`' if health['llm_enabled'] else 'off'}"
    )
    folder = st.text_input("PDF folder", value=stats.get("papers_dir", ""))
    c1, c2 = st.columns(2)
    if c1.button("Index Papers", type="primary", use_container_width=True, disabled=health["indexing"]):
        try:
            api("POST", "/index", json={"papers_dir": folder or None, "rebuild": False})
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    with c2.popover("Rebuild / clear", use_container_width=True):
        st.write("Rebuild re-processes every PDF. Clear deletes the index (PDFs are untouched).")
        if st.button("Rebuild index", disabled=health["indexing"]):
            api("POST", "/index", json={"papers_dir": folder or None, "rebuild": True})
            st.rerun()
        if st.button("Clear index", disabled=health["indexing"]):
            api("DELETE", "/index")
            st.rerun()

    @st.fragment(run_every="2s" if health["indexing"] else None)
    def index_progress():
        status = api("GET", "/index/status")
        if status["state"] in ("idle",):
            return
        total = max(status["total_files"], 1)
        done = status["processed_files"]
        if status["state"] not in ("done", "error"):
            st.progress(min(done / total, 1.0), text=f"{status['state']}: {done}/{status['total_files']} PDFs")
            st.caption(status["message"])
        elif status["state"] == "done":
            st.success(
                f"{status['message']} - new {status['new']}, modified {status['modified']}, "
                f"deleted {status['deleted']}, unchanged {status['unchanged']}"
            )
        else:
            st.error(status["message"])
        if status["failed"]:
            with st.expander(f"{len(status['failed'])} files failed"):
                st.write(status["failed"])
        if status["state"] in ("done", "error") and health["indexing"]:
            st.rerun()  # refresh stats once indexing finishes

    index_progress()

    st.divider()
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Save uploads"):
        files = [("files", (u.name, u.getvalue(), "application/pdf")) for u in uploads]
        res = api("POST", "/upload", files=files, timeout=300)
        st.success(f"Saved {len(res['saved'])} file(s) to {res['folder']}. Click Index Papers to add them.")
        if res["rejected"]:
            st.warning(res["rejected"])

# ---------------------------------------------------------------------------
# Main: search
# ---------------------------------------------------------------------------
st.title("Research paper search")
st.caption("Hybrid semantic + keyword retrieval with cross-encoder reranking. Every explanation cites PDF excerpts.")

with st.form("search"):
    query = st.text_area("What evidence are you looking for?", value=EXAMPLE_QUERY, height=100)
    c1, c2, c3 = st.columns([1, 1, 2])
    top_k = c1.number_input("Number of results", 1, 50, 10)
    synthesize = c2.toggle("LLM explanations", value=health["llm_enabled"], disabled=not health["llm_enabled"])
    with st.expander("Filters"):
        f1, f2, f3, f4 = st.columns(4)
        in_vitro = f1.checkbox("In vitro")
        in_vivo = f1.checkbox("In vivo")
        has_figures = f2.checkbox("Has figures/tables")
        has_quant = f2.checkbox("Has quantitative data")
        exclude_reviews = f3.checkbox("Exclude review articles")
        folders = ["(any)"] + list(stats.get("folders", {}).keys())
        folder_filter = f3.selectbox("Folder", folders)
        year_min = f4.number_input("Year from", 0, 2100, 0, help="0 = no limit")
        year_max = f4.number_input("Year to", 0, 2100, 0, help="0 = no limit")
        title_contains = st.text_input("Title contains")
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    payload = {
        "query": query,
        "top_k": int(top_k),
        "synthesize": bool(synthesize),
        "filters": {
            "in_vitro": in_vitro or None,
            "in_vivo": in_vivo or None,
            "has_figures": has_figures or None,
            "has_quantitative": has_quant or None,
            "exclude_reviews": exclude_reviews,
            "folder": None if folder_filter == "(any)" else folder_filter,
            "year_min": year_min or None,
            "year_max": year_max or None,
            "title_contains": title_contains or None,
        },
    }
    with st.spinner("Searching..."):
        try:
            st.session_state["response"] = api("POST", "/search", json=payload, timeout=300)
        except RuntimeError as exc:
            st.error(str(exc))
            st.session_state.pop("response", None)

resp = st.session_state.get("response")
if resp:
    for w in resp["warnings"]:
        st.warning(w)
    with st.expander("Query expansion and timings"):
        st.write({"variants": resp["expanded_queries"], "candidate papers": resp["total_candidates"],
                  "timings_ms": resp["timings_ms"]})
    if not resp["results"]:
        st.info("No results.")

    for r in resp["results"]:
        with st.container(border=True):
            head, score = st.columns([5, 1])
            head.markdown(f"### {r['rank']}. {html.escape(r['title'])}")
            score.metric("Relevance", f"{r['relevance_score']:.2f}")
            authors = ", ".join(r["authors"][:6]) + (" et al." if len(r["authors"]) > 6 else "")
            doi = f"[{r['doi']}](https://doi.org/{r['doi']})" if r["doi"] else "n/a"
            st.markdown(
                f"{authors or '_authors not detected_'} · {r['year'] or 'year n/a'} · DOI: {doi}  \n"
                f"`{r['filename']}`"
            )
            badges = [
                f":blue-badge[{r['study_type']}]",
                f":{STRENGTH_COLORS.get(r['evidence_strength'], 'gray')}-badge[evidence: {r['evidence_strength']}]",
            ]
            if r["is_review"]:
                badges.append(":gray-badge[review article]")
            source = {"openai": "OpenAI", "claude": "Claude"}.get(r["explanation_source"], "automatic")
            badges.append(f":gray-badge[{source} explanation]")
            st.markdown(" ".join(badges))
            st.markdown(f"**Why it matches:** {r['explanation']}")
            if r["missing_evidence"]:
                st.markdown(f"**Not shown in excerpts:** {r['missing_evidence']}")
            st.markdown(
                f"**Pages:** {', '.join(map(str, r['pages']))} · **Sections:** {'; '.join(r['sections'][:5])}"
            )
            st.link_button(f"Open PDF at page {r['pages'][0] if r['pages'] else 1}", r["pdf_url"])

            cited = set(r["cited_evidence_ids"])
            with st.expander(f"Supporting excerpts ({len(r['evidence'])})"):
                for e in r["evidence"]:
                    mark = " · **cited**" if e["evidence_id"] in cited else ""
                    st.markdown(
                        f"**[{e['evidence_id']}]** p. {e['page']} · {e['section']} · "
                        f"_{e['chunk_type'].replace('_', ' ')}_ · score {e['score']:.2f}{mark} · "
                        f"[open page]({e['pdf_url']})"
                    )
                    st.markdown("> " + html.escape(e["text"]).replace("\n", "\n> "))
                    feats = e["features"]
                    if feats["timepoints"]:
                        st.caption(f"Time points detected: {', '.join(feats['timepoints'])}")
            if r["captions"]:
                with st.expander(f"Figure / table captions ({len(r['captions'])})"):
                    for e in r["captions"]:
                        st.markdown(f"**{e['label'] or e['section']}** (p. {e['page']}) · [open page]({e['pdf_url']})")
                        st.markdown("> " + html.escape(e["text"]))
