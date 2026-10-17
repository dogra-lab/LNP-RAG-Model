# LNP-RAG-Model

Retrieves the top 100 papers from the local PDF library (`papers/`) for a query about a step of the
LNP delivery journey, then screens each candidate with an LLM (OpenAI or Claude) using the LNP screening prompt
(`lnp_rag/screen.py::_system_prompt`).

## Pipeline

1. **Ingest** (`ingest.py`, run once): every PDF is parsed with PyMuPDF into title, abstract,
   figure/table captions, Results section and full-text chunks. Identical files in several folders
   are deduplicated. Review articles are flagged heuristically. Chunks are embedded with
   `BAAI/bge-base-en-v1.5` and indexed with BM25.
2. **Retrieve** (`search.py`): hybrid search (dense + BM25, reciprocal-rank fusion at paper level)
   picks the top `--candidates` papers (default 200).
3. **Screen**: each candidate goes to the LLM (`gpt-5-mini` by default with `OPENAI_API_KEY`,
   or `claude-opus-5-5` with `--provider anthropic`) with the system prompt built from
   the step, and a user message with the title, abstract, captions and the Results excerpts most
   relevant to the query. Review articles are marked `REVIEW ARTICLE`. The model returns
   `score / verdict / study_type / rationale` as schema-validated JSON.
4. **Rank**: include > borderline > exclude, then by score. The top `--top` (default 100) are
   printed and saved to `results/<timestamp>_<query>.csv` and `.json`.

## Validated papers (`paper used/`)

`paper used/` holds the curated in-vitro papers. Supplements (`P007B.pdf`, `P06B.pdf`, ...) are
merged into their main paper. Every search returns all of them in a "validated" section, ordered
by relevance to the query and not re-screened. The additional results come from the library with
copies of validated papers removed (same title, or title+abstract similarity >= 0.94). Library
papers are also ranked by how similar they are to the validated set, to surface papers you may be
missing (turn off with `--no-validated-similarity` or the checkbox in the UI). Re-index after
changing the folder: `.venv/bin/python ingest.py --only-validated` (about a minute).

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
export OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY with --provider anthropic
.venv/bin/python ingest.py            # builds ./index (re-run when papers change)
```

## Usage

```bash
# Preset step (see lnp_rag/steps.py), kinetic data required
.venv/bin/python search.py "LNP internalization kinetics" --step cellular_uptake --strict-kinetic

# Custom step
.venv/bin/python search.py "ApoE adsorption on ionizable LNPs" \
    --step-name "Protein corona" --step-description "plasma protein adsorption onto LNPs" \
    --guidance "Count ApoE/albumin binding measured over time as kinetic evidence."

# Retrieval only (no API calls, free)
.venv/bin/python search.py "endosomal escape efficiency" --no-llm
```

Useful flags: `--only-relevant` (drop `exclude`), `--candidates N`, `--top N`, `--provider`, `--model`,
`--effort low|medium|high`, `--workers N`.

Screening verdicts are cached in `index/screen_cache.jsonl`, so repeating a query with the same
step settings costs nothing. With the defaults, a fresh query screens 200 papers at roughly
4-5k input tokens each.

## Web app (React)

```bash
cd frontend && npm install && npm run build && cd ..   # build the React UI once (re-run after UI changes)
.venv/bin/uvicorn app:app --port 8000                   # then open http://localhost:8000
```

The page shows the top papers with verdict, LLM score, study type, review flag and rationale.
You can filter by verdict, study type or review status, filter by text, re-sort, download a CSV,
and click a title to open the PDF. Each search is also saved to `results/`.

For UI development with hot reload, run `npm run dev` in `frontend/` (port 5173, proxies `/api` to
port 8000) alongside uvicorn.
