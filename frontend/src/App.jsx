import { useEffect, useMemo, useRef, useState } from "react";

const VERDICTS = ["include", "borderline", "exclude", "error"];
const STUDY_TYPES = { in_vitro: "In vitro", in_vivo: "In vivo", both: "In vitro + in vivo", unclassified: "Unclassified" };
const IN_VITRO_ANY = "in_vitro_any"; // filter value: in_vitro or both
const DATA_TYPES = {
  time_course: "Time course",
  dose_response: "Dose response",
  uptake_quantification: "Uptake quantified",
  intracellular_accumulation: "Intracellular accumulation",
  endosomal_escape: "Endosomal escape",
  uptake_pathway: "Uptake pathway",
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function downloadCsv(rows, cols, filename) {
  const blob = new Blob([toCsv(rows, cols)], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function toCsv(rows, cols) {
  const esc = (v) => `"${String(v ?? "").replaceAll('"', '""')}"`;
  return [cols.join(","), ...rows.map((r, i) => cols.map((c) => esc(c === "rank" ? i + 1 : r[c])).join(","))].join("\n");
}

// Every search screens for this step; edit lnp_rag/steps.py to change its definition.
const SEARCH_DEFAULTS = { step: "uptake_pkpd", strict_kinetic: false, top: 100, candidates: 500 };

export default function App() {
  const [config, setConfig] = useState(null);
  const [configError, setConfigError] = useState("");
  const [query, setQuery] = useState("");
  const [job, setJob] = useState(null); // {status, done, total, error, result}
  const pollRef = useRef(null);

  useEffect(() => {
    api("/api/config")
      .then(setConfig)
      .catch((e) => setConfigError(e.message));
    return () => clearInterval(pollRef.current);
  }, []);

  const running = job && !["done", "error"].includes(job.status);
  const llmAvailable = config && (config.providers.openai || config.providers.anthropic);
  const canSearch = Boolean(config) && !running;
  const step = config?.steps.find((s) => s.id === SEARCH_DEFAULTS.step);
  const model = config?.default_provider && config.default_models[config.default_provider];

  async function submit(e) {
    e.preventDefault();
    if (!canSearch) return;
    const body = { ...SEARCH_DEFAULTS, query, no_llm: !llmAvailable };
    setJob({ status: "queued", done: 0, total: 0 });
    try {
      const { job_id } = await api("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const j = await api(`/api/jobs/${job_id}`);
          setJob(j);
          if (j.status === "done" || j.status === "error") clearInterval(pollRef.current);
        } catch (err) {
          clearInterval(pollRef.current);
          setJob({ status: "error", error: err.message });
        }
      }, 1000);
    } catch (err) {
      setJob({ status: "error", error: err.message });
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <Logo />
          <div>
            <div className="brand-name">LNP Literature Console</div>
            <div className="brand-sub">Lipid nanoparticle delivery · evidence screening</div>
          </div>
        </div>
        <div className="sys">
          <Indicator state={configError ? "bad" : config ? "ok" : "wait"} label="API" />
          <Indicator state={!config ? "wait" : llmAvailable ? "ok" : "warn"} label="LLM screening" />
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <Panel title="Screening protocol">
            <dl className="kv">
              <dt>Step</dt>
              <dd>{step?.name ?? "Cellular uptake (in vitro)"}</dd>
              <dt>Criteria</dt>
              <dd>{config?.learned_criteria ? "Learned from your papers" : "Step guidance only"}</dd>
              <dt>Pipeline</dt>
              <dd className="mono">
                {SEARCH_DEFAULTS.candidates} retrieved → top {SEARCH_DEFAULTS.top}
              </dd>
            </dl>
            {step?.description && <p className="panel-note">{step.description}</p>}
            {config && !config.learned_criteria && (
              <p className="panel-note">
                Run <code>python learn.py</code> to learn inclusion criteria from your papers in “paper used”.
              </p>
            )}
          </Panel>

          <Panel title="Corpus">
            <div className="metrics single">
              <Metric
                value={config ? (config.paper_count + config.validated_count).toLocaleString() : "—"}
                label="Papers"
              />
            </div>
          </Panel>

          <Panel title="Screening model">
            {config && llmAvailable ? (
              <dl className="kv">
                <dt>Provider</dt>
                <dd className="cap">{config.default_provider}</dd>
                <dt>Model</dt>
                <dd className="mono">{model}</dd>
              </dl>
            ) : (
              <p className="panel-note">
                {config ? (
                  <>
                    No API key found. Add <code>OPENAI_API_KEY</code> to <code>.env</code> to enable LLM screening.
                    Searches run retrieval only.
                  </>
                ) : (
                  "Connecting…"
                )}
              </p>
            )}
          </Panel>
        </aside>

        <main className="main">
          {configError && (
            <div className="alert error">Can't reach the API server: {configError}. Is uvicorn running?</div>
          )}

          <form className="card console" onSubmit={submit}>
            <label className="field-label" htmlFor="query">
              Research query
            </label>
            <div className="query-row">
              <input
                id="query"
                className="query"
                placeholder="e.g. time course of LNP internalization measured by flow cytometry"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                autoFocus
              />
              <button className="primary" type="submit" disabled={!canSearch}>
                {running ? "Running…" : "Run screen"}
              </button>
            </div>
            <p className="hint">
              Leave blank to screen for the protocol step alone. Results are saved to <code>results/</code>.
            </p>
          </form>

          {job && <Status job={job} />}
          {!job && !configError && <EmptyState />}
          {job?.status === "done" && <Summary output={job.result} saved={job.saved} job={job} />}
          {job?.status === "done" && <Results output={job.result} />}
        </main>
      </div>
    </div>
  );
}

function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="13" fill="none" stroke="currentColor" strokeWidth="2" />
      <circle cx="16" cy="16" r="8.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="2.5 2" />
      <circle cx="16" cy="16" r="3.5" fill="currentColor" />
    </svg>
  );
}

function Indicator({ state, label }) {
  return (
    <span className={`indicator ${state}`}>
      <span className="dot" />
      {label}
    </span>
  );
}

function Panel({ title, children }) {
  return (
    <section className="panel">
      <h3 className="panel-title">{title}</h3>
      {children}
    </section>
  );
}

function Metric({ value, label, tone }) {
  return (
    <div className={`metric ${tone || ""}`}>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="card empty">
      <div className="empty-title">No screen run yet</div>
      <p>
        Enter a query to retrieve candidate papers from the library, screen them against the protocol with the LLM,
        and rank them by relevance.
      </p>
    </div>
  );
}

const STAGES = [
  { key: "retrieving", label: "Retrieve", sub: "Embedding search" },
  { key: "screening", label: "Screen", sub: "LLM review" },
  { key: "done", label: "Rank", sub: "Results ready" },
];

function Status({ job }) {
  if (job.status === "error") return <div className="alert error">Search failed: {job.error}</div>;
  if (job.status === "done") {
    return job.result.meta.errors ? (
      <div className="alert warn">{job.result.meta.errors} papers failed screening; see their rationale.</div>
    ) : null;
  }
  const current = STAGES.findIndex((s) => s.key === job.status);
  const pct = job.total ? Math.round((100 * job.done) / job.total) : 0;
  return (
    <div className="card status">
      <ol className="stepper">
        {STAGES.map((s, i) => (
          <li key={s.key} className={i < current ? "complete" : i === current ? "active" : ""}>
            <span className="step-num">{i + 1}</span>
            <span>
              <span className="step-label">{s.label}</span>
              <span className="step-sub">{s.sub}</span>
            </span>
          </li>
        ))}
      </ol>
      <div className="status-line">
        <span>
          {job.status === "screening"
            ? "Screening papers with the LLM"
            : job.status === "retrieving"
            ? "Retrieving candidate papers…"
            : "Starting job…"}
        </span>
        {job.status === "screening" && (
          <span className="mono">
            {job.done} / {job.total} · {pct}%
          </span>
        )}
      </div>
      <div className="bar">
        <div className={job.status === "screening" ? "fill" : "fill indeterminate"} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function Summary({ output, saved, job }) {
  const { results, validated, meta } = output;
  const count = (v) => results.filter((r) => r.verdict === v).length;
  return (
    <section className="summary">
      <div className="metrics wide">
        <Metric value={validated.length + results.length} label="Papers ranked" />
        {meta.model && <Metric value={validated.length + count("include")} label="Include" tone="include" />}
        {meta.model && <Metric value={count("borderline")} label="Borderline" tone="borderline" />}
        {meta.model && <Metric value={count("exclude")} label="Exclude" tone="exclude" />}
      </div>
      {saved && (
        <p className="hint">
          Saved as <code>{saved}</code>
          {job.included_dir && (
            <>
              {" "}· PDFs of {job.n_included} included papers copied to <code>{job.included_dir}/</code>
            </>
          )}
        </p>
      )}
    </section>
  );
}

// Papers from "paper used/" are already validated: they are listed first as "include" and are not re-screened.
function Results({ output }) {
  const { results, validated, meta } = output;
  const screened = Boolean(meta.model);
  const [verdicts, setVerdicts] = useState(new Set(VERDICTS));
  const [studyType, setStudyType] = useState("");
  const [dataType, setDataType] = useState("");
  const [reviews, setReviews] = useState("all");
  const [text, setText] = useState("");
  const [sort, setSort] = useState("rank");

  const counts = useMemo(() => {
    const c = Object.fromEntries(VERDICTS.map((v) => [v, 0]));
    results.forEach((r) => r.verdict && c[r.verdict]++);
    c.include += validated.length;
    return c;
  }, [results, validated]);

  const shown = useMemo(() => {
    const q = text.trim().toLowerCase();
    const known = validated.map((r) => ({ ...r, known: true, verdict: "include", folders: "" }));
    let rows = [...known, ...results].map((r, i) => ({ ...r, rank: i + 1 }));
    rows = rows.filter(
      (r) =>
        (!screened || verdicts.has(r.verdict)) &&
        (r.known ||
          ((!studyType ||
            r.study_type === studyType ||
            (studyType === IN_VITRO_ANY && ["in_vitro", "both"].includes(r.study_type))) &&
            (!dataType || (r.data_types || []).includes(dataType)))) &&
        (reviews === "all" || (reviews === "only") === r.is_review) &&
        (!q || `${r.label || ""} ${r.title} ${r.rationale || ""} ${r.abstract}`.toLowerCase().includes(q))
    );
    // Validated papers stay on top under every sort.
    const ref = (a, b) => Number(Boolean(b.known)) - Number(Boolean(a.known));
    if (sort === "score") rows.sort((a, b) => ref(a, b) || (b.score ?? 0) - (a.score ?? 0));
    if (sort === "retrieval")
      rows.sort((a, b) => ref(a, b) || (a.retrieval_rank ?? a.relevance_rank) - (b.retrieval_rank ?? b.relevance_rank));
    return rows;
  }, [results, validated, verdicts, studyType, dataType, reviews, text, sort, screened]);

  function toggle(v) {
    setVerdicts((s) => {
      const next = new Set(s);
      next.has(v) ? next.delete(v) : next.add(v);
      return next;
    });
  }

  function download() {
    const cols = ["rank", "title", "verdict", "score", "study_type", "data_types", "is_review", "rationale", "folders", "path"];
    downloadCsv(shown, cols, `${(meta.query || meta.step_name).slice(0, 50).replace(/[^a-z0-9]+/gi, "_")}_papers.csv`);
  }

  return (
    <section className="card section">
      <div className="section-head">
        <div>
          <h2>
            Papers <span className="count">{validated.length + results.length}</span>
          </h2>
          <p className="muted small">
            {meta.query && <>Query: “{meta.query}” · </>}
            Step: {meta.step_name}
            {meta.learned_criteria && " · criteria learned from your papers"}
            {meta.strict_kinetic && " · kinetic data required"}
            {screened ? ` · screened with ${meta.model}` : " · retrieval only (not screened)"}
          </p>
        </div>
        <button onClick={download}>Export CSV</button>
      </div>

      <div className="filters">
        {screened &&
          VERDICTS.filter((v) => counts[v] > 0).map((v) => (
            <button
              key={v}
              className={`chip ${v} ${verdicts.has(v) ? "on" : ""}`}
              onClick={() => toggle(v)}
              aria-pressed={verdicts.has(v)}
            >
              {v} <b>{counts[v]}</b>
            </button>
          ))}
        {screened && (
          <select value={studyType} onChange={(e) => setStudyType(e.target.value)}>
            <option value="">All study types</option>
            <option value={IN_VITRO_ANY}>In vitro (incl. in vitro + in vivo)</option>
            {Object.entries(STUDY_TYPES).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        )}
        {screened && (
          <select value={dataType} onChange={(e) => setDataType(e.target.value)}>
            <option value="">All data types</option>
            {Object.entries(DATA_TYPES).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        )}
        <select value={reviews} onChange={(e) => setReviews(e.target.value)}>
          <option value="all">Reviews + research</option>
          <option value="exclude">Research only</option>
          <option value="only">Reviews only</option>
        </select>
        <select value={sort} onChange={(e) => setSort(e.target.value)}>
          <option value="rank">Sort: overall rank</option>
          {screened && <option value="score">Sort: LLM score</option>}
          <option value="retrieval">Sort: retrieval rank</option>
        </select>
        <input className="filter-text" placeholder="Filter text…" value={text} onChange={(e) => setText(e.target.value)} />
        <span className="muted small mono">{shown.length} shown</span>
      </div>

      <ol className="papers">
        {shown.map((r) => (
          <Paper key={r.id} paper={r} screened={screened} />
        ))}
      </ol>
      {shown.length === 0 && <p className="muted pad">No papers match these filters.</p>}
    </section>
  );
}

function Paper({ paper, screened }) {
  const [open, setOpen] = useState(false);
  return (
    <li className={`paper ${screened || paper.known ? `v-${paper.verdict}` : ""}`}>
      <div className="rank">{paper.rank}</div>
      <div className="body">
        <a className="title" href={`/api/papers/${paper.id}/pdf`} target="_blank" rel="noreferrer">
          {paper.title}
        </a>
        <div className="tags">
          {paper.known && <span className="badge include">include</span>}
          {paper.files > 1 &&
            Array.from({ length: paper.files - 1 }, (_, i) => (
              <a
                key={i}
                className="tag link-tag"
                href={`/api/papers/${paper.id}/pdf?file=${i + 1}`}
                target="_blank"
                rel="noreferrer"
              >
                Supplementary PDF
              </a>
            ))}
          {screened && !paper.known && <span className={`badge ${paper.verdict}`}>{paper.verdict}</span>}
          {screened && !paper.known && paper.verdict !== "error" && (
            <span className="score" title="LLM relevance score">
              <span className="score-bar">
                <span style={{ width: `${Math.round(paper.score * 100)}%` }} />
              </span>
              {paper.score.toFixed(2)}
            </span>
          )}
          {screened && paper.study_type && <span className="tag">{STUDY_TYPES[paper.study_type]}</span>}
          {(paper.data_types || []).map((d) => (
            <span key={d} className="tag data">
              {DATA_TYPES[d] ?? d}
            </span>
          ))}
          {paper.is_review && <span className="tag review">Review</span>}
          {paper.folders && <span className="tag subtle">{paper.folders}</span>}
        </div>
        {screened && paper.rationale && <p className="rationale">{paper.rationale}</p>}
        {paper.abstract && (
          <>
            <button className="link small" onClick={() => setOpen((o) => !o)}>
              {open ? "Hide abstract" : "Show abstract"}
            </button>
            {open && <p className="abstract">{paper.abstract}</p>}
          </>
        )}
      </div>
    </li>
  );
}
