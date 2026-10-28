import { useEffect, useRef, useState } from "react";
import { agentSearch, getHealth, getStats, type AgentSearchRequest, type AgentSearchResponse, type Health } from "./api";
import SearchForm from "./components/SearchForm";
import FolderSummary from "./components/FolderSummary";
import Results from "./components/Results";

// The four folders searched by default (config/retrieval.yaml -> agents.folders).
export const DEFAULT_FOLDERS = ["In vitro", "In vitro + in vivo", "Unclassified", "Cellular_Uptake_Papers"];

const LAST_RESULT_KEY = "lnp-finder:last-result";

function loadLastResult(): AgentSearchResponse | null {
  try {
    const raw = sessionStorage.getItem(LAST_RESULT_KEY);
    return raw ? (JSON.parse(raw) as AgentSearchResponse) : null;
  } catch {
    return null;
  }
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [folderCounts, setFolderCounts] = useState<Record<string, number>>({});
  const [result, setResult] = useState<AgentSearchResponse | null>(loadLastResult);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState<{ req: AgentSearchRequest; started: number } | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((e: Error) => setHealthError(e.message));
    getStats().then((s) => setFolderCounts(s.folders)).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setElapsed(Date.now() - running.started), 500);
    return () => window.clearInterval(id);
  }, [running]);

  async function runSearch(req: AgentSearchRequest) {
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setElapsed(0);
    setRunning({ req, started: Date.now() });
    try {
      const resp = await agentSearch(req, controller.signal);
      setResult(resp);
      try {
        sessionStorage.setItem(LAST_RESULT_KEY, JSON.stringify(resp));
      } catch {
        /* storage full or blocked: the result still shows */
      }
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError((e as Error).message);
    } finally {
      setRunning(null);
      abortRef.current = null;
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>◎</span>
          <div>
            <h1>LNP Paper Finder</h1>
            <p className="muted">Every relevant paper in your library, checked by a critic agent per folder</p>
          </div>
        </div>
        <StatusPill health={health} error={healthError} />
      </header>

      <main>
        <SearchForm
          folderCounts={folderCounts}
          llmAvailable={!!health?.llm_provider}
          running={!!running}
          initialQuery={result?.query ?? ""}
          onSearch={runSearch}
          onCancel={() => abortRef.current?.abort()}
        />

        {running && <RunningPanel req={running.req} elapsed={elapsed} />}
        {error && !running && (
          <div className="banner error" role="alert">
            <strong>Search failed.</strong> {error}
          </div>
        )}

        {result && !running && (
          <>
            {result.warnings.length > 0 && (
              <div className="banner warn">
                {result.warnings.map((w) => (
                  <div key={w}>{w}</div>
                ))}
              </div>
            )}
            <FolderSummary result={result} />
            <Results key={result.query + result.timings_ms.total} result={result} />
          </>
        )}

        {!result && !running && !error && (
          <div className="empty">
            <p>Describe what the papers should contain, e.g.</p>
            <p className="example">
              “Quantitative cellular uptake of LNPs measured by flow cytometry at several time points”
            </p>
            <p className="muted">
              Four retriever agents scan every chunk (text, figure and table captions) of their folder, and
              four critic agents check each candidate. There is no top-k: all accepted papers are returned.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

function StatusPill({ health, error }: { health: Health | null; error: string | null }) {
  if (error) return <span className="pill bad" title={error}>API offline</span>;
  if (!health) return <span className="pill">Connecting…</span>;
  return (
    <span className="pill ok" title={health.llm_model ?? "no LLM key: heuristic critic"}>
      {health.indexed_papers.toLocaleString()} papers indexed · critic: {health.llm_provider ?? "heuristic"}
      {health.indexing && " · indexing…"}
    </span>
  );
}

function RunningPanel({ req, elapsed }: { req: AgentSearchRequest; elapsed: number }) {
  const s = Math.floor(elapsed / 1000);
  const clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  return (
    <section className="card running" aria-live="polite">
      <div className="running-head">
        <span className="spinner" aria-hidden />
        <div>
          <strong>Agents are searching… {clock}</strong>
          <p className="muted">
            Searching every folder usually takes 3–5 minutes: the retrievers rescore candidate passages, then the
            critics review each candidate{req.use_critic ? " with the LLM" : ""}.
          </p>
        </div>
      </div>
      <ul className="agent-list">
        {req.folders.map((f) => (
          <li key={f}>
            <span className="dot pulse" /> <strong>{f}</strong>
            <span className="muted"> retriever → critic</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
