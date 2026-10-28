import type { AgentSearchResponse } from "../api";

const secs = (ms?: number) => (ms == null ? "–" : `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`);

export default function FolderSummary({ result }: { result: AgentSearchResponse }) {
  const totalRejected = result.folders.reduce((n, f) => n + f.rejected.length, 0);
  return (
    <section className="summary">
      <div className="headline">
        <div>
          <span className="big">{result.accepted.length}</span> papers accepted
          <span className="muted"> · {totalRejected} rejected by critics · {secs(result.timings_ms.total)}</span>
        </div>
        {result.expanded_queries.length > 0 && (
          <details className="variants">
            <summary>Query variants used</summary>
            <ul>
              {result.expanded_queries.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ul>
          </details>
        )}
      </div>

      <div className="folder-grid">
        {result.folders.map((f) => (
          <div className="folder-card" key={f.folder}>
            <div className="folder-name">{f.folder}</div>
            <div className="funnel">
              <Stat n={f.papers_in_folder} label="papers" />
              <span className="arrow">→</span>
              <Stat n={f.candidates + f.near_misses} label="to critic" title={`${f.candidates} candidates + ${f.near_misses} near-misses`} />
              <span className="arrow">→</span>
              <Stat n={f.accepted.length} label="accepted" tone="ok" />
            </div>
            <div className="muted small">
              {f.chunks_scanned.toLocaleString()} chunks scanned · {f.near_misses} near-misses · {f.dropped} weak
              matches dropped
            </div>
            <div className="muted small">
              retriever {secs(f.timings_ms.retriever)} · critic {secs(f.timings_ms.critic)}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Stat({ n, label, tone, title }: { n: number; label: string; tone?: "ok"; title?: string }) {
  return (
    <div className={`stat ${tone ?? ""}`} title={title}>
      <span className="n">{n}</span>
      <span className="l">{label}</span>
    </div>
  );
}
