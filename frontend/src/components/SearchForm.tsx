import { useState, type FormEvent } from "react";
import type { AgentSearchRequest } from "../api";
import { DEFAULT_FOLDERS } from "../App";

interface Props {
  folderCounts: Record<string, number>;
  llmAvailable: boolean;
  running: boolean;
  initialQuery: string;
  onSearch: (req: AgentSearchRequest) => void;
  onCancel: () => void;
}

export default function SearchForm({ folderCounts, llmAvailable, running, initialQuery, onSearch, onCancel }: Props) {
  const [query, setQuery] = useState(initialQuery);
  const [folders, setFolders] = useState<string[]>(DEFAULT_FOLDERS);
  const [useCritic, setUseCritic] = useState(true);
  const [excludeReviews, setExcludeReviews] = useState(false);
  const [yearMin, setYearMin] = useState("");
  const [yearMax, setYearMax] = useState("");

  // Show the configured folders first, then any other indexed folder (e.g. uploads).
  const allFolders = [...DEFAULT_FOLDERS, ...Object.keys(folderCounts).filter((f) => !DEFAULT_FOLDERS.includes(f))];
  const tooShort = query.trim().length < 3;

  function toggle(f: string) {
    setFolders((cur) => (cur.includes(f) ? cur.filter((x) => x !== f) : [...cur, f]));
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    if (tooShort || folders.length === 0 || running) return;
    onSearch({
      query: query.trim(),
      folders: allFolders.filter((f) => folders.includes(f)),
      use_critic: useCritic,
      filters: {
        exclude_reviews: excludeReviews,
        year_min: yearMin ? Number(yearMin) : null,
        year_max: yearMax ? Number(yearMax) : null,
      },
    });
  }

  return (
    <form className="card search" onSubmit={submit}>
      <label htmlFor="query" className="label">What should the papers contain?</label>
      <textarea
        id="query"
        rows={3}
        value={query}
        placeholder="e.g. quantitative cellular uptake of lipid nanoparticles at multiple time points, with histograms or curves"
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
        }}
        disabled={running}
      />

      <div className="row">
        <span className="label small">Folders (one retriever + critic agent each)</span>
        <div className="chips">
          {allFolders.map((f) => (
            <button
              type="button"
              key={f}
              className={`chip ${folders.includes(f) ? "on" : ""}`}
              aria-pressed={folders.includes(f)}
              onClick={() => toggle(f)}
              disabled={running}
            >
              {f}
              {folderCounts[f] != null && <span className="count">{folderCounts[f]}</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="options">
        <label className="check" title={llmAvailable ? "" : "No LLM key configured: verdicts will be heuristic"}>
          <input type="checkbox" checked={useCritic} onChange={(e) => setUseCritic(e.target.checked)} disabled={running} />
          LLM critic{!llmAvailable && <span className="muted"> (no key, heuristic)</span>}
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={excludeReviews}
            onChange={(e) => setExcludeReviews(e.target.checked)}
            disabled={running}
          />
          Exclude review articles
        </label>
        <label className="years">
          Years
          <input inputMode="numeric" placeholder="from" value={yearMin} onChange={(e) => setYearMin(e.target.value.replace(/\D/g, ""))} disabled={running} />
          –
          <input inputMode="numeric" placeholder="to" value={yearMax} onChange={(e) => setYearMax(e.target.value.replace(/\D/g, ""))} disabled={running} />
        </label>

        <div className="actions">
          {running ? (
            <button type="button" className="btn ghost" onClick={onCancel}>Cancel</button>
          ) : (
            <span className="muted hint">⌘/Ctrl + Enter</span>
          )}
          <button type="submit" className="btn primary" disabled={running || tooShort || folders.length === 0}>
            {running ? "Searching…" : "Find papers"}
          </button>
        </div>
      </div>
    </form>
  );
}
