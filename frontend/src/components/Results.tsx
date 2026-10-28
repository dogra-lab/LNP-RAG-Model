import { useMemo, useState } from "react";
import type { AgentPaper, AgentSearchResponse } from "../api";
import PaperCard from "./PaperCard";

type Tab = "accepted" | "rejected";
type Sort = "score" | "year";

export default function Results({ result }: { result: AgentSearchResponse }) {
  const [tab, setTab] = useState<Tab>("accepted");
  const [folder, setFolder] = useState<string>("all");
  const [text, setText] = useState("");
  const [sort, setSort] = useState<Sort>("score");
  const [captionOnly, setCaptionOnly] = useState(false);

  const rejected = useMemo(() => result.folders.flatMap((f) => f.rejected.map((p) => ({ ...p, found_by: [f.folder] }))), [result]);
  const source: AgentPaper[] = tab === "accepted" ? result.accepted : rejected;

  const papers = useMemo(() => {
    const needle = text.trim().toLowerCase();
    const list = source.filter(
      (p) =>
        (folder === "all" || p.folder_labels.includes(folder) || p.found_by.includes(folder)) &&
        (!captionOnly || p.caption_match) &&
        (!needle || `${p.title} ${p.filename} ${p.reason}`.toLowerCase().includes(needle)),
    );
    return list.sort((a, b) =>
      sort === "year" ? (b.year ?? 0) - (a.year ?? 0) || b.retriever_score - a.retriever_score : b.retriever_score - a.retriever_score,
    );
  }, [source, folder, text, sort, captionOnly]);

  return (
    <section className="results">
      <div className="toolbar">
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === "accepted"} className={tab === "accepted" ? "on" : ""} onClick={() => setTab("accepted")}>
            Accepted <span className="count">{result.accepted.length}</span>
          </button>
          <button role="tab" aria-selected={tab === "rejected"} className={tab === "rejected" ? "on" : ""} onClick={() => setTab("rejected")}>
            Rejected <span className="count">{rejected.length}</span>
          </button>
        </div>
        <div className="filters">
          <select value={folder} onChange={(e) => setFolder(e.target.value)} aria-label="Folder">
            <option value="all">All folders</option>
            {result.folders.map((f) => (
              <option key={f.folder} value={f.folder}>{f.folder}</option>
            ))}
          </select>
          <select value={sort} onChange={(e) => setSort(e.target.value as Sort)} aria-label="Sort">
            <option value="score">Sort: retriever score</option>
            <option value="year">Sort: newest first</option>
          </select>
          <label className="check">
            <input type="checkbox" checked={captionOnly} onChange={(e) => setCaptionOnly(e.target.checked)} />
            Caption evidence
          </label>
          <input className="filter-text" placeholder="Filter results…" value={text} onChange={(e) => setText(e.target.value)} />
          <button className="btn ghost small" onClick={() => downloadCsv(papers, result.query, tab)} disabled={!papers.length}>
            CSV
          </button>
          <button className="btn ghost small" onClick={() => downloadJson(result)}>JSON</button>
        </div>
      </div>

      {papers.length === 0 ? (
        <p className="muted empty-list">No papers match these filters.</p>
      ) : (
        <ol className="paper-list">
          {papers.map((p, i) => (
            <PaperCard key={`${p.paper_id}-${p.found_by.join()}`} paper={p} rank={i + 1} />
          ))}
        </ol>
      )}
    </section>
  );
}

function download(name: string, type: string, body: string) {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

function slug(q: string) {
  return q.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 40).replace(/-$/, "");
}

function downloadCsv(papers: AgentPaper[], query: string, tab: Tab) {
  const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const header = ["verdict", "title", "filename", "year", "doi", "folders", "retriever_score", "route", "reason", "quotes", "pdf_url"];
  const rows = papers.map((p) =>
    [p.verdict, p.title, p.filename, p.year, p.doi, p.folder_labels.join("; "), p.retriever_score, p.route, p.reason,
      p.quotes.map((q) => `p.${q.page} ${q.label}: ${q.quote}`).join(" | "), p.pdf_url].map(esc).join(","),
  );
  download(`${tab}-${slug(query)}.csv`, "text/csv", [header.join(","), ...rows].join("\n"));
}

function downloadJson(result: AgentSearchResponse) {
  download(`agents-${slug(result.query)}.json`, "application/json", JSON.stringify(result, null, 2));
}
