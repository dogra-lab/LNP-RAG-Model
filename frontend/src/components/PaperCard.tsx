import { Fragment, useState, type ReactNode } from "react";
import type { AgentEvidence, AgentPaper } from "../api";

const TYPE_LABEL: Record<string, string> = {
  figure_caption: "Figure caption",
  table_caption: "Table caption",
  abstract: "Abstract",
  paragraph: "Text",
  title: "Title",
};

export default function PaperCard({ paper, rank }: { paper: AgentPaper; rank: number }) {
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState<string | null>(null);
  const byId = Object.fromEntries(paper.evidence.map((e) => [e.evidence_id, e]));
  const accepted = paper.verdict === "ACCEPT";

  function showEvidence(id: string) {
    setOpen(true);
    setHighlight(id);
    requestAnimationFrame(() =>
      document.getElementById(`${paper.paper_id}-${id}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  }

  return (
    <li className={`card paper ${accepted ? "accept" : "reject"}`}>
      <div className="paper-head">
        <span className="rank">{rank}</span>
        <div className="paper-title">
          <h3>{paper.title || paper.filename}</h3>
          <div className="meta">
            <span>{paper.filename}</span>
            {paper.year && <span>{paper.year}</span>}
            {paper.doi && (
              <a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer">doi:{paper.doi}</a>
            )}
          </div>
          <div className="badges">
            {paper.folder_labels.map((f) => (
              <span key={f} className="badge folder">{f}</span>
            ))}
            <span className={`badge ${accepted ? "ok" : "bad"}`}>{paper.verdict}</span>
            {paper.route === "near_miss" && (
              <span className="badge warn" title="Scored just under the retriever threshold; the critic re-checked it">
                near-miss {accepted ? "recovered" : ""}
              </span>
            )}
            {paper.caption_match && <span className="badge info">caption evidence</span>}
            {paper.found_by.length > 1 && <span className="badge">found in {paper.found_by.length} folders</span>}
            <span className="badge subtle">critic: {paper.critic}</span>
          </div>
        </div>
        <div className="paper-side">
          <div className="score" title="Retriever relevance (cross-encoder, boosted for captions/results)">
            {paper.retriever_score.toFixed(2)}
          </div>
          <a className="btn small" href={paper.pdf_url} target="_blank" rel="noreferrer">Open PDF</a>
        </div>
      </div>

      <div className="reason">
        <span className="reason-label">{accepted ? "Why it matches" : "Why rejected"}</span>
        <p>{withCitations(paper.reason, byId, showEvidence)}</p>
      </div>

      {paper.quotes.length > 0 && (
        <div className="quotes">
          {paper.quotes.map((q, i) => (
            <blockquote key={i}>
              “{q.quote}”
              <footer>
                <button className="cite" onClick={() => showEvidence(q.evidence_id)}>{q.evidence_id}</button>
                {q.label || "text"}, p.{q.page}
                {byId[q.evidence_id] && (
                  <a href={byId[q.evidence_id].pdf_url} target="_blank" rel="noreferrer"> · open at page</a>
                )}
              </footer>
            </blockquote>
          ))}
        </div>
      )}

      <button className="toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? "▾" : "▸"} Evidence shown to the critic ({paper.evidence.length})
      </button>
      {open && (
        <ul className="evidence">
          {paper.evidence.map((e) => (
            <EvidenceItem
              key={e.evidence_id}
              e={e}
              domId={`${paper.paper_id}-${e.evidence_id}`}
              cited={paper.cited_evidence_ids.includes(e.evidence_id)}
              highlight={highlight === e.evidence_id}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function EvidenceItem({ e, domId, cited, highlight }: { e: AgentEvidence; domId: string; cited: boolean; highlight: boolean }) {
  return (
    <li id={domId} className={`ev ${highlight ? "hl" : ""}`}>
      <div className="ev-head">
        <span className="cite static">{e.evidence_id}</span>
        <span className={`badge ${e.chunk_type.endsWith("caption") ? "info" : "subtle"}`}>{TYPE_LABEL[e.chunk_type] ?? e.chunk_type}</span>
        <span className="muted small">{e.label || e.section} · p.{e.page} · rerank {e.rerank_score.toFixed(2)}</span>
        {cited && <span className="badge ok">cited</span>}
        <a href={e.pdf_url} target="_blank" rel="noreferrer" className="small">open at page</a>
      </div>
      <p>{e.text}</p>
    </li>
  );
}

/** Turns "[E1]" / "[E1, E2]" in the critic's reason into clickable evidence chips. */
function withCitations(text: string, byId: Record<string, AgentEvidence>, onClick: (id: string) => void): ReactNode[] {
  const parts = text.split(/(\[(?:E\d+(?:\s*,\s*)?)+\])/g);
  return parts.map((part, i) => {
    const ids = /^\[(?:E\d+(?:\s*,\s*)?)+\]$/.test(part) ? part.match(/E\d+/g) ?? [] : [];
    if (!ids.length) return <Fragment key={i}>{part}</Fragment>;
    return (
      <Fragment key={i}>
        {ids.map((id) =>
          byId[id] ? (
            <button key={id} className="cite" onClick={() => onClick(id)} title={`${byId[id].label || byId[id].section}, p.${byId[id].page}`}>
              {id}
            </button>
          ) : (
            <span key={id} className="cite static">{id}</span>
          ),
        )}
      </Fragment>
    );
  });
}
