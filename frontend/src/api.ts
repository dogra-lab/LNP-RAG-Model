// Types mirror lnp_rag/models.py (AgentSearchRequest / AgentSearchResponse).

export type Verdict = "ACCEPT" | "REJECT";
export type ChunkType = "title" | "abstract" | "paragraph" | "figure_caption" | "table_caption";

export interface AgentEvidence {
  evidence_id: string;
  chunk_id: string;
  chunk_type: ChunkType;
  section: string;
  label: string;
  page: number;
  text: string;
  rerank_score: number;
  pdf_url: string;
}

export interface CitedQuote {
  evidence_id: string;
  quote: string;
  page: number;
  label: string;
}

export interface AgentPaper {
  paper_id: string;
  title: string;
  filename: string;
  folder_labels: string[];
  year: number | null;
  doi: string | null;
  retriever_score: number;
  route: "candidate" | "near_miss";
  caption_match: boolean;
  verdict: Verdict;
  reason: string;
  critic: string;
  cited_evidence_ids: string[];
  quotes: CitedQuote[];
  evidence: AgentEvidence[];
  found_by: string[];
  pdf_url: string;
}

export interface FolderReport {
  folder: string;
  papers_in_folder: number;
  chunks_scanned: number;
  candidates: number;
  near_misses: number;
  dropped: number;
  accepted: AgentPaper[];
  rejected: AgentPaper[];
  timings_ms: Record<string, number>;
  warnings: string[];
}

export interface AgentSearchResponse {
  query: string;
  expanded_queries: string[];
  accepted: AgentPaper[];
  folders: FolderReport[];
  timings_ms: Record<string, number>;
  warnings: string[];
}

export interface SearchFilters {
  year_min?: number | null;
  year_max?: number | null;
  exclude_reviews?: boolean;
}

export interface AgentSearchRequest {
  query: string;
  folders: string[];
  filters: SearchFilters;
  use_critic: boolean;
}

export interface Health {
  status: string;
  indexed_papers: number;
  indexing: boolean;
  llm_provider: string | null;
  llm_model: string | null;
}

export interface Stats {
  papers: number;
  chunks: number;
  folders: Record<string, number>;
}

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new Error("Cannot reach the API. Is it running? (uvicorn lnp_rag.api.main:app --port 8000)");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    if (res.status === 502 || res.status === 504) {
      detail = "The API is not responding. Start it with: uvicorn lnp_rag.api.main:app --port 8000";
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const getHealth = () => request<Health>("/health");
export const getStats = () => request<Stats>("/stats");

export const agentSearch = (req: AgentSearchRequest, signal?: AbortSignal) =>
  request<AgentSearchResponse>("/agents/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
