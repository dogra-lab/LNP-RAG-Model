"""Build and load the hybrid (dense + BM25) paper index."""

from __future__ import annotations

import html
import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .extract import extract_paper, file_sha1

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")
_STOP = set(
    "the and for with that this from were was are been have has which into their than also these "
    "using used use our not but can may its after both between all each such other more most".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def load_embedder():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(EMBED_MODEL, device=device)


def build_index(papers_dir: Path, index_dir: Path, workers: int = 8) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(papers_dir.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs under {papers_dir}")

    # Deduplicate identical files that appear in several folders, keeping every folder label.
    by_hash: dict[str, list[Path]] = {}
    with ProcessPoolExecutor(workers) as pool:
        for path, sha in zip(pdfs, tqdm(pool.map(file_sha1, pdfs, chunksize=8), total=len(pdfs), desc="hashing")):
            by_hash.setdefault(sha, []).append(path)
    print(f"{len(by_hash)} unique papers ({len(pdfs) - len(by_hash)} duplicate files)")

    shas = list(by_hash)
    firsts = [str(by_hash[s][0]) for s in shas]
    papers, chunk_texts, chunk_owner, failed = [], [], [], []
    with ProcessPoolExecutor(workers) as pool:
        records = pool.map(extract_paper, firsts, chunksize=4)
        for sha, rec in zip(shas, tqdm(records, total=len(shas), desc="extracting")):
            if rec is None:
                failed.append(str(by_hash[sha][0]))
                continue
            chunks = rec.pop("chunks")
            rec["id"] = sha[:12]
            rec["folders"] = sorted({p.parent.name for p in by_hash[sha]})
            rec["all_paths"] = [str(p) for p in by_hash[sha]]
            pid = len(papers)
            papers.append(rec)
            chunk_texts.extend(chunks)
            chunk_owner.extend([pid] * len(chunks))

    if failed:
        print(f"Skipped {len(failed)} unreadable/image-only PDFs (listed in {index_dir / 'failed.txt'})")
        (index_dir / "failed.txt").write_text("\n".join(failed))
    _write_index(index_dir, papers, chunk_texts, chunk_owner)


def _validated_groups(papers_dir: Path) -> dict[str, list[Path]]:
    """Group 'P007A Sarkar et al 2026.pdf' with its supplement 'P007B.pdf' (and P06B with P006)."""
    groups: dict[str, list[Path]] = {}
    for pdf in sorted(papers_dir.glob("*.pdf")):
        m = re.match(r"P0*(\d+)", pdf.stem)
        groups.setdefault(f"P{int(m.group(1)):03d}" if m else pdf.stem, []).append(pdf)
    return groups


def _is_supplement(pdf: Path) -> bool:
    stem = pdf.stem.strip()
    return bool(re.fullmatch(r"P\d+B", stem) or re.search(r"\d{4}B$", stem))  # 'P013B', 'Munsin et al 2021B'


def build_validated_index(papers_dir: Path, index_dir: Path) -> None:
    """Index the curated 'paper used' folder; supplements are merged into their main paper."""
    index_dir.mkdir(parents=True, exist_ok=True)
    papers, chunk_texts, chunk_owner = [], [], []
    for key, files in tqdm(_validated_groups(papers_dir).items(), desc="validated papers"):
        mains = [f for f in files if not _is_supplement(f)] or files[:1]
        supps = [f for f in files if f not in mains]
        rec = extract_paper(str(mains[0]))
        if rec is None:
            print(f"Could not read {mains[0]}")
            continue
        chunks = rec.pop("chunks")
        supp_texts = []
        for supp in supps:
            srec = extract_paper(str(supp))
            if srec:
                rec["captions"] += [f"Supplementary: {c}" for c in srec["captions"]]
                supp_texts.append(srec["results"] or srec["abstract"])
                chunks += srec["chunks"][1:]  # skip the supplement's own title/abstract chunk
        rec["supplement"] = " ".join(supp_texts)[:6000]
        rec["id"] = "v-" + file_sha1(mains[0])[:10]
        rec["label"] = mains[0].stem.strip()
        rec["validated"] = True
        rec["folders"] = [papers_dir.name]
        rec["all_paths"] = [str(f) for f in mains + supps]
        pid = len(papers)
        papers.append(rec)
        chunk_texts.extend(chunks)
        chunk_owner.extend([pid] * len(chunks))
    print(f"{len(papers)} validated papers ({sum(len(f) for f in _validated_groups(papers_dir).values())} files)")
    _write_index(index_dir, papers, chunk_texts, chunk_owner)


def _write_index(index_dir: Path, papers: list[dict], chunk_texts: list[str], chunk_owner: list[int]) -> None:
    print(f"Indexed {len(papers)} papers -> {len(chunk_texts)} chunks")
    model = load_embedder()
    emb = model.encode(
        chunk_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    )
    np.save(index_dir / "embeddings.npy", emb.astype(np.float16))
    np.save(index_dir / "chunk_owner.npy", np.asarray(chunk_owner, dtype=np.int32))

    with open(index_dir / "chunks.jsonl", "w") as f:
        for text in chunk_texts:
            f.write(json.dumps(text) + "\n")
    with open(index_dir / "papers.jsonl", "w") as f:
        for rec in papers:
            f.write(json.dumps(rec) + "\n")

    print("Building BM25 index...")
    bm25 = BM25Okapi([tokenize(t) for t in tqdm(chunk_texts, desc="tokenizing")])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"Done. Index written to {index_dir}")


class PaperIndex:
    def __init__(self, index_dir: Path):
        if not (index_dir / "papers.jsonl").exists():
            raise SystemExit(f"No index at {index_dir}. Run `python ingest.py` first.")
        with open(index_dir / "papers.jsonl") as f:
            self.papers = [json.loads(line) for line in f]
        for p in self.papers:
            p["title"] = html.unescape(p["title"])
        with open(index_dir / "chunks.jsonl") as f:
            self.chunks = [json.loads(line) for line in f]
        self.emb = np.load(index_dir / "embeddings.npy").astype(np.float32)
        self.owner = np.load(index_dir / "chunk_owner.npy")
        with open(index_dir / "bm25.pkl", "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        self._model = None

    def _paper_scores(self, chunk_scores: np.ndarray) -> np.ndarray:
        """Paper score = best chunk + 0.5 * mean of its top-3 chunks (rewards repeated evidence)."""
        n = len(self.papers)
        best = np.full(n, -np.inf, dtype=np.float32)
        np.maximum.at(best, self.owner, chunk_scores)
        order = np.lexsort((-chunk_scores, self.owner))
        top3_sum = np.zeros(n, dtype=np.float32)
        top3_cnt = np.zeros(n, dtype=np.float32)
        prev, rank = -1, 0
        for i in order:
            p = self.owner[i]
            rank = rank + 1 if p == prev else 0
            prev = p
            if rank < 3:
                top3_sum[p] += chunk_scores[i]
                top3_cnt[p] += 1
        return best + 0.5 * top3_sum / np.maximum(top3_cnt, 1)

    def paper_vectors(self) -> np.ndarray:
        """One normalized vector per paper: the mean of its chunk embeddings."""
        if getattr(self, "_paper_vecs", None) is None:
            vecs = np.zeros((len(self.papers), self.emb.shape[1]), dtype=np.float32)
            np.add.at(vecs, self.owner, self.emb)
            self._paper_vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-9)
        return self._paper_vecs

    def head_vectors(self) -> np.ndarray:
        """Embedding of each paper's first chunk (title + abstract)."""
        return self.emb[np.unique(self.owner, return_index=True)[1]]

    def encode(self, queries: list[str]) -> np.ndarray:
        if self._model is None:
            self._model = load_embedder()
        return self._model.encode([QUERY_PREFIX + q for q in queries], normalize_embeddings=True).astype(np.float32)

    def best_chunks(self, pid: int, vecs: np.ndarray, k: int = 4) -> list[int]:
        """The paper's k chunks closest to any of the query vectors."""
        ids = np.flatnonzero(self.owner == pid)
        return ids[np.argsort(-(self.emb[ids] @ vecs.T).max(1))[:k]].tolist()

    def search(
        self, query: str, k: int, extra_scores: list[np.ndarray] = (), exclude: set[int] = frozenset()
    ) -> list[tuple[int, float, list[int]]]:
        """Hybrid retrieval with reciprocal-rank fusion.

        extra_scores: additional per-paper score arrays fused as equal-weight rankings.
        Returns up to k (paper_idx, fused_score, best_chunk_ids) tuples, best first.
        """
        q = self.encode([query])[0]
        dense_chunks = self.emb @ q
        bm25_chunks = self.bm25.get_scores(tokenize(query)).astype(np.float32)

        fused = np.zeros(len(self.papers))
        for scores in (self._paper_scores(dense_chunks), self._paper_scores(bm25_chunks), *extra_scores):
            ranks = np.empty(len(scores), dtype=np.int64)
            ranks[np.argsort(-scores)] = np.arange(len(scores))
            fused += 1.0 / (60 + ranks)
        if exclude:
            fused[list(exclude)] = -np.inf

        top = [p for p in np.argsort(-fused)[:k] if np.isfinite(fused[p])]
        # Per-paper evidence: chunks ranked by combined (normalized) dense + BM25 score.
        combined = dense_chunks + bm25_chunks / (bm25_chunks.max() or 1.0)
        out = []
        for p in top:
            ids = np.flatnonzero(self.owner == p)
            best = ids[np.argsort(-combined[ids])[:4]].tolist()
            out.append((int(p), float(fused[p]), best))
        return out
