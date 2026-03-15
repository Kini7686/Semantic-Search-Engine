"""
Load FAISS + BM25 indexes and passages; expose semantic, BM25, and hybrid search.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
INDEX_DIR = BASE / "indexes"
PASSAGES_PATH = DATA_DIR / "passages.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hybrid weights
SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# Reranker: how many candidates to fetch before reranking (top-k after rerank is from hybrid_search k)
RERANK_CANDIDATES = 30
# Max passage length (chars) to send to reranker to avoid token limit errors
RERANK_MAX_PASSAGE_CHARS = 2000

# Singleton state
_passages: list[dict] | None = None
_faiss_index: faiss.IndexFlatIP | None = None
_bm25: BM25Okapi | None = None
_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def load_indexes() -> None:
    """Load FAISS index, BM25 index, and passages from disk. Call once at startup."""
    global _passages, _faiss_index, _bm25, _model

    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        _passages = json.load(f)

    _faiss_index = faiss.read_index(str(FAISS_PATH))
    with open(BM25_PATH, "rb") as f:
        _bm25 = pickle.load(f)

    _model = SentenceTransformer(MODEL_NAME)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string; returns L2-normalized vector (1, dim)."""
    if _model is None:
        load_indexes()
    vec = _model.encode([query], normalize_embeddings=True)
    return np.ascontiguousarray(vec.astype(np.float32))


def semantic_search(query: str, k: int = 20) -> list[tuple[int, float]]:
    """
    FAISS top-k nearest neighbors by cosine similarity.
    Returns list of (passage_index, score) where score is inner product (cosine for normalized).
    """
    if _faiss_index is None or _passages is None:
        load_indexes()
    q = embed_query(query)
    scores, indices = _faiss_index.search(q, min(k, len(_passages)))
    out = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            out.append((int(idx), float(scores[0][i])))
    return out


def bm25_search(query: str, k: int = 20) -> list[tuple[int, float]]:
    """
    BM25 top-k. Returns list of (passage_index, bm25_score).
    """
    if _bm25 is None or _passages is None:
        load_indexes()
    tokenized = tokenize_for_bm25(query)
    scores = _bm25.get_scores(tokenized)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span <= 0:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


def _load_reranker() -> CrossEncoder:
    """Load cross-encoder reranker lazily (first time reranking is used)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def semantic_search_results(query: str, k: int = 10) -> list[dict]:
    """Like hybrid_search format: list of {rank, passage, title, score} for semantic only."""
    if _passages is None:
        load_indexes()
    pairs = semantic_search(query, k=k)
    return [
        {"rank": r, "passage": _passages[idx]["text"], "title": _passages[idx]["title"], "score": round(score, 4)}
        for r, (idx, score) in enumerate(pairs, 1)
    ]


def bm25_search_results(query: str, k: int = 10) -> list[dict]:
    """Like hybrid_search format: list of {rank, passage, title, score} for BM25 only."""
    if _passages is None:
        load_indexes()
    pairs = bm25_search(query, k=k)
    return [
        {"rank": r, "passage": _passages[idx]["text"], "title": _passages[idx]["title"], "score": round(score, 4)}
        for r, (idx, score) in enumerate(pairs, 1)
    ]


def _hybrid_candidates(query: str, candidate_k: int) -> list[tuple[int, float]]:
    """Get top candidate_k passages by hybrid (semantic + BM25) score. Returns [(idx, score), ...]."""
    sem_pairs = semantic_search(query, k=20)
    bm25_pairs = bm25_search(query, k=20)
    sem_scores = _normalize_scores([s for _, s in sem_pairs])
    bm25_scores_raw = [s for _, s in bm25_pairs]
    bm25_scores = _normalize_scores(bm25_scores_raw)
    by_idx: dict[int, tuple[float, float]] = {}
    for i, (idx, _) in enumerate(sem_pairs):
        by_idx[idx] = (sem_scores[i], by_idx.get(idx, (0, 0))[1])
    for i, (idx, _) in enumerate(bm25_pairs):
        prev = by_idx.get(idx, (0, 0))
        by_idx[idx] = (prev[0], bm25_scores[i])
    combined = [
        (idx, SEMANTIC_WEIGHT * s + BM25_WEIGHT * b)
        for idx, (s, b) in by_idx.items()
    ]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:candidate_k]


def hybrid_search(query: str, k: int = 10, use_reranker: bool = True) -> list[dict]:
    """
    Combine semantic (0.6) and BM25 (0.4) to get candidates; optionally rerank with a
    cross-encoder and return top-k with passage text, title, and score.
    """
    if _passages is None:
        load_indexes()

    candidate_k = RERANK_CANDIDATES if use_reranker else k
    top = _hybrid_candidates(query, candidate_k)
    if not top:
        return []

    if not use_reranker:
        results = []
        for rank, (idx, score) in enumerate(top[:k], 1):
            p = _passages[idx]
            results.append({
                "rank": rank,
                "passage": p["text"],
                "title": p["title"],
                "score": round(score, 4),
            })
        return results

    # Rerank: score each (query, passage) with cross-encoder, then take top-k
    def _truncate(t: str) -> str:
        return t[:RERANK_MAX_PASSAGE_CHARS] if len(t) > RERANK_MAX_PASSAGE_CHARS else t
    pairs = [(query, _truncate(_passages[idx]["text"])) for idx, _ in top]
    reranker = _load_reranker()
    raw_scores = reranker.predict(pairs)
    # predict() can return scalar for single pair; ensure list for zip
    reranker_scores = np.atleast_1d(raw_scores).tolist()
    # Sort by reranker score descending; keep idx via enumerate
    indexed = list(zip(top, reranker_scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_reranked = indexed[:k]
    r_scores = [s for _, s in top_reranked]
    r_norm = _normalize_scores(r_scores)
    results = []
    for rank, (entry, norm_score) in enumerate(zip(top_reranked, r_norm), 1):
        idx = entry[0][0]  # entry is ((idx, hybrid_score), reranker_score)
        p = _passages[idx]
        results.append({
            "rank": rank,
            "passage": p["text"],
            "title": p["title"],
            "score": round(norm_score, 4),
        })
    return results


def get_passages_count() -> int:
    if _passages is None:
        load_indexes()
    return len(_passages)
