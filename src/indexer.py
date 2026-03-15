import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
INDEX_DIR = BASE / "indexes"
PASSAGES_PATH = DATA_DIR / "passages.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"


def tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        passages = json.load(f)

    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    n, d = embeddings.shape
    assert n == len(passages), "Mismatch between passages and embeddings count"

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"FAISS index built: {n} vectors, dim {d} -> {FAISS_PATH}")

    tokenized = [tokenize_for_bm25(p["text"]) for p in passages]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index built: {n} passages -> {BM25_PATH}")


if __name__ == "__main__":
    main()
