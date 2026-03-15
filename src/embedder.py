import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from data_loader import OUTPUT_PATH as PASSAGES_PATH

INDEX_DIR = Path(__file__).resolve().parent.parent / "indexes"
OUTPUT_PATH = INDEX_DIR / "embeddings.npy"
BATCH_SIZE = 64
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    import json
    PASSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        passages = json.load(f)

    texts = [p["text"] for p in passages]
    n = len(texts)
    print(f"Total passages to embed: {n}")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0

    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    np.save(OUTPUT_PATH, embeddings)

    print(f"Total passages embedded: {n}")
    print(f"Embedding dimension: {dim}")
    print(f"Time taken: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
