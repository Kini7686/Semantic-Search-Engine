"""
Load Wikipedia Simple English, clean text, chunk into overlapping passages.
Output: data/passages.json
"""
import re
import json
from pathlib import Path

from datasets import load_dataset


# Chunk config
PASSAGE_TOKENS = 256
OVERLAP_TOKENS = 50
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "passages.json"


def clean_text(text: str) -> str:
    """Remove HTML tags, special characters, normalize whitespace."""
    if not text or not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace common special chars with space
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^\w\s.,;:!?'\"-]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str) -> list[str]:
    """Simple word tokenization (split on whitespace). Used for chunking."""
    return text.split()


def chunk_document(text: str, title: str, doc_id: str, passage_start_id: int) -> list[dict]:
    """
    Split document into overlapping passages of 256 tokens with 50-token overlap.
    Returns list of {"id", "text", "title"} dicts.
    """
    tokens = tokenize_words(text)
    if not tokens:
        return []

    passages = []
    step = PASSAGE_TOKENS - OVERLAP_TOKENS
    local_idx = 0

    for start in range(0, len(tokens), step):
        end = min(start + PASSAGE_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        passage_text = " ".join(chunk_tokens).strip()
        if not passage_text:
            continue
        passages.append({
            "id": passage_start_id + local_idx,
            "text": passage_text,
            "title": title,
        })
        local_idx += 1
        if end >= len(tokens):
            break

    return passages


def load_wikipedia(split: str = "train[:5000]") -> list[dict]:
    """Load Wikipedia Simple English from HuggingFace. Returns list of {title, text}."""
    for name, config in [
        ("wikimedia/wikipedia", "20220301.simple"),
        ("wikipedia", "20220301.simple"),
        ("wikimedia/wikipedia", "20231101.simple"),
    ]:
        try:
            dataset = load_dataset(name, config, split=split)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not load Wikipedia dataset. Tried wikimedia/wikipedia and wikipedia.")

    rows = []
    for row in dataset:
        title = row.get("title") or ""
        text = row.get("text") or ""
        rows.append({"title": title, "text": text})
    return rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Wikipedia Simple English (5000 articles)...")
    docs = load_wikipedia()
    print(f"Loaded {len(docs)} articles.")

    all_passages = []
    passage_id = 0

    for doc in docs:
        title = clean_text(doc["title"])
        text = clean_text(doc["text"])
        if not text:
            continue
        for p in chunk_document(text, title, str(passage_id), passage_id):
            all_passages.append(p)
            passage_id = p["id"] + 1

    print(f"Total passages: {len(all_passages)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_passages, f, ensure_ascii=False, indent=0)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
