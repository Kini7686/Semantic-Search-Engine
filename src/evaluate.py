import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.retriever import load_indexes, semantic_search_results, bm25_search_results, hybrid_search

OUTPUT_PATH = BASE / "evaluation_results.json"

TEST_QUERIES = [
    ("what causes climate change", "climate"),
    ("symptoms of heart disease", "heart"),
    ("who wrote Romeo and Juliet", "Shakespeare"),
    ("how does photosynthesis work", "photosynthesis"),
    ("capital of France", "Paris"),
    ("when did World War 2 end", "1945"),
    ("what is the speed of light", "light"),
    ("who invented the telephone", "telephone"),
    ("how do vaccines work", "vaccine"),
    ("what is democracy", "democracy"),
    ("largest planet in solar system", "Jupiter"),
    ("what is global warming", "warming"),
    ("symptoms of diabetes", "diabetes"),
    ("who painted the Mona Lisa", "Mona Lisa"),
    ("how do earthquakes happen", "earthquake"),
    ("what is artificial intelligence", "intelligence"),
    ("who was Albert Einstein", "Einstein"),
    ("what is the Great Wall of China", "China"),
    ("how do birds fly", "birds"),
    ("what causes rain", "rain"),
    ("who discovered America", "America"),
    ("what is the internet", "internet"),
    ("how does the heart work", "heart"),
    ("what is evolution", "evolution"),
    ("who wrote Harry Potter", "Rowling"),
    ("what is gravity", "gravity"),
    ("how do computers work", "computer"),
    ("what is the United Nations", "United Nations"),
    ("who was Napoleon", "Napoleon"),
    ("what is recycling", "recycling"),
    ("how do plants grow", "plants"),
    ("what is the solar system", "solar"),
    ("who invented the light bulb", "Edison"),
    ("what is pollution", "pollution"),
    ("how does the brain work", "brain"),
    ("what is the Renaissance", "Renaissance"),
    ("who was Cleopatra", "Cleopatra"),
    ("what is renewable energy", "energy"),
    ("how do airplanes fly", "airplane"),
    ("what is the water cycle", "water"),
    ("who was Leonardo da Vinci", "Leonardo"),
    ("what is biodiversity", "biodiversity"),
    ("how does digestion work", "digestion"),
    ("what is the Roman Empire", "Roman"),
    ("who was Marie Curie", "Curie"),
    ("what is recycling", "recycle"),
    ("how do magnets work", "magnet"),
    ("what is the Industrial Revolution", "Industrial"),
    ("who was Isaac Newton", "Newton"),
    ("what is ozone layer", "ozone"),
]


def passage_contains_keyword(passage_text: str, keyword: str) -> bool:
    return keyword.lower() in passage_text.lower()


def recall_at_k(results: list[dict], keyword: str, k: int = 5) -> bool:
    for r in results[:k]:
        if passage_contains_keyword(r["passage"], keyword):
            return True
    return False


def mrr_at_k(results: list[dict], keyword: str, k: int = 10) -> float:
    for r in results[:k]:
        if passage_contains_keyword(r["passage"], keyword):
            return 1.0 / r["rank"]
    return 0.0


def run_evaluation():
    load_indexes()
    n_queries = len(TEST_QUERIES)

    results_bm25 = []
    results_semantic = []
    results_hybrid = []

    for query, keyword in TEST_QUERIES:
        results_bm25.append(bm25_search_results(query, k=10))
        results_semantic.append(semantic_search_results(query, k=10))
        results_hybrid.append(hybrid_search(query, k=10))

    def compute_metrics(result_lists):
        recall5 = sum(recall_at_k(rs, kw, 5) for (_, kw), rs in zip(TEST_QUERIES, result_lists)) / n_queries
        mrr10 = sum(mrr_at_k(rs, kw, 10) for (_, kw), rs in zip(TEST_QUERIES, result_lists)) / n_queries
        return {"recall_at_5": round(recall5, 4), "mrr_at_10": round(mrr10, 4)}

    metrics_bm25 = compute_metrics(results_bm25)
    metrics_semantic = compute_metrics(results_semantic)
    metrics_hybrid = compute_metrics(results_hybrid)

    print("\n--- Evaluation: BM25 vs Semantic vs Hybrid ---\n")
    print(f"{'Method':<12}  {'Recall@5':>10}  {'MRR@10':>10}")
    print("-" * 36)
    print(f"{'BM25':<12}  {metrics_bm25['recall_at_5']:>10.4f}  {metrics_bm25['mrr_at_10']:>10.4f}")
    print(f"{'Semantic':<12}  {metrics_semantic['recall_at_5']:>10.4f}  {metrics_semantic['mrr_at_10']:>10.4f}")
    print(f"{'Hybrid':<12}  {metrics_hybrid['recall_at_5']:>10.4f}  {metrics_hybrid['mrr_at_10']:>10.4f}")
    print()

    improvement_mrr = (metrics_hybrid["mrr_at_10"] - metrics_bm25["mrr_at_10"]) / (metrics_bm25["mrr_at_10"] or 1) * 100
    print(f"Hybrid vs BM25 MRR improvement: {improvement_mrr:.1f}%")
    print()

    out = {
        "num_queries": n_queries,
        "bm25": metrics_bm25,
        "semantic": metrics_semantic,
        "hybrid": metrics_hybrid,
        "hybrid_vs_bm25_mrr_improvement_pct": round(improvement_mrr, 2),
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()
