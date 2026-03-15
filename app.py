import os
import time
from flask import Flask, request, jsonify, send_from_directory

from src.retriever import load_indexes, hybrid_search, get_passages_count

app = Flask(__name__, static_folder="static", static_url_path="")


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": getattr(e, "description", None) or str(e) or "Internal server error"}), 500


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health", methods=["GET"])
def health():
    try:
        n = get_passages_count()
    except Exception:
        n = 0
    return jsonify({"status": "ok", "passages_indexed": n})


@app.route("/search", methods=["POST"])
def search():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    query = data.get("query")
    if query is None:
        return jsonify({"error": "Missing field: query"}), 400
    if not isinstance(query, str):
        return jsonify({"error": "query must be a string"}), 400
    query = query.strip()
    if not query:
        return jsonify({"error": "query cannot be empty"}), 400

    top_k = data.get("top_k", 10)
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        return jsonify({"error": "top_k must be an integer between 1 and 100"}), 400

    use_reranker = data.get("rerank", True)
    if not isinstance(use_reranker, bool):
        use_reranker = True

    t0 = time.perf_counter()
    try:
        results = hybrid_search(query, k=top_k, use_reranker=use_reranker)
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return jsonify({
        "query": query,
        "results": results,
        "latency_ms": latency_ms,
    })


def main():
    load_indexes()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
