# Semantic Search Engine MVP

A search engine that retrieves passages by **meaning**, not just keywords. Queries like "symptoms of heart disease" also return results about "chest pain" and "cardiovascular risk factors" via dense vector embeddings and hybrid ranking.

## Tech Stack

- **Python 3.10**
- **sentence-transformers** (all-MiniLM-L6-v2) — dense embeddings; **CrossEncoder** (ms-marco-MiniLM-L-6-v2) — reranker
- **FAISS** — vector index for fast similarity search
- **rank-bm25** — sparse keyword index
- **Flask** — REST API

## Quick Start

### 1. Create environment and install

```bash
cd semantic-search
python -m venv venv
# Windows: venv\Scripts\activate
# Unix: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the pipeline (run in order)

```bash
# Download Wikipedia, clean, chunk → data/passages.json
python src/data_loader.py

# Embed all passages → indexes/embeddings.npy
python src/embedder.py

# Build FAISS + BM25 indexes → indexes/faiss.index, indexes/bm25.pkl
python src/indexer.py
```

### 3. Run the API and website

```bash
python app.py
# Server at http://127.0.0.1:5000
```

- **Website:** open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser to search with the web UI.
- **API:** `POST /search` body: `{"query": "...", "top_k": 10, "rerank": true}` (rerank uses a cross-encoder; set `"rerank": false` to skip). `GET /health`.

### 5. Evaluate

```bash
python src/evaluate.py
# Prints BM25 vs Semantic vs Hybrid comparison; writes evaluation_results.json
```

## Docker

```bash
docker build -t semantic-search .
docker run -p 5000:5000 semantic-search
```

The Dockerfile runs the full pipeline (data_loader → embedder → indexer) during build, so the first build can take ~10–15 minutes.

## Host on Hugging Face Spaces (free)

1. **Push your code to GitHub** (e.g. `https://github.com/YourUsername/Semantic-Search-Engine`).

2. **Create a Space** at [huggingface.co/new-space](https://huggingface.co/new-space):
   - **Name:** e.g. `semantic-search`
   - **License:** pick one (e.g. MIT)
   - **Select:** **Docker** (not Gradio/Streamlit)
   - Create the Space.

3. **Connect the Space to your repo:**
   - In the Space, open the **Settings** (gear) tab.
   - Under **Repository**, use **“Clone from a repo”** or **“Add a Dockerfile”**.
   - If you use **“Clone from a repo”**: paste your GitHub repo URL. The Space will use the **Dockerfile** in the root.
   - Alternatively, in the Space’s **Files** tab you can upload your project (or drag the Dockerfile and paste its content). The Space needs at least a **Dockerfile** in the repo root.

4. **Build and run:**
   - The Space will run `docker build` using your Dockerfile (installs deps, runs data_loader → embedder → indexer, then starts the app).
   - **First build can take 15–30 minutes** (embedding ~18K passages on CPU). You can watch logs in the Space’s **Logs** tab.
   - When it’s ready, your app will be at:  
     `https://huggingface.co/spaces/YourUsername/semantic-search`

5. **Use the app:** Open the Space URL in a browser; use the search box. The app listens on port **7860** on Spaces (handled by the Dockerfile `ENV PORT=7860`).

## API

| Endpoint       | Method | Description                    |
|----------------|--------|--------------------------------|
| `/search`      | POST   | Hybrid search (body: `query`, `top_k`) |
| `/health`      | GET    | Status and passage count       |

## Performance Targets

- Query latency: &lt; 200 ms for top-10
- ~50K passages from 5000 Wikipedia articles
- Hybrid vs BM25: ~15–25% MRR improvement
- Embedding: ~10 min for 50K passages on CPU

## Layout

```
semantic-search/
├── data/
│   └── passages.json
├── indexes/
│   ├── faiss.index
│   ├── bm25.pkl
│   └── embeddings.npy
├── static/
│   └── index.html    # Web UI
├── src/
│   ├── data_loader.py
│   ├── embedder.py
│   ├── indexer.py
│   ├── retriever.py
│   └── evaluate.py
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Dataset

Uses **Wikipedia Simple English** from HuggingFace: `wikipedia` / `20220301.simple`, first 5000 articles. If the dataset name has changed, use `wikimedia/wikipedia` with config `20220301.simple`.
