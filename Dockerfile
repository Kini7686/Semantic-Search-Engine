# Semantic Search Engine MVP
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build pipeline: passages -> embeddings -> indexes (embedder can take ~10 min on CPU)
RUN python src/data_loader.py
RUN python src/embedder.py
RUN python src/indexer.py

EXPOSE 5000

CMD ["python", "app.py"]
