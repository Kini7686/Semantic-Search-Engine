FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python src/data_loader.py
RUN python src/embedder.py
RUN python src/indexer.py

EXPOSE 7860
ENV PORT=7860

CMD ["python", "app.py"]
