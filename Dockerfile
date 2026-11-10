FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY ingest.py chatbot.py streamlit_app.py ./
COPY data/sources.json ./data/sources.json

RUN useradd --create-home appuser && mkdir -p /app/logs /app/data/documents && chown -R appuser /app
USER appuser

EXPOSE 8000 8501

# Default: API (auto-ingests on first start if the collection is empty)
CMD ["sh", "-c", "python scripts/ensure_index.py && uvicorn app.api:app --host 0.0.0.0 --port 8000"]
