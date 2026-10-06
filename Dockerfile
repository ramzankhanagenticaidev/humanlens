FROM python:3.12-slim AS base

# Prevent Python from buffering stdout/stderr (so `docker logs` is live)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: none required beyond what's in the slim image for this
# stack (no compiler needed — scikit-learn/numpy/scipy ship wheels).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the NLTK tokenizer data at build time so the container
# doesn't need network access on first run.
RUN python -c "import nltk; nltk.download('punkt_tab')"

COPY . .

# Persist the trained model + SQLite history under /app/data, which
# docker-compose.yml mounts as a named volume. These match the
# defaults docker-compose.yml sets at runtime, so the model trained
# during this build is what a fresh container actually loads (Docker
# seeds an empty named volume with the image's existing directory
# content on first start).
ENV HUMANLENS_DB_PATH=/app/data/humanlens.db \
    HUMANLENS_MODEL_DIR=/app/data/models
RUN mkdir -p /app/data && \
    python scripts/build_seed_dataset.py && \
    python scripts/train_model.py

# Create a non-root user
RUN useradd --create-home --uid 1000 humanlens && chown -R humanlens:humanlens /app
USER humanlens

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
