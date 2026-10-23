FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only PyTorch first: the default wheel pulls ~2 GB of CUDA libraries you can't use
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model at build time so the first request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY api.py rag.py ./
COPY static ./static

RUN useradd --create-home appuser && chown -R appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]