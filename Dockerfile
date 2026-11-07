# Filings Q&A API: FastAPI + hybrid retrieval (fastembed / ONNX Runtime) + Qwen2.5-1.5B via llama.cpp, CPU only.
FROM python:3.13-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
COPY serving/requirements.txt .
# Portable build: no -march=native, so the image runs on any x86-64 machine.
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF" pip wheel --no-cache-dir -w /wheels -r requirements.txt

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    RAG_INDEX_DIR=/app/index RAG_MODEL_DIR=/app/models
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY src/filings/__init__.py src/filings/config.py src/filings/retrieval.py src/filings/llm.py filings/
COPY serving/app app/
COPY serving/index index/
COPY scripts/fetch_models.py /tmp/fetch_models.py
RUN python /tmp/fetch_models.py /app/models /app/index/meta.json
ENV RAG_LLM_PATH=/app/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf

RUN useradd --create-home --uid 10001 api && chown -R api /app
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

LABEL org.opencontainers.image.source="https://github.com/dvskaff/filings-rag-eval" \
      org.opencontainers.image.description="Q&A over SEC filings with citations: hybrid retrieval + local LLM (CPU)" \
      org.opencontainers.image.licenses="MIT"
