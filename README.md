# Financial Filings Q&A: RAG & LLM Evaluation

[![CI](https://github.com/dvskaff/filings-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/dvskaff/filings-rag-eval/actions/workflows/ci.yml)
[![Evaluation](https://github.com/dvskaff/filings-rag-eval/actions/workflows/pipeline.yml/badge.svg)](https://github.com/dvskaff/filings-rag-eval/actions/workflows/pipeline.yml)
![Python](https://img.shields.io/badge/python-3.13-blue) ![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp-black)
![ONNX](https://img.shields.io/badge/embeddings-ONNX%20Runtime-lightgrey) ![FastAPI](https://img.shields.io/badge/FastAPI-0.162-teal)

**Live dashboard: https://dvskaff.github.io/filings-rag-eval/** · **API reference: https://dvskaff.github.io/filings-rag-eval/api.html**

Can a system answer analysts' questions about 10-Ks, 10-Qs and earnings reports correctly, show where each answer came
from, and say "I don't know" instead of guessing? This project builds a retrieval-augmented generation (RAG) system
over real SEC filings and evaluates it on **FinanceBench**, a public benchmark of analyst questions with gold answers
and evidence pages. Everything uses open models on a CPU: no API keys, no GPU.

* **Retrieval:** BM25 keyword search plus dense embeddings (`bge-small-en-v1.5`, ONNX), fused with reciprocal-rank
  fusion and reranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`). Every passage keeps its filing and page number.
* **Generation:** `Qwen2.5-1.5B-Instruct` (4-bit GGUF) via llama.cpp at temperature 0. The prompt tells it to answer
  only from the excerpts, cite them as [1], [2], and say "I don't know" when the excerpts don't contain the answer.
* **Evaluation:**
  * retrieval recall@k and MRR, measured against the annotated evidence pages, for each method, searching one
    filing or all filings;
  * answer accuracy with RAG vs. without retrieval;
  * citations of the evidence page;
  * refusal on control questions given the *wrong company's* filing.
  * Answers are graded by an automatic grader that is itself **validated against FinanceBench's human labels** on
    published GPT-4 Turbo answers.
* **Production pieces:** a quality gate that stops publishing if retrieval regresses, an answer cache that makes
  unchanged re-runs free, and a FastAPI service packaged in a Docker image on GitHub Container Registry (index and all
  models inside), smoke-tested with a real question before every push.

## Results

Rewritten automatically after every evaluation run.

<!-- RESULTS:START -->
_FinanceBench open-source sample: 150 questions · 84 filings · 44,074 passages · LLM: Qwen2.5-1.5B-Instruct (Q4_K_M, llama.cpp) · last run 2026-11-07 03:24 UTC_

| Metric | Result |
|---|---|
| Retrieval recall@5, single filing (hybrid + rerank) | 49% (BM25 alone 34%; MRR 0.37) |
| Correct answers with RAG | 27% (refused 36%, wrong 37%) |
| Correct answers without retrieval (closed book) | 14% |
| Refuses when given the wrong company's filing | 84% of 38 |
| Answers citing the gold evidence page | 12% |
| Grader agreement with FinanceBench human labels | 85% of 600 answers |
| Reference: published GPT-4 Turbo + retrieval (same grader / human label) | 38% / 50% |

**API image:** `docker run -p 8000:8000 ghcr.io/dvskaff/filings-rag-api:latest`
<!-- RESULTS:END -->

## How it works

```
FinanceBench (pinned commit): 150 questions · gold answers · evidence pages · 84 filings (PDF) · published GPT-4 answers + human labels
   │ src/filings/corpus.py      PDF → page text (pdfium) → ~1,200-char passages on line breaks (tables stay whole)
   ▼
src/filings/retrieval.py        BM25 (sparse) + bge-small embeddings (ONNX) → reciprocal-rank fusion → cross-encoder rerank
   ▼
src/filings/evaluate.py         recall@1/3/5/10, MRR (single filing / all filings) ── quality gate (recall@5 floor)
   ▼
src/filings/llm.py              Qwen2.5-1.5B-Instruct (llama.cpp): RAG · closed book · wrong-filing controls (cached)
   ▼
src/filings/grading.py          Correct / Incorrect / Refusal: numbers within 1.5% in any unit, yes/no polarity,
                                key-term recall, refusal phrasing; validated against human labels
   ▼
dashboard + API reference (GitHub Pages) · README results · Docker image (FastAPI + index + models) on GHCR
```

### API

```bash
docker run -p 8000:8000 ghcr.io/dvskaff/filings-rag-api:latest
curl -s localhost:8000/documents
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?", "doc_name": "3M_2018_10K"}'
```

`POST /ask` returns the answer, whether it refused, the cited passages (filing and page) and all retrieved passages.
`POST /search` is retrieval only.

### Limitations

A 1.5-billion-parameter model on a CPU is much weaker than frontier models at multi-step financial arithmetic, so
expect many wrong or refused answers. The point is a measured, honest system, not a leaderboard score. With 150
questions, accuracy has a confidence interval of about ±8 points. Free-text answers are graded approximately, and the
grader was tuned on the published human labels, so its agreement figure is optimistic for new data.

## Repository layout

```
src/filings/   config · corpus · retrieval · llm · grading · evaluate · pipeline · report · readme
               fixture.py: SYNTHETIC filings (tiny PDFs) and questions for tests (pages built from it say so)
serving/       app/main.py (FastAPI) · requirements.txt (API image only)
scripts/       fetch_models.py (downloads the three models into the image)
Dockerfile     the API image (llama.cpp compiled for any x86-64 CPU)
tests/         grader cases, PDF parsing, BM25 and search, evaluation and cache, quality gate, API
```

## Run it

Everything runs in GitHub Actions. Locally:

```bash
pip install -r requirements-dev.txt
pytest -q                                              # offline: synthetic filings, stand-in models
PYTHONPATH=src python -m filings.pipeline               # full evaluation (downloads ~1.5 GB of models and filings)
RAG_QUESTION_LIMIT=10 PYTHONPATH=src python -m filings.pipeline   # quick run on 10 questions
```

Benchmark: [FinanceBench](https://github.com/patronus-ai/financebench) by Patronus AI ([paper](https://arxiv.org/abs/2311.11944)).
Models: BAAI bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2, Qwen2.5-1.5B-Instruct. Code: MIT license.
