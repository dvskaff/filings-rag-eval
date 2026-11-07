"""Project settings. Everything can be overridden with environment variables (tests and CI)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ[name]).resolve() if os.environ.get(name) else default


@dataclass
class Config:
    # FinanceBench open-source sample (Patronus AI), pinned to a commit so the benchmark never moves under us.
    benchmark_repo: str = "patronus-ai/financebench"
    benchmark_commit: str = "cc39aeb4afdf33909ee1412188bf89035950c2eb"

    # Models (all open, no API keys): ONNX embedding + reranker via fastembed, a small instruct LLM via llama.cpp.
    embedder: str = field(default_factory=lambda: os.environ.get("RAG_EMBEDDER", "BAAI/bge-small-en-v1.5"))
    reranker: str = field(
        default_factory=lambda: os.environ.get("RAG_RERANKER", "Xenova/ms-marco-MiniLM-L-6-v2")
    )
    llm_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    llm_file: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    llm_name: str = "Qwen2.5-1.5B-Instruct (Q4_K_M, llama.cpp)"
    backend: str = field(
        default_factory=lambda: os.environ.get("RAG_BACKEND", "real")
    )  # "fake" for offline tests
    fixture: bool = field(default_factory=lambda: os.environ.get("RAG_FIXTURE") == "1")

    chunk_chars: int = 1200
    chunk_overlap: int = 200
    candidates: int = 30  # hybrid candidates passed to the reranker
    context_k: int = 5  # passages given to the LLM
    max_answer_tokens: int = 256
    unanswerable_every: int = 4  # every 4th question is also asked against the wrong company's filing
    question_limit: int = field(
        default_factory=lambda: int(os.environ.get("RAG_QUESTION_LIMIT", "0"))
    )  # 0 = all
    min_recall_at_5: float = 0.25  # quality gate: absolute floor for single-filing recall@5 ...
    max_recall_drop: float = 0.05  # ... and no drop of more than 5 points from the last published run

    data_dir: Path = field(default_factory=lambda: _env_path("RAG_DATA_DIR", ROOT / "data"))
    site_dir: Path = field(default_factory=lambda: _env_path("RAG_SITE_DIR", ROOT / "site"))
    bundle_dir: Path = field(default_factory=lambda: _env_path("RAG_BUNDLE_DIR", ROOT / "serving" / "index"))
    readme_path: Path = field(default_factory=lambda: _env_path("RAG_README", ROOT / "README.md"))

    repo_url: str = "https://github.com/dvskaff/filings-rag-eval"
    site_url: str = "https://dvskaff.github.io/filings-rag-eval/"
    image: str = "ghcr.io/dvskaff/filings-rag-api"

    def raw_url(self, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.benchmark_repo}/{self.benchmark_commit}/{path}"
