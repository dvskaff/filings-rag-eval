import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))
os.environ.setdefault("RAG_BACKEND", "fake")

from filings import corpus, retrieval  # noqa: E402
from filings.config import Config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return replace(Config(), data_dir=tmp_path / "data", site_dir=tmp_path / "site", bundle_dir=tmp_path / "bundle",
                   readme_path=tmp_path / "README.md", fixture=True, backend="fake")  # fmt: skip


@pytest.fixture
def bench(cfg):
    b = corpus.download(cfg)
    chunks = corpus.build_chunks(cfg, b["dir"], b["documents"])
    emb, rr = retrieval.load_models(cfg)
    return b, chunks, retrieval.Retriever(chunks, emb.passages(chunks["text"].tolist()), emb, rr)
