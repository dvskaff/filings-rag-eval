"""Filings Q&A API: hybrid retrieval + reranking + a local LLM, with page citations.

uvicorn app.main:app --port 8000      (from serving/, with src/ on PYTHONPATH)
docker run -p 8000:8000 ghcr.io/dvskaff/filings-rag-api
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from filings import llm as L
from filings.retrieval import MODES, FastEmbedder, FastReranker, HashEmbedder, OverlapReranker, Retriever

INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", Path(__file__).resolve().parents[1] / "index"))
STATE: dict = {}


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    doc_name: str | None = Field(
        None, description="Restrict to one filing, e.g. 3M_2018_10K (see GET /documents)"
    )
    k: int = Field(5, ge=1, le=20)
    method: str = Field("Hybrid + rerank", description=" | ".join(MODES))


class Passage(BaseModel):
    rank: int
    doc_name: str
    page: int = Field(description="1-based page number in the filing PDF")
    score: float
    text: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    doc_name: str = Field(description="The filing to answer from, e.g. 3M_2018_10K (see GET /documents)")


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Passage]
    passages: list[Passage]
    model: str
    seconds: float


def load(index_dir: Path | None = None) -> None:
    d = index_dir or INDEX_DIR
    meta = json.loads((d / "meta.json").read_text())
    chunks = pd.read_parquet(d / "chunks.parquet")
    emb = np.load(d / "embeddings.npy")
    if os.environ.get("RAG_BACKEND") == "fake":  # tests
        embedder, reranker, model = HashEmbedder(emb.shape[1]), OverlapReranker(), L.FakeLLM()
    else:
        cache = os.environ.get("RAG_MODEL_DIR")
        embedder = FastEmbedder(meta["embedder"], cache_dir=cache)
        reranker = FastReranker(meta["reranker"], cache_dir=cache)
        model = L.LlamaCppLLM(meta["llm_repo"], meta["llm_file"], meta["llm_name"])
    STATE.update(meta=meta, retriever=Retriever(chunks, emb, embedder, reranker), llm=model,
                 documents=sorted(chunks["doc_name"].unique()))  # fmt: skip


@asynccontextmanager
async def lifespan(app: FastAPI):
    load()
    yield


app = FastAPI(title="Financial Filings Q&A API", version="1.0", lifespan=lifespan,
              description="Answers questions about SEC filings with citations to filing pages, or says it doesn't know.")  # fmt: skip


def _passage(h) -> Passage:
    return Passage(rank=h.rank, doc_name=h.doc_name, page=h.page + 1, score=round(h.score, 4), text=h.text)


@app.get("/health")
def health() -> dict:
    m = STATE["meta"]
    return {"status": "ok", "passages": m["passages"], "llm": m["llm_name"], "index": m.get("bundle")}


@app.get("/documents")
def documents() -> list[str]:
    return STATE["documents"]


@app.post("/search", response_model=list[Passage])
def search(req: SearchRequest) -> list[Passage]:
    if req.method not in MODES:
        raise HTTPException(422, f"method must be one of {MODES}")
    if req.doc_name and req.doc_name not in STATE["documents"]:
        raise HTTPException(404, f"Unknown filing {req.doc_name!r}; see GET /documents")
    hits = STATE["retriever"].search(req.question, req.doc_name, k=req.k, mode=req.method)
    return [_passage(h) for h in hits]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.doc_name not in STATE["documents"]:
        raise HTTPException(404, f"Unknown filing {req.doc_name!r}; see GET /documents")
    t0 = time.time()
    m = STATE["meta"]
    hits = STATE["retriever"].search(req.question, req.doc_name, k=m["context_k"], candidates=m["candidates"])
    answer = STATE["llm"].chat(L.rag_messages(req.question, req.doc_name, hits), 256)
    cited = [_passage(hits[i - 1]) for i in L.citations(answer, len(hits))]
    return AskResponse(answer=answer, refused=answer.strip().startswith("I don't know"), citations=cited,
                       passages=[_passage(h) for h in hits], model=STATE["llm"].name, seconds=round(time.time() - t0, 2))  # fmt: skip
