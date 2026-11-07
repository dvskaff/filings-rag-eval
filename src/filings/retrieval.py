"""Retrieval: BM25 keyword search, dense embedding search, reciprocal-rank fusion, and cross-encoder reranking.

Real models run on ONNX Runtime via fastembed (no GPU, no API key). `HashEmbedder` / `OverlapReranker` are small
deterministic stand-ins used by the offline tests.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = set("the a an of in to for and or is are was were be by on at as with that this its it from than which what "
           "how does did do has have had their there based answer question using use give response".split())  # fmt: skip


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


# --------------------------------------------------------------------------- models
class HashEmbedder:
    """Deterministic bag-of-words hashing vectors (tests only)."""

    name = "hash-embedder (test stand-in)"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FastEmbedder:
    def __init__(self, model: str, threads: int | None = None, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.name = model
        self.model = TextEmbedding(model, threads=threads, cache_dir=cache_dir)

    def passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.passage_embed(texts, batch_size=64)), dtype=np.float32)

    def query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self.model.query_embed(text))), dtype=np.float32)


class OverlapReranker:
    """Query-term overlap score (tests only)."""

    name = "overlap-reranker (test stand-in)"

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) / (1 + len(q)) for t in texts], dtype=float)


class FastReranker:
    def __init__(self, model: str, threads: int | None = None, cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self.model = TextCrossEncoder(model, threads=threads, cache_dir=cache_dir)

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.rerank(query, texts, batch_size=32)), dtype=float)


def load_models(cfg) -> tuple:
    if cfg.backend == "fake":
        return HashEmbedder(), OverlapReranker()
    return FastEmbedder(cfg.embedder), FastReranker(cfg.reranker)


# --------------------------------------------------------------------------- BM25
class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) over the passages, with sparse postings so filtered search stays fast."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        from scipy.sparse import csr_matrix

        docs = [tokenize(t) for t in texts]
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, toks in enumerate(docs):
            counts: dict[int, int] = {}
            for t in toks:
                j = self.vocab.setdefault(t, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            rows += [i] * len(counts)
            cols += list(counts)
            vals += list(counts.values())
        n = len(docs)
        tf = csr_matrix((vals, (rows, cols)), shape=(n, max(len(self.vocab), 1)), dtype=np.float32)
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1.0))
        tf = tf.tocoo()
        weight = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.matrix = csr_matrix((weight, (tf.row, tf.col)), shape=tf.shape).tocsc()

    def scores(self, query: str) -> np.ndarray:
        ids = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not ids:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return np.asarray(self.matrix[:, ids] @ self.idf[ids]).ravel()


# --------------------------------------------------------------------------- retriever
MODES = ["BM25", "Dense", "Hybrid (RRF)", "Hybrid + rerank"]


@dataclass
class Hit:
    rank: int
    chunk_id: str
    doc_name: str
    page: int
    score: float
    text: str


class Retriever:
    def __init__(self, chunks: pd.DataFrame, embeddings: np.ndarray, embedder, reranker, rrf_k: int = 60):
        self.chunks = chunks.reset_index(drop=True)
        self.emb = embeddings
        self.embedder, self.reranker, self.rrf_k = embedder, reranker, rrf_k
        self.bm25 = BM25(self.chunks["text"].tolist())
        self.doc_rows = {
            d: np.flatnonzero(self.chunks["doc_name"].to_numpy() == d)
            for d in self.chunks["doc_name"].unique()
        }

    def _rank(self, scores: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
        s = scores[rows]
        top = np.argsort(-s, kind="stable")[:n]
        return rows[top]

    def search(self, question: str, doc_name: str | None = None, k: int = 5, mode: str = "Hybrid + rerank",
               candidates: int = 30) -> list[Hit]:  # fmt: skip
        rows = (
            self.doc_rows.get(doc_name, np.array([], dtype=int)) if doc_name else np.arange(len(self.chunks))
        )
        if len(rows) == 0:
            return []
        bm = self.bm25.scores(question)
        dense = self.emb @ self.embedder.query(question)
        if mode == "BM25":
            order, score = self._rank(bm, rows, k), bm
        elif mode == "Dense":
            order, score = self._rank(dense, rows, k), dense
        else:
            pool = max(candidates, k)
            fused: dict[int, float] = {}
            for ranked in (self._rank(bm, rows, pool * 2), self._rank(dense, rows, pool * 2)):
                for r, i in enumerate(ranked):
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.rrf_k + r + 1)
            cand = np.array(sorted(fused, key=lambda i: -fused[i])[:pool])
            if mode == "Hybrid (RRF)":
                order, score = cand[:k], np.zeros(len(self.chunks))
                score[cand] = [fused[int(i)] for i in cand]
            else:
                rr = self.reranker.scores(question, self.chunks["text"].iloc[cand].tolist())
                pick = np.argsort(-rr, kind="stable")[:k]
                order, score = cand[pick], np.zeros(len(self.chunks))
                score[cand] = rr
        c = self.chunks
        return [Hit(r + 1, c.at[i, "chunk_id"], c.at[i, "doc_name"], int(c.at[i, "page"]), float(score[i]), c.at[i, "text"])
                for r, i in enumerate(order)]  # fmt: skip
