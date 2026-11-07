"""End-to-end evaluation run.

download benchmark -> parse PDFs into passages -> embed (cached) -> retrieval evaluation (quality gate) ->
answer every question three ways (RAG, closed book, wrong filing) with the local LLM (cached) -> grade ->
grader validation against human labels -> API index bundle, dashboard, README.

python -m filings.pipeline              # real models and the FinanceBench sample
python -m filings.pipeline --fixture    # offline: synthetic filings and stand-in models (tests and CI)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import corpus, evaluate, retrieval
from . import llm as L
from .config import Config

log = logging.getLogger("filings")


def embeddings(cfg: Config, chunks, embedder) -> np.ndarray:
    """Passage embeddings, recomputed only when the passages or the embedding model change."""
    cache = cfg.data_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    key = f"{corpus.fingerprint(chunks)}-{embedder.name}"
    meta, arr = cache / "embeddings.json", cache / "embeddings.npy"
    if meta.exists() and arr.exists() and json.loads(meta.read_text()).get("key") == key:
        log.info("Embeddings: reusing cache")
        return np.load(arr)
    t0 = time.time()
    emb = embedder.passages(chunks["text"].tolist())
    np.save(arr, emb)
    meta.write_text(json.dumps({"key": key, "shape": list(emb.shape)}))
    log.info("Embeddings: %s passages in %.0fs", f"{len(emb):,}", time.time() - t0)
    return emb


def write_bundle(cfg: Config, chunks, emb) -> str:
    """What the API image needs: passages + embeddings + model names. Returns a content hash (the image tag)."""
    b = cfg.bundle_dir
    b.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(b / "chunks.parquet", index=False)
    np.save(b / "embeddings.npy", emb)
    meta = {"embedder": cfg.embedder, "reranker": cfg.reranker, "llm_repo": cfg.llm_repo, "llm_file": cfg.llm_file,
            "llm_name": cfg.llm_name, "passages": len(chunks), "documents": sorted(chunks["doc_name"].unique()),
            "context_k": cfg.context_k, "candidates": cfg.candidates}  # fmt: skip
    digest = hashlib.sha256(
        (corpus.fingerprint(chunks) + json.dumps(meta, sort_keys=True)).encode()
    ).hexdigest()[:12]
    (b / "meta.json").write_text(json.dumps({**meta, "bundle": digest}, indent=2))
    return digest


def run(cfg: Config) -> dict:
    t0 = time.time()
    bench = corpus.download(cfg)
    q = corpus.load_questions(bench["dir"], cfg.question_limit)
    chunks = corpus.build_chunks(cfg, bench["dir"], bench["documents"])
    embedder, reranker = retrieval.load_models(cfg)
    emb = embeddings(cfg, chunks, embedder)
    retriever = retrieval.Retriever(chunks, emb, embedder, reranker)

    ranks = evaluate.retrieval_eval(retriever, q, cfg.candidates)
    rsum = evaluate.retrieval_summary(ranks)
    best = rsum[(rsum["setting"] == "Single filing") & (rsum["method"] == "Hybrid + rerank")].iloc[0]
    log.info(
        "Retrieval (single filing, hybrid + rerank): recall@5 %.3f, MRR %.3f",
        best["recall@5"],
        best["mrr@10"],
    )
    baseline_file = cfg.data_dir / "cache" / "baseline.json"
    previous = json.loads(baseline_file.read_text())["recall@5"] if baseline_file.exists() else None
    if best["recall@5"] < cfg.min_recall_at_5:
        raise RuntimeError(
            f"Quality gate: recall@5 {best['recall@5']:.3f} is below {cfg.min_recall_at_5}; not publishing"
        )
    if previous is not None and best["recall@5"] < previous - cfg.max_recall_drop:
        raise RuntimeError(
            f"Quality gate: recall@5 fell from {previous:.3f} to {best['recall@5']:.3f}; not publishing"
        )

    model = L.CachedLLM(L.load_llm(cfg), cfg.data_dir / "cache" / "generations.jsonl")
    t1 = time.time()
    gen = evaluate.generation_eval(cfg, retriever, model, q)
    gsum = evaluate.generation_summary(gen)
    log.info("Generation: %d answers in %.0fs (%d cached)", len(gen), time.time() - t1, model.hits)
    validation = evaluate.grader_validation(corpus.load_reference(bench["dir"]), q)

    bundle = write_bundle(cfg, chunks, emb)
    baseline_file.write_text(
        json.dumps({"recall@5": float(best["recall@5"]), "mrr@10": float(best["mrr@10"])})
    )
    run_info = {"questions": len(q), "documents": len(bench["documents"]), "passages": len(chunks),
                "retrieval": rsum.to_dict(orient="records"), "generation": gsum, "grader": validation,
                "models": {"embedder": embedder.name, "reranker": reranker.name, "llm": model.name},
                "bundle": bundle, "cached_answers": model.hits, "seconds": round(time.time() - t0)}  # fmt: skip
    from .readme import update_readme
    from .report import build_site

    summary = build_site(cfg, run_info, gen, ranks, q)
    if not cfg.fixture:
        update_readme(cfg.readme_path, summary)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"bundle={bundle}\n")
    log.info("Done in %ss", run_info["seconds"])
    return run_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fixture", action="store_true", help="offline run on synthetic filings")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.fixture:
        base = Path(os.environ.get("RAG_FIXTURE_DIR", "/tmp/rag-fixture"))
        shutil.rmtree(base, ignore_errors=True)
        os.environ.update({"RAG_FIXTURE": "1", "RAG_BACKEND": os.environ.get("RAG_BACKEND", "fake")})
        for k, sub in (("RAG_DATA_DIR", "data"), ("RAG_SITE_DIR", "site")):
            os.environ.setdefault(k, str(base / sub))
    info = run(Config())
    print(json.dumps({k: info[k] for k in ("questions", "passages", "bundle", "seconds")}))


if __name__ == "__main__":
    main()
