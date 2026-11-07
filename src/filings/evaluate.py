"""Evaluation: retrieval quality, answer accuracy, citations and abstention, plus grader validation."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import llm as L
from .grading import answer_kind, grade
from .retrieval import MODES, Retriever

log = logging.getLogger(__name__)
KS = (1, 3, 5, 10)


def _first_relevant(hits, evidence: list[tuple[str, int]]) -> int | None:
    ev = set(evidence)
    for h in hits:
        if (h.doc_name, h.page) in ev:
            return h.rank
    return None


def retrieval_eval(retriever: Retriever, q: pd.DataFrame, candidates: int) -> pd.DataFrame:
    """For every question, setting and method: the rank of the first passage from a gold evidence page."""
    rows = []
    for r in q.itertuples():
        for setting, doc in (("Single filing", r.doc_name), ("All filings", None)):
            for mode in MODES:
                hits = retriever.search(r.question, doc, k=max(KS), mode=mode, candidates=candidates)
                rows.append({"financebench_id": r.financebench_id, "setting": setting, "method": mode,
                             "rank": _first_relevant(hits, r.evidence_pages),
                             "doc_hit_5": r.doc_name in {h.doc_name for h in hits[:5]}})  # fmt: skip
    return pd.DataFrame(rows)


def retrieval_summary(ranks: pd.DataFrame) -> pd.DataFrame:
    def agg(d):
        rk = d["rank"]
        out = {f"recall@{k}": float((rk <= k).mean()) for k in KS}
        out["mrr@10"] = float((1 / rk).fillna(0).mean())
        out["right_filing@5"] = float(d["doc_hit_5"].mean())
        return pd.Series(out)

    return ranks.groupby(["setting", "method"], sort=False).apply(agg, include_groups=False).reset_index()


def wrong_filing(q: pd.DataFrame, r) -> str:
    """A filing from a different company (deterministic), for the unanswerable control questions."""
    others = sorted(set(q.loc[q["company"] != r.company, "doc_name"]))
    return others[int(str(r.financebench_id).split("_")[-1]) % len(others)] if others else r.doc_name


def generation_eval(cfg, retriever: Retriever, model, q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, r in enumerate(q.itertuples()):
        kind = answer_kind(r.answer)
        t0 = time.time()
        hits = retriever.search(
            r.question, r.doc_name, k=cfg.context_k, mode="Hybrid + rerank", candidates=cfg.candidates
        )
        ans = model.chat(L.rag_messages(r.question, r.doc_name, hits), cfg.max_answer_tokens)
        cited = L.citations(ans, len(hits))
        ev = set(r.evidence_pages)
        rows.append({"financebench_id": r.financebench_id, "setting": "RAG (single filing)", "question": r.question,
                     "doc_name": r.doc_name, "gold": r.answer, "kind": kind, "answer": ans, "grade": grade(ans, r.answer, kind, r.question),
                     "cited": cited, "context_has_evidence": any((h.doc_name, h.page) in ev for h in hits),
                     "cited_evidence": any((hits[i - 1].doc_name, hits[i - 1].page) in ev for i in cited),
                     "pages": [h.page + 1 for h in hits], "seconds": time.time() - t0})  # fmt: skip
        closed = model.chat(L.closed_messages(r.question), 128)
        rows.append({"financebench_id": r.financebench_id, "setting": "Closed book (no retrieval)", "question": r.question,
                     "doc_name": r.doc_name, "gold": r.answer, "kind": kind, "answer": closed,
                     "grade": grade(closed, r.answer, kind, r.question)})  # fmt: skip
        if n % cfg.unanswerable_every == 0:
            other = wrong_filing(q, r)
            hits = retriever.search(
                r.question, other, k=cfg.context_k, mode="Hybrid + rerank", candidates=cfg.candidates
            )
            wrong = model.chat(L.rag_messages(r.question, other, hits), cfg.max_answer_tokens)
            rows.append({"financebench_id": r.financebench_id, "setting": "Wrong filing (should refuse)",
                         "question": r.question, "doc_name": other, "gold": r.answer, "kind": kind, "answer": wrong,
                         "grade": grade(wrong, r.answer, kind, r.question)})  # fmt: skip
        if (n + 1) % 10 == 0:
            log.info("Generated answers for %d/%d questions", n + 1, len(q))
    return pd.DataFrame(rows)


def generation_summary(g: pd.DataFrame) -> dict:
    out = {}
    for setting, d in g.groupby("setting", sort=False):
        s = {"questions": int(len(d)), "correct": float((d["grade"] == "Correct").mean()),
             "incorrect": float((d["grade"] == "Incorrect").mean()), "refusal": float((d["grade"] == "Refusal").mean())}  # fmt: skip
        s["by_kind"] = {
            k: {"n": int(len(x)), "correct": float((x["grade"] == "Correct").mean())}
            for k, x in d.groupby("kind")
        }
        if setting.startswith("RAG"):
            d = d.astype({"context_has_evidence": bool, "cited_evidence": bool})
            answered = d[d["grade"] != "Refusal"]
            s["citation_rate"] = float(answered["cited"].map(len).gt(0).mean()) if len(answered) else 0.0
            s["cited_evidence_rate"] = float(answered["cited_evidence"].mean()) if len(answered) else 0.0
            s["context_has_evidence"] = float(d["context_has_evidence"].mean())
            ok = d[d["context_has_evidence"]]
            s["correct_when_evidence_retrieved"] = (
                float((ok["grade"] == "Correct").mean()) if len(ok) else 0.0
            )
            s["median_seconds"] = float(np.median(d["seconds"]))
        out[setting] = s
    return out


def grader_validation(reference: pd.DataFrame, q: pd.DataFrame) -> dict:
    """Run our grader on FinanceBench's published GPT-4 answers and compare with the human labels."""
    if reference.empty:
        return {}
    gold = q.set_index("financebench_id")["answer"]
    ref = reference[reference["financebench_id"].isin(gold.index)].copy()
    ref["gold"] = ref["financebench_id"].map(gold)
    ref["q"] = ref["financebench_id"].map(q.set_index("financebench_id")["question"])
    ref["kind"] = ref["gold"].map(answer_kind)
    ref["auto"] = [
        grade(a, g, k, qq)
        for a, g, k, qq in zip(ref["model_answer"], ref["gold"], ref["kind"], ref["q"], strict=True)
    ]
    agree = ref["auto"] == ref["label"]
    modes = {m: {"n": int(len(d)), "human_correct": float((d["label"] == "Correct").mean()),
                 **{f"auto_{k.lower()}": float((d["auto"] == k).mean()) for k in ("Correct", "Incorrect", "Refusal")}}
             for m, d in ref.groupby("mode", sort=False)}  # fmt: skip
    return {"n": int(len(ref)), "agreement": float(agree.mean()),
            "by_kind": {k: float(a.mean()) for k, a in agree.groupby(ref["kind"])},
            "confusion": pd.crosstab(ref["label"], ref["auto"]).to_dict(), "modes": modes}  # fmt: skip
