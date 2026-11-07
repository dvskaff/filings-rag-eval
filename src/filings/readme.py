"""Rewrites the results block in README.md after each run."""

from __future__ import annotations

import re
from pathlib import Path


def _p(v) -> str:
    return "–" if v is None else f"{v:.0%}"


def results_block(s: dict) -> str:
    g, b, bm, v = s["generation"], s["retrieval_best"], s["retrieval_bm25"], s["grader"]
    rag = g.get("RAG (single filing)", {})
    closed = g.get("Closed book (no retrieval)", {})
    wrong = g.get("Wrong filing (should refuse)", {})
    ref = v.get("modes", {}).get("singleStore", {})
    lines = [
        f"_FinanceBench open-source sample: {s['questions']} questions · {s['documents']} filings · {s['passages']:,} "
        f"passages · LLM: {s['models']['llm']} · last run {s['built_at']}_",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Retrieval recall@5, single filing (hybrid + rerank) | {_p(b['recall@5'])} (BM25 alone {_p(bm['recall@5'])}; MRR {b['mrr@10']:.2f}) |",
        f"| Correct answers with RAG | {_p(rag.get('correct'))} (refused {_p(rag.get('refusal'))}, wrong {_p(rag.get('incorrect'))}) |",
        f"| Correct answers without retrieval (closed book) | {_p(closed.get('correct'))} |",
        f"| Refuses when given the wrong company's filing | {_p(wrong.get('refusal'))} of {wrong.get('questions', 0)} |",
        f"| Answers citing the gold evidence page | {_p(rag.get('cited_evidence_rate'))} |",
        f"| Grader agreement with FinanceBench human labels | {_p(v.get('agreement'))} of {v.get('n', 0)} answers |",
    ]
    if ref:
        lines.append(f"| Reference: published GPT-4 Turbo + retrieval (same grader / human label) | "
                     f"{_p(ref['auto_correct'])} / {_p(ref['human_correct'])} |")  # fmt: skip
    lines += ["", f"**API image:** `docker run -p 8000:8000 {s['image']}:latest`"]
    return "\n".join(lines)


def update_readme(readme: Path, summary: dict) -> None:
    text = readme.read_text()
    pattern = r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)"
    readme.write_text(
        re.sub(pattern, lambda m: m.group(1) + results_block(summary) + m.group(2), text, flags=re.S)
    )
