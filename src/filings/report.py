"""Dashboard (site/index.html), API reference (site/api.html) and results.json."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from .config import ROOT, Config
from .report_base import BLUE, CSS, LAYOUT_JS, VIOLET, _card, _div, _style, _table

GOOD, BAD, REF = "#1a7f37", "#c4352b", "#b5b3ab"
EXTRA_CSS = """
.cards { grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); }
main a { color:#2a78d6; }
pre { background:#f0efec; border-radius:8px; padding:12px; overflow-x:auto; font-size:12.5px; white-space:pre-wrap; }
td.l:first-child { min-width:140px; }
.qa td, .qa th { vertical-align:top; white-space:normal; text-align:left; font-size:12.5px; }
.tag { display:inline-block; border-radius:999px; padding:0 8px; font-size:11.5px; font-weight:600; }
.tag.Correct { background:#e3f3e7; color:#1a7f37; } .tag.Incorrect { background:#fbe3e1; color:#c4352b; }
.tag.Refusal { background:#f0efec; color:#52514e; }
"""
SETTINGS = ["RAG (single filing)", "Closed book (no retrieval)", "Wrong filing (should refuse)"]
REF_NAMES = {"singleStore": "GPT-4 Turbo + retrieval, single filing", "sharedStore": "GPT-4 Turbo + retrieval, all filings",
             "closedBook": "GPT-4 Turbo, closed book", "oracle": "GPT-4 Turbo given the evidence page"}  # fmt: skip


def _pct(v, d=0) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:.{d}f}%"


def fig_retrieval(rsum: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for setting, color in (("Single filing", BLUE), ("All filings", VIOLET)):
        d = rsum[rsum["setting"] == setting]
        fig.add_trace(go.Bar(x=d["method"], y=d["recall@5"], name=setting, marker_color=color,
                             text=[f"{v:.0%}" for v in d["recall@5"]], textposition="outside",
                             hovertemplate="%{x}: %{y:.1%}<extra>" + setting + "</extra>"))  # fmt: skip
    fig.update_traces(marker_line_width=0)
    _style(fig, "Recall@5: the gold evidence page is among the top 5 passages", height=360, hovermode="closest",
           barmode="group", bargap=0.3)  # fmt: skip
    fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
    return fig


def fig_outcomes(rows: list[tuple[str, float, float, float]]) -> go.Figure:
    names = [r[0] for r in rows][::-1]
    fig = go.Figure()
    for i, (label, color) in enumerate((("Correct", GOOD), ("Incorrect", BAD), ("Refusal", REF))):
        vals = [r[i + 1] for r in rows][::-1]
        fig.add_trace(go.Bar(y=names, x=vals, orientation="h", name=label, marker_color=color,
                             text=[f"{v:.0%}" if v >= 0.06 else "" for v in vals], textposition="inside",
                             hovertemplate="%{y}<br>" + label + ": %{x:.1%}<extra></extra>"))  # fmt: skip
    fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=1)
    _style(fig, "Answer outcomes (auto-graded with the same grader)", height=140 + 44 * len(rows), hovermode="closest",
           barmode="stack", margin=dict(l=300, r=20, t=74, b=40))  # fmt: skip
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    return fig


def api_page(cfg: Config) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "serving"))
    from app.main import app

    (cfg.site_dir / "openapi.json").write_text(json.dumps(app.openapi(), indent=2))
    (cfg.site_dir / "api.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Filings Q&A API reference</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'></head><body>"
        "<redoc spec-url='openapi.json'></redoc>"
        "<script src='https://cdn.jsdelivr.net/npm/redoc@2.5.1/bundles/redoc.standalone.js'></script></body></html>"
    )


def build_site(cfg: Config, run: dict, gen: pd.DataFrame, ranks: pd.DataFrame, q: pd.DataFrame) -> dict:
    site = cfg.site_dir
    site.mkdir(parents=True, exist_ok=True)
    built_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rsum = pd.DataFrame(run["retrieval"])
    g, val = run["generation"], run["grader"]
    rag, closed, wrong = (g.get(s, {}) for s in SETTINGS)
    best = rsum[(rsum["setting"] == "Single filing") & (rsum["method"] == "Hybrid + rerank")].iloc[0]
    bm25 = rsum[(rsum["setting"] == "Single filing") & (rsum["method"] == "BM25")].iloc[0]
    banner = "<div class='synthetic'>SYNTHETIC TEST DATA: generated for automated tests; these numbers are not real</div>" if cfg.fixture else ""  # fmt: skip

    cards = "".join([
        _card("Correct answers (RAG)", _pct(rag.get("correct")), f"vs. {_pct(closed.get('correct'))} without retrieval"),
        _card("Recall@5", _pct(best["recall@5"]), f"hybrid + rerank; BM25 alone {_pct(bm25['recall@5'])}"),
        _card("Refuses when it should", _pct(wrong.get("refusal")), "asked about one company with another's filing"),
        _card("Cites the evidence page", _pct(rag.get("cited_evidence_rate")), "of answered questions"),
        _card("Grader vs. humans", _pct(val.get("agreement")), f"agreement on {val.get('n', 0)} labeled answers"),
        _card("Median latency", f"{rag.get('median_seconds', 0):.0f} s", "CPU only, retrieval + generation"),
    ])  # fmt: skip

    rrows = [[html.escape(r["setting"]), html.escape(r["method"]), *[_pct(r[f"recall@{k}"]) for k in (1, 3, 5, 10)],
              f"{r['mrr@10']:.3f}", _pct(r["right_filing@5"])] for r in rsum.to_dict(orient="records")]  # fmt: skip
    outcome_rows = [
        (f"This system: {s}", g[s]["correct"], g[s]["incorrect"], g[s]["refusal"]) for s in SETTINGS if s in g
    ]
    for m, v in val.get("modes", {}).items():
        outcome_rows.append(
            (f"{REF_NAMES.get(m, m)} (published)", v["auto_correct"], v["auto_incorrect"], v["auto_refusal"])
        )
    ref_rows = [[html.escape(REF_NAMES.get(m, m)), str(v["n"]), _pct(v["human_correct"]), _pct(v["auto_correct"])]
                for m, v in val.get("modes", {}).items()]  # fmt: skip
    kind_rows = []
    for k in ("numeric", "yes/no", "text"):
        if k in rag.get("by_kind", {}):
            kind_rows.append([k, str(rag["by_kind"][k]["n"]), _pct(rag["by_kind"][k]["correct"]),
                              _pct(closed.get("by_kind", {}).get(k, {}).get("correct")),
                              _pct(val.get("by_kind", {}).get(k))])  # fmt: skip
    ex = gen[gen["setting"] == SETTINGS[0]].copy()
    ex["context_has_evidence"] = ex["context_has_evidence"].astype(bool)
    sample = pd.concat(
        [ex[ex["grade"] == lab].head(n) for lab, n in (("Correct", 6), ("Incorrect", 4), ("Refusal", 2))]
    )
    qa_rows = [[f"{html.escape(r.question)}<div class='muted'>{html.escape(r.doc_name)} · retrieved pages {r.pages}</div>",
                html.escape(str(r.gold))[:300], html.escape(r.answer)[:500],
                f"<span class='tag {r.grade}'>{r.grade}</span>"] for r in sample.itertuples()]  # fmt: skip
    has_ev, no_ev = rag.get("correct_when_evidence_retrieved"), None
    rag_rows = ex
    if len(rag_rows):
        miss = rag_rows[~rag_rows["context_has_evidence"]]
        no_ev = float((miss["grade"] == "Correct").mean()) if len(miss) else None
    example_req = ('curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \\\n'
                   '     -d \'{"question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?", '
                   '"doc_name": "3M_2018_10K"}\'')  # fmt: skip

    api_page(cfg)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Financial Filings Q&amp;A</title>
<meta name="description" content="Retrieval-augmented question answering over SEC filings, evaluated on the FinanceBench benchmark: hybrid retrieval, reranking, a local open LLM, citations and refusal checks.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📑</text></svg>">
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
<style>{CSS}{EXTRA_CSS}</style></head>
<body>{banner}
<header><div class="wrap">
  <h1>Financial filings Q&amp;A: RAG and LLM evaluation</h1>
  <p>Can a system answer analysts' questions about 10-Ks, 10-Qs and earnings reports correctly, show where the answer
  came from, and say "I don't know" instead of guessing? Evaluated on the {run["questions"]} open-source FinanceBench
  questions over {run["documents"]} real filings ({run["passages"]:,} passages), using only open models on a CPU.</p>
  <a class="btn" href="questions.html">Explore every question</a><a class="btn" href="api.html">API reference</a>
  <a class="btn" href="{cfg.repo_url}">Code on GitHub</a>
</div></header>
<main>
<div class="cards">{cards}</div>

<section><h2>Retrieval: does it find the right page?</h2>
<p>A passage counts as relevant if it comes from a page FinanceBench's annotators marked as evidence. Keyword search
(BM25) and embedding search ({html.escape(run["models"]["embedder"])}) are combined with reciprocal-rank fusion, then
the top {cfg.candidates} candidates are reranked by a cross-encoder ({html.escape(run["models"]["reranker"])}).
"Single filing" searches the filing the question is about; "All filings" searches all {run["documents"]}.</p>
{_div(fig_retrieval(rsum))}
{_table(["Setting", "Method", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "Right filing in top 5"], rrows, left_cols=2)}
</section>

<section><h2>Answers: correct, wrong, or "I don't know"?</h2>
<p>Each question is answered three ways by {html.escape(run["models"]["llm"])}: with the top {cfg.context_k} passages
from its filing (RAG), with no retrieval at all (closed book), and, for every {cfg.unanswerable_every}th question,
with passages from a <i>different company's</i> filing, where the only right answer is to refuse. For reference, the
same grader scores FinanceBench's published GPT-4 Turbo answers.</p>
{_div(fig_outcomes(outcome_rows))}
<div class="grid2"><div><h3>By answer type (RAG vs. closed book)</h3>
{_table(["Answer type", "Questions", "RAG", "Closed book", "Grader agreement"], kind_rows)}
<p class="muted">Citations: {_pct(rag.get("citation_rate"))} of answered questions cite an excerpt, and
{_pct(rag.get("cited_evidence_rate"))} cite the gold evidence page. The evidence page was retrieved for
{_pct(rag.get("context_has_evidence"))} of questions; when it was, {_pct(has_ev)} were answered correctly, and when it
was not, {_pct(no_ev)}. That splits the errors into retrieval misses and reasoning or reading errors.</p></div>
<div><h3>Published GPT-4 Turbo results (FinanceBench)</h3>
{_table(["Setting", "Questions", "Correct: human", "Correct: grader"], ref_rows)}</div></div>
</section>

<section><h2>Can the grader be trusted?</h2>
<p>Answers are graded automatically: numbers must match the gold answer within 1.5% (in any of the usual units),
yes/no answers must have the same polarity, free-text answers must contain at least half of the gold answer's key
terms (words the question already contains don't count, so restating the question earns nothing), and refusals are
detected by phrasing. Checked against FinanceBench's {val.get("n", 0)} human-labeled GPT-4 answers, it agrees
{_pct(val.get("agreement"))} of the time ({_pct(val.get("by_kind", {}).get("numeric"))} on numeric answers, less on
free text). On those answers it marks fewer correct than the human reviewers did. The grader was tuned on those
labels, so the agreement figure is optimistic for new data, and free-text grades are approximate.</p>
</section>

<section><h2>Example answers</h2>
<p>A few examples below; the <b><a href="questions.html">question explorer</a></b> has all {run["questions"]}, with
the passages the model saw, which of them came from the gold evidence page, and the closed-book and wrong-filing
answers.</p>
<div class="tablewrap"><table class="qa"><thead><tr><th>Question</th><th>Gold answer</th><th>System answer</th><th>Grade</th></tr></thead>
<tbody>{"".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in qa_rows)}</tbody></table></div>
</section>

<section><h2>Serving</h2>
<p>The same retriever and model are packaged as a FastAPI service in a Docker image on GitHub Container Registry,
with the passage index and all three models inside: no API keys, no GPU. It is built, smoke-tested with a real
question and pushed whenever the index or models change.</p>
<pre>docker run -p 8000:8000 {cfg.image}:latest
{html.escape(example_req)}</pre>
<p><a href="api.html">API reference</a> · <code>GET /health</code>, <code>GET /documents</code>,
<code>POST /search</code> (retrieval only), <code>POST /ask</code> (answer with citations).</p>
</section>

<section><h2>How it works</h2><ul class="method">
<li><b>Data:</b> the FinanceBench open-source sample (Patronus AI): {run["questions"]} questions written by financial
analysts, with gold answers and evidence pages, about {run["documents"]} public-company SEC filings and earnings
reports (PDF). The repository is pinned to a commit.</li>
<li><b>Indexing:</b> PDF text per page (pdfium), split into ~{cfg.chunk_chars:,}-character passages on line
boundaries so table rows stay whole. Each passage keeps its filing and page number and starts with a header (company,
filing, page title), so a passage from the middle of a table still says what it belongs to.</li>
<li><b>Generation:</b> a small instruct model run with llama.cpp, temperature 0, told to answer only from the
excerpts, cite them, and reply "I don't know" otherwise. Answers are cached by prompt, so unchanged re-runs are free.</li>
<li><b>Quality gate:</b> the run stops before publishing if single-filing recall@5 falls below {cfg.min_recall_at_5:.0%}
or drops more than {cfg.max_recall_drop * 100:.0f} points from the last published run.</li>
<li><b>Limits:</b> a 1.5B-parameter model on a CPU is far weaker than frontier models at multi-step financial
arithmetic; 150 questions give wide confidence intervals (±8 points); free-text grading is approximate.</li>
</ul></section>
</main>
<footer>Benchmark: FinanceBench (Patronus AI) · Built {built_at} · <a href="{cfg.repo_url}">Source code</a></footer>
<script>{LAYOUT_JS}</script>
</body></html>"""
    (site / "index.html").write_text(page, encoding="utf-8")
    (site / ".nojekyll").touch()
    summary = {**{k: run[k] for k in ("questions", "documents", "passages", "models", "bundle", "grader")},
               "retrieval_best": best.to_dict(), "retrieval_bm25": bm25.to_dict(), "generation": g,
               "synthetic": cfg.fixture, "built_at": built_at, "image": cfg.image}  # fmt: skip
    (site / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
