"""Question explorer: a page (site/questions.html) for browsing every benchmark question and what the system did with
it: the passages it retrieved (and whether they came from a gold evidence page), its answer with citations, the
closed-book answer, the wrong-filing control, the gold answer and FinanceBench's evidence, and the grades.

Everything is exported to data/questions.json after each evaluation; the page filters and renders it in the browser.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
RAG, CLOSED, WRONG = "RAG (single filing)", "Closed book (no retrieval)", "Wrong filing (should refuse)"
MAX_EVIDENCE_CHARS = 2500


def _clean(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def export(gen: pd.DataFrame, q: pd.DataFrame) -> list[dict]:
    by = {s: d.set_index("financebench_id") for s, d in gen.groupby("setting")}
    out = []
    for r in q.itertuples():
        fid = r.financebench_id
        rag = by.get(RAG, pd.DataFrame()).loc[fid] if fid in by.get(RAG, pd.DataFrame()).index else None
        if rag is None:
            continue
        closed = by[CLOSED].loc[fid] if CLOSED in by and fid in by[CLOSED].index else None
        wrong = by[WRONG].loc[fid] if WRONG in by and fid in by[WRONG].index else None
        evidence = [{"doc_name": e.get("evidence_doc_name") or e.get("doc_name") or r.doc_name,
                     "page": int(e["evidence_page_num"]) + 1,
                     "text": _clean(e.get("evidence_text"))[:MAX_EVIDENCE_CHARS]}
                    for e in (r.evidence if isinstance(r.evidence, list) else [])]  # fmt: skip
        out.append({
            "id": fid, "company": _clean(getattr(r, "company", "")), "doc_name": r.doc_name,
            "question_type": _clean(getattr(r, "question_type", "")), "question": r.question,
            "gold": _clean(r.answer), "justification": _clean(getattr(r, "justification", "")),
            "kind": rag["kind"], "evidence": evidence,
            "rag": {"answer": rag["answer"], "grade": rag["grade"], "passages": list(rag["passages"]),
                    "evidence_retrieved": bool(rag["context_has_evidence"]),
                    "cited_evidence": bool(rag["cited_evidence"])},
            "closed": None if closed is None else {"answer": closed["answer"], "grade": closed["grade"]},
            "wrong": None if wrong is None else {"doc_name": wrong["doc_name"], "answer": wrong["answer"],
                                                 "grade": wrong["grade"]},
        })  # fmt: skip
    return out


# --------------------------------------------------------------------------- page
PAGE_CSS = """
.explorer { display:grid; grid-template-columns:minmax(0,360px) minmax(0,1fr); gap:20px; align-items:start; }
@media (max-width: 900px) { .explorer { grid-template-columns:minmax(0,1fr); } }
.filters { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
.filters input[type=search], .filters select { font:inherit; font-size:13.5px; padding:6px 9px; border:1px solid #c3c2b7;
  border-radius:8px; background:#fff; }
.filters input[type=search] { flex:1 1 100%; }
.chip-btn { border:1px solid #c3c2b7; background:#fff; border-radius:999px; padding:3px 11px; font-size:13px;
  color:#104281; cursor:pointer; }
.chip-btn.on { background:#104281; color:#fff; border-color:#104281; }
#count { font-size:12.5px; color:#898781; margin:2px 0 6px; }
.qlist { max-height:72vh; overflow:auto; border:1px solid #e1e0d9; border-radius:10px; background:#fff; }
@media (max-width: 900px) { .qlist { max-height:340px; } }
.qitem { padding:8px 10px; border-bottom:1px solid #eeede8; cursor:pointer; font-size:13px; line-height:1.4; }
.qitem:hover { background:#f6f6f3; } .qitem.sel { background:#e3eefb; }
.qitem .co { font-size:11.5px; color:#898781; display:flex; justify-content:space-between; gap:6px; }
.detail h2 { margin-top:0; font-size:18px; line-height:1.35; }
.meta { font-size:12.5px; color:#898781; margin:-2px 0 12px; }
.answers { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr)); gap:10px; margin-bottom:14px; }
.box { border:1px solid #e1e0d9; border-radius:10px; padding:10px 12px; background:#fff; }
.box .lbl { font-size:11.5px; text-transform:uppercase; letter-spacing:.05em; color:#898781; margin-bottom:4px;
  display:flex; justify-content:space-between; gap:8px; align-items:center; }
.box p { margin:0; font-size:13.5px; white-space:pre-wrap; overflow-wrap:anywhere; }
.box.gold { border-color:#b9dcc3; background:#f3faf5; }
.psg { border:1px solid #e1e0d9; border-left:4px solid #e1e0d9; border-radius:8px; padding:8px 10px; margin:8px 0;
  background:#fff; }
.psg.ev { border-left-color:#1a7f37; }
.psg .hd { font-size:12.5px; color:#52514e; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.psg pre { white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
  margin:6px 0 0; max-height:220px; overflow:auto; color:#0b0b0b; }
.flag { display:inline-block; border-radius:999px; padding:0 8px; font-size:11.5px; font-weight:600; }
.flag.ev { background:#e3f3e7; color:#1a7f37; } .flag.cite { background:#e3eefb; color:#104281; }
.flag.no { background:#f0efec; color:#52514e; }
"""

PAGE_JS = """
const data = await (await fetch("data/questions.json")).json();
const Q = data.questions;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
const GRADE = { Correct: "Correct", Incorrect: "Wrong", Refusal: "Refused" };
const tag = (g) => `<span class="tag ${g}">${GRADE[g] || g}</span>`;
let grade = "all", current = null;
for (const k of [...new Set(Q.map((q) => q.kind))].sort()) $("kind").insertAdjacentHTML("beforeend", `<option value="${k}">${k}</option>`);
for (const t of [...new Set(Q.map((q) => q.question_type).filter(Boolean))].sort())
  $("qtype").insertAdjacentHTML("beforeend", `<option value="${esc(t)}">${esc(t)}</option>`);
document.querySelectorAll("[data-grade]").forEach((b) => b.onclick = () => {
  grade = b.dataset.grade; document.querySelectorAll("[data-grade]").forEach((x) => x.classList.toggle("on", x === b)); render();
});
["search", "kind", "qtype", "evid"].forEach((id) => $(id).addEventListener("input", render));
function filtered() {
  const s = $("search").value.trim().toLowerCase(), k = $("kind").value, t = $("qtype").value, e = $("evid").value;
  return Q.filter((q) => (grade === "all" || q.rag.grade === grade) && (!k || q.kind === k) && (!t || q.question_type === t)
    && (!e || String(q.rag.evidence_retrieved) === e)
    && (!s || (q.question + " " + q.company + " " + q.doc_name).toLowerCase().includes(s)));
}
function render() {
  const list = filtered();
  $("count").textContent = `${list.length} of ${Q.length} questions · ` +
    ["Correct", "Incorrect", "Refusal"].map((g) => `${list.filter((q) => q.rag.grade === g).length} ${GRADE[g].toLowerCase()}`).join(" · ");
  $("list").innerHTML = list.map((q) => `<div class="qitem${q.id === current ? " sel" : ""}" data-id="${q.id}">
    <div class="co"><span>${esc(q.company)} · ${esc(q.doc_name.replace(/_/g, " "))}</span>${tag(q.rag.grade)}</div>${esc(q.question)}</div>`).join("")
    || "<p class='muted' style='padding:10px'>No questions match.</p>";
  $("list").querySelectorAll(".qitem").forEach((el) => el.onclick = () => show(el.dataset.id, true));
}
function passage(p, evidenceMarks = true) {
  const flags = [evidenceMarks ? (p.evidence ? '<span class="flag ev">gold evidence page</span>' : "") : "",
    p.cited ? '<span class="flag cite">cited in the answer</span>' : ""].join("");
  return `<div class="psg${evidenceMarks && p.evidence ? " ev" : ""}"><div class="hd"><b>[${p.n}]</b>
    ${esc(p.doc_name.replace(/_/g, " "))}, page ${p.page} · rerank score ${p.score.toFixed(2)} ${flags}</div><pre>${esc(p.text)}</pre></div>`;
}
function show(id, scroll = false) {
  const q = Q.find((x) => x.id === id) || Q[0]; if (!q) return;
  current = q.id; history.replaceState(null, "", "#" + q.id); render();
  const r = q.rag;
  const ev = q.evidence.map((e) => `<div class="psg ev"><div class="hd"><b>${esc(e.doc_name.replace(/_/g, " "))}, page ${e.page}</b>
    <span class="flag ev">annotated by FinanceBench</span></div><pre>${esc(e.text)}</pre></div>`).join("");
  $("detail").innerHTML = `<h2>${esc(q.question)}</h2>
    <div class="meta">${esc(q.company)} · ${esc(q.doc_name.replace(/_/g, " "))} · ${esc(q.question_type)} · ${q.kind} answer · ${esc(q.id)}</div>
    <div class="answers">
      <div class="box"><div class="lbl">System answer (RAG) ${tag(r.grade)}</div><p>${esc(r.answer)}</p></div>
      <div class="box gold"><div class="lbl">Gold answer</div><p>${esc(q.gold)}</p>
        ${q.justification ? `<p class="muted" style="margin-top:6px">${esc(q.justification)}</p>` : ""}</div>
      ${q.closed ? `<div class="box"><div class="lbl">Without retrieval (closed book) ${tag(q.closed.grade)}</div><p>${esc(q.closed.answer)}</p></div>` : ""}
      ${q.wrong ? `<div class="box"><div class="lbl">Given ${esc(q.wrong.doc_name.replace(/_/g, " "))} instead ${
        q.wrong.grade === "Refusal" ? '<span class="tag Correct">Refused (right)</span>' : '<span class="tag Incorrect">Answered (should refuse)</span>'}</div>
        <p>${esc(q.wrong.answer)}</p></div>` : ""}
    </div>
    <h3>Passages the model saw ${r.evidence_retrieved ? '<span class="flag ev">gold evidence retrieved</span>' : '<span class="flag no">gold evidence not retrieved</span>'}</h3>
    ${r.passages.map((p) => passage(p)).join("")}
    <details${r.evidence_retrieved ? "" : " open"}><summary>FinanceBench's evidence for this question</summary>${ev || "<p class='muted'>None listed.</p>"}</details>`;
  if (scroll && window.innerWidth < 900) $("detail").scrollIntoView({behavior: "smooth"});
}
render();
show(decodeURIComponent(location.hash.slice(1)) || (Q[0] && Q[0].id));
"""


def write_page(site: Path, questions: list[dict], run: dict, repo_url: str, synthetic: bool) -> None:
    from .report import EXTRA_CSS
    from .report_base import CSS

    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "questions.json").write_text(json.dumps({"questions": questions}, separators=(",", ":")))
    banner = "<div class='synthetic'>SYNTHETIC TEST DATA: built by the test suite; these numbers are not real</div>" if synthetic else ""  # fmt: skip
    chips = "".join(f"<button class='chip-btn{' on' if g == 'all' else ''}' data-grade='{g}'>{label}</button>"
                    for g, label in [("all", "All"), ("Correct", "Correct"), ("Incorrect", "Wrong"), ("Refusal", "Refused")])  # fmt: skip
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Filings Q&amp;A Question Explorer</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📑</text></svg>">
<style>{CSS}{EXTRA_CSS}{PAGE_CSS}</style></head><body>{banner}
<header><div class="wrap"><h1>Question explorer</h1>
<p>Every one of the {len(questions)} FinanceBench questions and what the system did with it: the passages it
retrieved from the filing (green when they come from a page FinanceBench's annotators marked as evidence), the
answer {html.escape(run["models"]["llm"])} gave with its citations, the answer without retrieval, and the grade
against the gold answer. Pick a question, or filter to see where it goes wrong.</p>
<a class="btn" href="./">Back to the dashboard</a><a class="btn" href="api.html">API reference</a>
<a class="btn" href="{repo_url}">Code on GitHub</a></div></header>
<main><section><div class="explorer"><div>
<div class="filters"><input type="search" id="search" placeholder="Search questions or companies">
{chips}
<select id="kind" aria-label="Answer type"><option value="">Any answer type</option></select>
<select id="qtype" aria-label="Question type"><option value="">Any question type</option></select>
<select id="evid" aria-label="Evidence retrieved"><option value="">Evidence retrieved or not</option>
<option value="true">Gold evidence retrieved</option><option value="false">Gold evidence missed</option></select></div>
<div id="count"></div><div class="qlist" id="list"></div></div>
<div class="detail" id="detail"><p class="muted">Loading…</p></div></div></section>
<p class="muted">Grades come from the project's automatic grader, which agrees with FinanceBench's human reviewers
{run["grader"].get("agreement", 0):.0%} of the time on their labeled answers; free-text grades are approximate. Passages
are shown as the model saw them, starting with the header the index adds (company, filing, page).</p></main>
<footer>Benchmark: FinanceBench (Patronus AI) · filings: SEC EDGAR · <a href="{repo_url}">Source code</a></footer>
<script type="module">{PAGE_JS}</script></body></html>"""
    (site / "questions.html").write_text(page, encoding="utf-8")
    log.info("Question explorer: %d questions", len(questions))
