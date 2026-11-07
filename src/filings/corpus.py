"""The benchmark and its documents: FinanceBench questions, the SEC filings they are about (PDF), page text and
overlapping passages ("chunks") that keep their page number so every answer can cite where it came from."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config

log = logging.getLogger(__name__)
RESULT_MODES = [
    "singleStore",
    "sharedStore",
    "closedBook",
    "oracle",
]  # published GPT-4 results with human labels


def session() -> requests.Session:
    s = requests.Session()
    s.mount(
        "https://",
        HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))),
    )
    s.headers["User-Agent"] = "filings-rag-eval (github.com/dvskaff/filings-rag-eval)"
    return s


def download(cfg: Config) -> dict:
    """Questions, document info, the published GPT-4 results, and the PDFs the questions refer to."""
    if cfg.fixture:
        from .fixture import make_benchmark

        return make_benchmark(cfg.data_dir / "benchmark")
    base = cfg.data_dir / "benchmark"
    (base / "pdfs").mkdir(parents=True, exist_ok=True)
    http = session()

    def fetch(path: str) -> Path:
        out = base / path
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            r = http.get(cfg.raw_url(path), timeout=300)
            r.raise_for_status()
            out.write_bytes(r.content)
        return out

    questions = fetch("data/financebench_open_source.jsonl")
    fetch("data/financebench_document_information.jsonl")
    for mode in RESULT_MODES:
        fetch(f"results/gpt-4-1106-preview_{mode}.jsonl")
    docs = sorted(set(pd.read_json(questions, lines=True)["doc_name"]))
    for d in docs:
        pdf = fetch(f"pdfs/{d}.pdf")
        if not pdf.read_bytes()[:5] == b"%PDF-":
            raise RuntimeError(f"{d}.pdf is not a PDF (Git LFS pointer or error page?)")
    log.info("Benchmark: %d documents downloaded", len(docs))
    return {"dir": base, "documents": docs}


def load_questions(base: Path, limit: int = 0) -> pd.DataFrame:
    q = pd.read_json(base / "data" / "financebench_open_source.jsonl", lines=True)
    # The files use "doc_name" inside each evidence item (the README says "evidence_doc_name"); accept both.
    q["evidence_pages"] = [
        sorted(
            {
                (e.get("evidence_doc_name") or e.get("doc_name") or doc, int(e["evidence_page_num"]))
                for e in ev
            }
        )
        for ev, doc in zip(q["evidence"], q["doc_name"], strict=True)
    ]
    q["answer"] = q["answer"].astype(str)
    return q.head(limit) if limit else q


def load_reference(base: Path) -> pd.DataFrame:
    """Published GPT-4-Turbo answers in four settings, each labeled by FinanceBench's human reviewers."""
    frames = []
    for mode in RESULT_MODES:
        path = base / "results" / f"gpt-4-1106-preview_{mode}.jsonl"
        if path.exists():
            frames.append(pd.read_json(path, lines=True).assign(mode=mode))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].str.replace(" Answer", "", regex=False)
    return df


# --------------------------------------------------------------------------- parsing and chunking
def pdf_pages(path: Path) -> list[str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return [pdf[i].get_textpage().get_text_bounded() for i in range(len(pdf))]
    finally:
        pdf.close()


def chunk_page(text: str, size: int, overlap: int) -> list[str]:
    """Split a page into overlapping windows, breaking on line ends so table rows stay whole."""
    lines = []
    for ln in text.replace("\r", "\n").split("\n"):
        ln = ln.strip()
        while len(ln) > size:  # a very long line (prose without line breaks) is cut at a space
            cut = ln.rfind(" ", 0, size) if " " in ln[:size] else size
            lines.append(ln[:cut])
            ln = ln[cut:].strip()
        if ln:
            lines.append(ln)
    chunks, cur = [], ""
    for line in lines:
        if cur and len(cur) + len(line) + 1 > size:
            chunks.append(cur)
            cur = cur[-overlap:].split("\n", 1)[-1] if overlap else ""
        cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def build_chunks(cfg: Config, base: Path, documents: list[str]) -> pd.DataFrame:
    """One row per passage: id, document, company, page (0-indexed, as in FinanceBench), text."""
    info = pd.read_json(base / "data" / "financebench_document_information.jsonl", lines=True).set_index(
        "doc_name"
    )
    rows = []
    for d in documents:
        company = info.loc[d, "company"] if d in info.index else d
        for page, text in enumerate(pdf_pages(base / "pdfs" / f"{d}.pdf")):
            # Each passage starts with a short header (company, filing, page title) so a passage cut from the middle
            # of a table still says which company and which statement it belongs to.
            title = " ".join(ln.strip() for ln in text.split("\n") if ln.strip())[:160]
            header = f"{company} | {d.replace('_', ' ')} | page {page + 1} | {title}"
            for j, c in enumerate(chunk_page(text, cfg.chunk_chars, cfg.chunk_overlap)):
                rows.append({"chunk_id": f"{d}:{page}:{j}", "doc_name": d, "company": company,
                             "doc_type": info.loc[d, "doc_type"] if d in info.index else "", "page": page,
                             "text": f"{header}\n{c}"})  # fmt: skip
    df = pd.DataFrame(rows)
    log.info("Chunks: %s passages from %d documents", f"{len(df):,}", len(documents))
    return df


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        json.dumps(df["chunk_id"].tolist() + df["text"].map(len).tolist()).encode()
    ).hexdigest()[:16]
