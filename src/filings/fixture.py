"""SYNTHETIC filings and questions in the FinanceBench layout, used only by tests and the offline CI run.

Two invented companies, a few PDF pages each (written with a tiny PDF writer so no extra library is needed), questions
with evidence pages, and a fake "published results" file with human labels. Pages built from it carry a red banner.
"""

from __future__ import annotations

import json
from pathlib import Path

COMPANIES = {
    "ACMEWIDGETS_2023_10K": (
        "Acme Widgets",
        [
            "Acme Widgets Inc. Annual Report 2023\nTable of Contents\nBusiness overview and risk factors",
            "Consolidated Statement of Cash Flows\n(Millions) 2023 2022\nNet income $ 1,210 $ 1,050\n"
            "Purchases of property, plant and equipment (PP&E) (845) (790)\nNet cash provided by operating activities 2,140 1,980",
            "Consolidated Balance Sheet\n(Millions) 2023 2022\nTotal current assets 5,400 5,100\n"
            "Total current liabilities 3,600 3,550\nInventories 1,200 1,150",
            "Segment information\nAcme Widgets operates two segments: Industrial and Consumer.\n"
            "Industrial revenue grew 8% while Consumer revenue declined 2%.",
        ],
    ),
    "BLUEHARBOR_2023_10K": (
        "Blue Harbor Foods",
        [
            "Blue Harbor Foods Corporation Form 10-K 2023\nTable of Contents\nForward-looking statements",
            "Consolidated Statements of Income\n(Millions) 2023 2022\nNet sales $ 9,870 $ 9,410\n"
            "Cost of sales 6,120 5,940\nOperating income 1,480 1,360",
            "Dividends\nBlue Harbor Foods paid cash dividends of $2.40 per share in 2023 compared with $2.20 in 2022.",
        ],
    ),
}
QUESTIONS = [
    (
        "ACMEWIDGETS_2023_10K",
        "What is the FY2023 capital expenditure (in USD millions) for Acme Widgets?",
        "$845",
        1,
    ),
    (
        "ACMEWIDGETS_2023_10K",
        "What were Acme Widgets total current liabilities at the end of FY2023 in USD millions?",
        "3,600",
        2,
    ),
    (
        "ACMEWIDGETS_2023_10K",
        "Did Acme Widgets consumer segment revenue grow in 2023?",
        "No, consumer revenue declined 2%",
        3,
    ),
    ("BLUEHARBOR_2023_10K", "What was Blue Harbor Foods net sales in FY2023 in USD millions?", "$9,870", 1),
    (
        "BLUEHARBOR_2023_10K",
        "What dividend per share did Blue Harbor Foods pay in 2023?",
        "$2.40 per share",
        2,
    ),
    (
        "BLUEHARBOR_2023_10K",
        "What was Blue Harbor Foods operating income in FY2023 (USD millions)?",
        "1,480",
        1,
    ),
]


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[str]) -> None:
    """A minimal valid PDF: one Helvetica text page per string, one line per newline."""
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for text in pages:
        lines = text.split("\n")
        stream = "BT /F1 11 Tf 50 750 Td 14 TL " + " ".join(f"({_escape(ln)}) Tj T*" for ln in lines) + " ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")  # fmt: skip
        kids.append(len(objs))
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def make_benchmark(base: Path) -> dict:
    (base / "pdfs").mkdir(parents=True, exist_ok=True)
    (base / "data").mkdir(exist_ok=True)
    (base / "results").mkdir(exist_ok=True)
    info = []
    for doc, (company, pages) in COMPANIES.items():
        write_pdf(base / "pdfs" / f"{doc}.pdf", pages)
        info.append(
            {"doc_name": doc, "doc_type": "10K", "doc_period": 2023, "company": company, "doc_link": ""}
        )
    qs, results = [], []
    for i, (doc, question, answer, page) in enumerate(QUESTIONS):
        fid = f"financebench_id_{i:05d}"
        text = COMPANIES[doc][1][page]
        qs.append({"financebench_id": fid, "company": COMPANIES[doc][0], "doc_name": doc, "question_type": "metrics-generated",
                   "question_reasoning": "Information extraction", "domain_question_num": None, "question": question,
                   "answer": answer, "justification": "", "dataset_subset_label": "OPEN_SOURCE",
                   "evidence": [{"evidence_text": text, "doc_name": doc, "evidence_page_num": page,
                                 "evidence_text_full_page": text}]})  # fmt: skip
        results.append({"financebench_id": fid, "model_name": "gpt-4-1106-preview", "eval_mode": "singleStore",
                        "question": question, "gold_answer": answer,
                        "model_answer": answer if i % 3 else "The filing does not include this information.",
                        "label": "Correct Answer" if i % 3 else "Refusal"})  # fmt: skip
    for name, rows in (("financebench_open_source", qs), ("financebench_document_information", info)):
        (base / "data" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (base / "results" / "gpt-4-1106-preview_singleStore.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n"
    )
    return {"dir": base, "documents": sorted(COMPANIES)}
