import numpy as np

from filings import corpus, fixture
from filings.retrieval import BM25, MODES, tokenize


def test_pdf_round_trip_and_chunks(cfg, tmp_path):
    pdf = tmp_path / "x.pdf"
    fixture.write_pdf(pdf, ["Line one (a)\nLine two 1,234", "Second page"])
    pages = corpus.pdf_pages(pdf)
    assert len(pages) == 2 and "Line two 1,234" in pages[0] and "(a)" in pages[0]
    chunks = corpus.chunk_page("\n".join(f"row {i} " + "x" * 40 for i in range(60)), size=300, overlap=60)
    assert len(chunks) > 5 and all(len(c) <= 360 for c in chunks)
    assert all(c.split("\n")[0].startswith("row") for c in chunks)  # breaks only on line ends


def test_chunks_keep_page_numbers(bench):
    b, chunks, _ = bench
    assert set(chunks["doc_name"]) == set(b["documents"])
    row = chunks[chunks["text"].str.contains("Purchases of property")].iloc[0]
    assert row["page"] == 1 and row["doc_name"] == "ACMEWIDGETS_2023_10K"


def test_bm25_prefers_rare_matching_terms():
    bm = BM25(["net income was high", "capital expenditure purchases of property", "income statement"])
    s = bm.scores("capital expenditure")
    assert np.argmax(s) == 1 and s[0] == 0
    assert "the" not in tokenize("What is the FY2023 capex?")


def test_search_modes_and_filters(bench):
    _, _, r = bench
    for mode in MODES:
        hits = r.search(
            "Acme Widgets capital expenditure purchases of property", "ACMEWIDGETS_2023_10K", k=3, mode=mode
        )
        assert hits and all(h.doc_name == "ACMEWIDGETS_2023_10K" for h in hits)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    top = r.search("Blue Harbor Foods dividends per share", None, k=1)[0]
    assert top.doc_name == "BLUEHARBOR_2023_10K" and top.page == 2
    assert r.search("anything", "NOT_A_FILING") == []
