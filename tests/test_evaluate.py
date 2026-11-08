from filings import corpus, evaluate
from filings import llm as L


def test_retrieval_eval_finds_evidence_pages(cfg, bench):
    b, _, r = bench
    q = corpus.load_questions(b["dir"])
    s = evaluate.retrieval_summary(evaluate.retrieval_eval(r, q, 30))
    best = s[(s["setting"] == "Single filing") & (s["method"] == "Hybrid + rerank")].iloc[0]
    assert best["recall@5"] == 1.0 and 0 < best["mrr@10"] <= 1
    assert set(s["method"]) == set(evaluate.MODES) and set(s["setting"]) == {"Single filing", "All filings"}


def test_generation_eval_settings_and_cache(cfg, bench, tmp_path):
    b, _, r = bench
    q = corpus.load_questions(b["dir"])
    cached = L.CachedLLM(L.FakeLLM(), tmp_path / "gen.jsonl")
    g = evaluate.generation_eval(cfg, r, cached, q)
    s = evaluate.generation_summary(g)
    assert set(s) == {"RAG (single filing)", "Closed book (no retrieval)", "Wrong filing (should refuse)"}
    assert (
        s["Wrong filing (should refuse)"]["refusal"] == 1.0
    )  # the stand-in refuses other companies' filings
    assert s["RAG (single filing)"]["citation_rate"] == 1.0
    # the question explorer gets the passages each answer was based on, flagged as evidence / cited
    from filings import explorer

    rag = g[g["setting"] == "RAG (single filing)"]
    for row in rag.itertuples():
        assert [p["page"] for p in row.passages] == row.pages
        assert any(p["evidence"] for p in row.passages) == row.context_has_evidence
        assert any(p["evidence"] and p["cited"] for p in row.passages) == row.cited_evidence
    qs = explorer.export(g, q)
    assert len(qs) == len(q) and all(x["rag"]["passages"] and x["closed"] for x in qs)
    assert sum(x["wrong"] is not None for x in qs) == (g["setting"] == "Wrong filing (should refuse)").sum()
    assert all(e["page"] >= 1 and e["text"] for x in qs for e in x["evidence"])
    again = L.CachedLLM(L.FakeLLM(), tmp_path / "gen.jsonl")
    evaluate.generation_eval(cfg, r, again, q)
    assert again.misses == 0 and again.hits == len(g)


def test_citations_and_prompt():
    assert L.citations("Capex was $845 million [2][3], see also [9] and [2, 1]", 5) == [2, 3, 1]
    msgs = L.rag_messages("Q?", "DOC", [])
    assert "I don't know based on the provided filing." in msgs[0]["content"] and msgs[1]["content"].endswith(
        "Question: Q?"
    )


def test_grader_validation_uses_human_labels(bench):
    b, _, _ = bench
    v = evaluate.grader_validation(corpus.load_reference(b["dir"]), corpus.load_questions(b["dir"]))
    assert v["n"] == 6 and v["agreement"] == 1.0 and "singleStore" in v["modes"]
