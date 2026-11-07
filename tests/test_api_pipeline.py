import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from filings import pipeline
from filings.readme import update_readme

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run(cfg):
    return pipeline.run(cfg)


def test_pipeline_builds_site_bundle_and_readme(cfg, run):
    assert run["questions"] == 6 and run["passages"] > 0
    for f in ("index.html", "api.html", "openapi.json", "results.json"):
        assert (cfg.site_dir / f).exists()
    assert "SYNTHETIC TEST DATA" in (cfg.site_dir / "index.html").read_text()
    meta = json.loads((cfg.bundle_dir / "meta.json").read_text())
    assert meta["bundle"] == run["bundle"] and meta["passages"] == run["passages"]
    shutil.copy(ROOT / "README.md", cfg.readme_path)
    update_readme(cfg.readme_path, json.loads((cfg.site_dir / "results.json").read_text()))
    assert "Retrieval recall@5" in cfg.readme_path.read_text()


def test_quality_gate_stops_publishing(cfg):
    from dataclasses import replace

    with pytest.raises(RuntimeError, match="Quality gate"):
        pipeline.run(replace(cfg, min_recall_at_5=1.01))
    assert not (cfg.site_dir / "index.html").exists()


def test_api(cfg, run):
    import app.main as api

    api.INDEX_DIR = cfg.bundle_dir
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        docs = c.get("/documents").json()
        assert docs == ["ACMEWIDGETS_2023_10K", "BLUEHARBOR_2023_10K"]
        hits = c.post("/search", json={"question": "Blue Harbor dividends", "k": 2}).json()
        assert len(hits) == 2 and hits[0]["page"] >= 1
        r = c.post("/ask", json={"question": "What dividend per share did Blue Harbor Foods pay in 2023?",
                                 "doc_name": "BLUEHARBOR_2023_10K"}).json()  # fmt: skip
        assert "2.40" in r["answer"] and r["citations"] and not r["refused"]
        wrong = c.post(
            "/ask",
            json={"question": "What dividend did Blue Harbor Foods pay?", "doc_name": "ACMEWIDGETS_2023_10K"},
        )
        assert wrong.json()["refused"]
        assert c.post("/ask", json={"question": "x?x", "doc_name": "NOPE"}).status_code == 404
        assert c.post("/search", json={"question": "abc", "method": "magic"}).status_code == 422


def test_quality_gate_catches_regressions(cfg):
    import json as _json

    pipeline.run(cfg)  # first run publishes and records its recall@5
    base = cfg.data_dir / "cache" / "baseline.json"
    _json.loads(base.read_text())
    base.write_text(_json.dumps({"recall@5": 1.2, "mrr@10": 1.0}))  # pretend the last run was much better
    with pytest.raises(RuntimeError, match="fell from"):
        pipeline.run(cfg)
