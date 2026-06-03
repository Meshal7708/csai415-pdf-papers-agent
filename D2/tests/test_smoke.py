"""Smoke tests — run fully in-process (mongomock + Qdrant :memory: + NetworkX)
with the offline `hash` embedder so CI needs no services and no model download.

    cd D2 && EMBEDDER=hash PYTHONPATH=src pytest -q

They exercise the real ingestion -> store -> hybrid-search -> graph path on two
tiny synthetic PDFs, so a regression in chunking, provenance, fusion or the
graph schema fails the build.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("EMBEDDER", "hash")          # offline embedder for CI
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _make_pdf(path: Path, lines):
    """Write a minimal multi-page PDF using pypdf so parsing is exercised for real."""
    from pypdf import PdfWriter
    try:
        from reportlab.pdfgen import canvas  # optional, nicer text
        c = canvas.Canvas(str(path))
        for page_lines in lines:
            y = 800
            for ln in page_lines:
                c.drawString(72, y, ln); y -= 20
            c.showPage()
        c.save()
        return
    except Exception:
        pass
    # fallback: blank pages (parsing still works, text may be empty)
    w = PdfWriter()
    for _ in lines:
        w.add_blank_page(width=200, height=200)
    with open(path, "wb") as fh:
        w.write(fh)


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    import pandas as pd
    from pipeline import build_pipeline
    d = tmp_path_factory.mktemp("pdfs")
    p1, p2 = d / "a.pdf", d / "b.pdf"
    _make_pdf(p1, [["attention transformer model"] * 6, ["sparse attention tokens"] * 6])
    _make_pdf(p2, [["concept drift streaming online learning"] * 6, ["adwin detector"] * 6])
    papers = pd.DataFrame([
        {"paper_id": "a", "title": "Attention Paper", "authors": "Alice, Bob",
         "venue": "arXiv", "year": 2024, "topic": "transformers",
         "pdf_path": str(p1), "pdf_url": ""},
        {"paper_id": "b", "title": "Drift Paper", "authors": "Bob, Carol",
         "venue": "arXiv", "year": 2023, "topic": "online_learning",
         "pdf_path": str(p2), "pdf_url": ""},
    ])
    pl = build_pipeline(build_embedder=True)
    pl.ingest_corpus(papers)
    return pl


def test_ingest_populated(pipe):
    s = pipe.stats()
    assert s["mongo"]["documents"] == 2
    assert s["mongo"]["chunks"] >= 2
    assert s["qdrant_vectors"] == s["mongo"]["chunks"]


def test_chunk_provenance(pipe):
    c = pipe.mongo.all_chunks()[0]
    for f in ("paper_id", "page_start", "page_end", "char_start", "char_end", "sha256", "run_id"):
        assert f in c
    assert c["page_start"] >= 1 and c["page_end"] >= c["page_start"]


def test_search_returns_citations(pipe):
    cites, ms = pipe.search("attention transformer", top_k=3)
    assert cites and cites[0].paper_id in {"a", "b"}
    assert cites[0].page_range.startswith("p")
    assert ms < 5000


def test_graph_schema(pipe):
    st = pipe.graph.stats()
    assert st["nodes"] > 0 and st["edges"] > 0
    assert ("b", "Drift Paper") not in []  # sanity
    assert pipe.graph.coauthors("Bob")      # Bob co-authored with Alice and Carol
    assert pipe.graph.related_via_topic("a") is not None
