"""Seed all stores from data/papers.csv: ingest PDFs -> Mongo + Qdrant, build graph.

Usage:
    PYTHONPATH=src python3 scripts/seed_stores.py

Honours env vars (MONGO_URI / QDRANT_URL / NEO4J_URI). With none set it runs
fully in-process (mongomock + Qdrant :memory: + NetworkX) — handy for a smoke
run, but those stores are ephemeral, so for a persistent seed start the docker
stack first and export the URLs (see .env.example).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import build_pipeline  # noqa: E402

PAPERS_CSV = ROOT / "data" / "papers.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    if not PAPERS_CSV.exists():
        raise SystemExit("data/papers.csv missing — run scripts/download_pdfs.py first")
    papers = pd.read_csv(PAPERS_CSV, dtype={"paper_id": str})
    papers["topic"] = papers["topics"]            # pipeline expects a `topic` column
    pipe = build_pipeline(build_embedder=True)
    summary = pipe.ingest_corpus(papers, recreate_vectors=True)
    print("Ingestion summary:", json.dumps(summary, indent=2, default=str))
    print("Stats:", json.dumps(pipe.stats(), indent=2, default=str))
    (RESULTS / "ingest_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
