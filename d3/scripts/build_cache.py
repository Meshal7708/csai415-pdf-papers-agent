"""Regenerate the committed chunk/doc cache from the PDFs — the one command that
makes D3 reproduce from a clean clone.

    python3 scripts/build_cache.py

Steps:
  1. download the 60 PDFs listed in ../D2/data/papers.csv (skips ones already
     present; open-access arXiv, polite parallel fetch);
  2. run the *exact* D2 ingest (PDF -> page map -> 900/150 sliding-window chunks
     with provenance) over each PDF;
  3. write data/cache/documents.jsonl + data/cache/chunks.jsonl.

These two jsonl files are committed so the pipeline, evaluator, ablation,
notebook and tests all run offline with no Neo4j/Mongo/Qdrant services. After
this, run scripts/build_embeddings.py to (re)build data/embeddings.npz.

The chunking is deterministic, so re-running reproduces byte-identical chunks.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                      # .../d3
sys.path.insert(0, str(ROOT.parent / "D2" / "src"))            # reuse D2 ingest
from ingest import ingest_pdf                                   # noqa: E402

PAPERS_CSV = ROOT.parent / "D2" / "data" / "papers.csv"
PDF_DIR = ROOT / "data" / "pdfs"
CACHE = ROOT / "data" / "cache"
HEADERS = {"User-Agent": "CSAI415-D3/0.1 (academic coursework)"}
RUN_ID = "d3-regen"


def _download(row) -> tuple[str, bool]:
    pid = row["paper_id"]
    dest = PDF_DIR / f"{pid}.pdf"
    if dest.exists() and dest.stat().st_size > 10_000:
        return pid, True
    url = row.get("pdf_url") or f"https://arxiv.org/pdf/{pid}"
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            data = urllib.request.urlopen(req, timeout=30).read()
            if data[:4] == b"%PDF":
                dest.write_bytes(data)
                return pid, True
        except Exception:
            pass
    return pid, False


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(PAPERS_CSV)))

    print(f"Downloading {len(rows)} PDFs ...", flush=True)
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for pid, good in ex.map(_download, rows):
            ok += good
            if not good:
                print(f"  FAILED {pid}", flush=True)
    print(f"  {ok}/{len(rows)} present", flush=True)

    print("Ingesting -> chunks ...", flush=True)
    docs, chunks = [], []
    for r in rows:
        pid = r["paper_id"]
        pdf = PDF_DIR / f"{pid}.pdf"
        if not pdf.exists():
            continue
        n_pages, cks = ingest_pdf(pid, pdf, 900, 150, 120, max_pages=20)
        if not cks:
            continue
        docs.append({
            "_id": pid, "title": r["title"], "authors": r["authors"],
            "venue": r.get("venue", "arXiv"), "year": int(r["year"]),
            "topic": r["topics"], "doi": r.get("doi", ""),
            "pdf_url": r.get("pdf_url", ""), "n_pages": n_pages,
            "n_chunks": len(cks), "run_id": RUN_ID,
        })
        chunks.extend(c.to_doc(RUN_ID) for c in cks)

    with open(CACHE / "documents.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(CACHE / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(docs)} docs, {len(chunks)} chunks to {CACHE}", flush=True)
    print("Next: python3 scripts/build_embeddings.py", flush=True)


if __name__ == "__main__":
    main()
