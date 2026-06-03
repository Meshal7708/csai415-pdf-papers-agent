"""Resumable, on-disk seeding for the no-Docker sandbox.

The in-process stores (mongomock / Qdrant :memory:) vanish when the process
exits, and embedding 60 PDFs on a SIMD-less CPU takes minutes — longer than a
single sandbox command may live. So this seeder is *resumable* and writes to
disk:

    data/qdrant_db/           on-disk Qdrant (persists across processes)
    data/cache/documents.jsonl
    data/cache/chunks.jsonl
    data/cache/ingested.json  manifest of completed paper_ids

Each invocation processes papers not yet in the manifest until a wall-clock
budget elapses, then exits cleanly. Re-run until it prints ALL_DONE. The
`eval_from_cache.py` step then reads these files back. On a laptop with Docker
you don't need any of this — `seed_stores.py` writes straight to the services.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import SETTINGS                       # noqa: E402
from embedder import get_embedder                 # noqa: E402
from ingest import ingest_pdf                      # noqa: E402
from stores.vector_store import VectorStore        # noqa: E402

import os
CACHE = Path(os.environ.get("CACHE_DIR") or (ROOT / "data" / "cache"))
CACHE.mkdir(parents=True, exist_ok=True)
# Qdrant local mode uses SQLite, which needs a POSIX-locking filesystem; on a
# networked mount that fails, so honour QDRANT_PATH (point it at local disk).
QDB = SETTINGS.qdrant_path or str(ROOT / "data" / "qdrant_db")
DOCS, CHUNKS, MANIFEST = CACHE / "documents.jsonl", CACHE / "chunks.jsonl", CACHE / "ingested.json"
BUDGET_S = 38


def load_manifest() -> set:
    return set(json.loads(MANIFEST.read_text())) if MANIFEST.exists() else set()


def main():
    papers = pd.read_csv(ROOT / "data" / "papers.csv", dtype={"paper_id": str})
    done = load_manifest()
    todo = [r for _, r in papers.iterrows() if r["paper_id"] not in done]
    if not todo:
        print(f"ALL_DONE — {len(done)} papers already ingested")
        return

    embedder = get_embedder(SETTINGS.embedder, SETTINGS.embed_model, SETTINGS.embed_dim)
    vec = VectorStore(SETTINGS.qdrant_url, SETTINGS.qdrant_collection,
                      SETTINGS.embed_dim, path=QDB)
    vec.ensure_collection()

    t0 = time.time()
    fdoc, fchk = open(DOCS, "a"), open(CHUNKS, "a")
    processed = 0
    for r in todo:
        if time.time() - t0 > BUDGET_S:
            break
        n_pages, chunks = ingest_pdf(
            r["paper_id"], r["pdf_path"], SETTINGS.chunk_size_chars,
            SETTINGS.chunk_overlap_chars, SETTINGS.min_chunk_chars, max_pages=SETTINGS.max_pages)
        if not chunks:
            done.add(r["paper_id"]); continue
        vecs = embedder.encode_documents([c.text for c in chunks])
        vec.upsert([c.chunk_id for c in chunks], vecs,
                   [{"paper_id": c.paper_id, "topic": r["topics"],
                     "page_start": c.page_start, "page_end": c.page_end} for c in chunks])
        fdoc.write(json.dumps({
            "_id": r["paper_id"], "title": r["title"], "authors": r["authors"],
            "venue": r["venue"], "year": int(r["year"]), "topic": r["topics"],
            "pdf_path": r["pdf_path"], "pdf_url": r["pdf_url"],
            "n_pages": n_pages, "n_chunks": len(chunks)}) + "\n")
        for c in chunks:
            fchk.write(json.dumps(c.to_doc("resumable")) + "\n")
        fdoc.flush(); fchk.flush()                # durable: survive a hard kill
        done.add(r["paper_id"]); processed += 1
        MANIFEST.write_text(json.dumps(sorted(done)))   # checkpoint after each paper
        print(f"  {r['paper_id']} ({r['topics']}): {n_pages}p -> {len(chunks)} chunks")
    fdoc.close(); fchk.close()
    MANIFEST.write_text(json.dumps(sorted(done)))
    remaining = len(papers) - len(done)
    print(f"batch: +{processed} papers in {time.time()-t0:.0f}s · done={len(done)}/{len(papers)}")
    print("ALL_DONE" if remaining == 0 else f"RESUME ({remaining} left)")


if __name__ == "__main__":
    main()
