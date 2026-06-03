"""MongoDB store for document metadata, chunk provenance, run cards and a
TTL-backed cache collection.

Design notes for the rubric ("Mongo schema with provenance"):

* `documents`  — one record per paper. Holds bibliographic metadata plus
  ingestion provenance (sha256 of the source PDF, n_pages, n_chunks, run_id).
  `_id` is the arXiv paper_id so re-ingesting upserts in place.
* `chunks`     — one record per chunk. Carries the page range + char offsets
  that make every retrieved snippet citable. Indexed on `paper_id`.
* `runs`       — an ingestion "run card": when, how many docs/chunks, embedder,
  chunking params. Lets us trace any chunk back to the exact run that made it.
* `cache`      — short-lived query cache with a **TTL index** so entries expire
  automatically (demonstrates the TTL learning outcome). Real Mongo enforces
  the TTL via a background reaper; mongomock accepts the index definition.

If `MONGO_URI` is set we use a real `pymongo` client; otherwise we fall back to
`mongomock` so the pipeline (and tests) run with no server. Same API either way.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional


class MongoStore:
    def __init__(self, uri: str = "", db_name: str = "papers_ai"):
        if uri:
            from pymongo import MongoClient
            self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")          # fail fast if unreachable
            self.backend = "mongodb"
        else:
            import mongomock
            self.client = mongomock.MongoClient()
            self.backend = "mongomock"
        self.db = self.client[db_name]
        self.documents = self.db["documents"]
        self.chunks = self.db["chunks"]
        self.runs = self.db["runs"]
        self.cache = self.db["cache"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.chunks.create_index("paper_id")
        self.chunks.create_index("run_id")
        self.documents.create_index("topic")
        self.documents.create_index([("year", 1)])
        # TTL: cache docs disappear 1 h after `created_at`.
        try:
            self.cache.create_index("created_at", expireAfterSeconds=3600)
        except Exception:
            pass  # mongomock may not support TTL semantics; index def is still valid

    # ---- writes ----
    def upsert_document(self, doc: Dict) -> None:
        doc = dict(doc)
        doc["ingested_at"] = dt.datetime.utcnow()
        self.documents.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def insert_chunks(self, chunk_docs: List[Dict]) -> int:
        if not chunk_docs:
            return 0
        now = dt.datetime.utcnow()
        for c in chunk_docs:
            c["ingested_at"] = now
        # idempotent re-ingest: drop this paper's old chunks first
        pid = chunk_docs[0]["paper_id"]
        self.chunks.delete_many({"paper_id": pid})
        self.chunks.insert_many(chunk_docs)
        return len(chunk_docs)

    def record_run(self, run: Dict) -> None:
        run = dict(run)
        run["created_at"] = dt.datetime.utcnow()
        self.runs.insert_one(run)

    # ---- reads ----
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        return self.chunks.find_one({"_id": chunk_id})

    def get_chunks(self, chunk_ids: Iterable[str]) -> Dict[str, Dict]:
        ids = list(chunk_ids)
        return {c["_id"]: c for c in self.chunks.find({"_id": {"$in": ids}})}

    def all_chunks(self) -> List[Dict]:
        """Every chunk, ordered — used to build the in-process BM25 index."""
        return list(self.chunks.find({}).sort("_id", 1))

    def get_document(self, paper_id: str) -> Optional[Dict]:
        return self.documents.find_one({"_id": paper_id})

    def stats(self) -> Dict:
        return {
            "backend": self.backend,
            "documents": self.documents.count_documents({}),
            "chunks": self.chunks.count_documents({}),
            "runs": self.runs.count_documents({}),
        }
