"""Wires the stores + embedder + searcher into one object and runs ingestion.

`build_pipeline()` is the single entry point used by the seed script, the API,
the evaluator and the tests. It honours env vars (real services if configured,
embedded fallbacks otherwise), so the exact same call works on a laptop with
docker and in the CI sandbox.
"""
from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import SETTINGS, Settings
from embedder import get_embedder
from hybrid_search import HybridSearcher
from ingest import ingest_pdf
from stores.graph_store import get_graph_store
from stores.mongo_store import MongoStore
from stores.vector_store import VectorStore


class Pipeline:
    def __init__(self, settings: Settings = SETTINGS, build_embedder: bool = True):
        self.s = settings
        self.mongo = MongoStore(settings.mongo_uri, settings.mongo_db)
        self.vector = VectorStore(settings.qdrant_url, settings.qdrant_collection,
                                  settings.embed_dim, path=settings.qdrant_path)
        self.embedder = get_embedder(settings.embedder, settings.embed_model, settings.embed_dim) \
            if build_embedder else None
        self.graph = get_graph_store(settings)
        self.searcher: Optional[HybridSearcher] = None

    # ---- ingestion ----
    def ingest_corpus(self, papers: pd.DataFrame, recreate_vectors: bool = True) -> Dict:
        """Ingest every paper that has a local PDF: parse -> chunk -> store -> embed."""
        run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        if recreate_vectors:
            self.vector.recreate()
        n_docs = n_chunks = n_pages_total = skipped = 0

        for _, row in papers.iterrows():
            pdf_path = row.get("pdf_path", "")
            if not pdf_path or not Path(pdf_path).exists():
                skipped += 1
                continue
            n_pages, chunks = ingest_pdf(
                row["paper_id"], pdf_path,
                self.s.chunk_size_chars, self.s.chunk_overlap_chars, self.s.min_chunk_chars,
                max_pages=self.s.max_pages,
            )
            if not chunks:
                skipped += 1
                continue

            self.mongo.upsert_document({
                "_id": row["paper_id"], "title": row["title"], "authors": row["authors"],
                "venue": row.get("venue", "arXiv"), "year": int(row["year"]),
                "topic": row["topic"], "doi": row.get("doi", ""),
                "pdf_path": pdf_path, "pdf_url": row.get("pdf_url", ""),
                "n_pages": n_pages, "n_chunks": len(chunks),
                "sha256": chunks[0].sha256, "run_id": run_id,
            })
            chunk_docs = [c.to_doc(run_id) for c in chunks]
            self.mongo.insert_chunks(chunk_docs)

            texts = [c.text for c in chunks]
            vecs = self.embedder.encode_documents(texts)
            payloads = [{"paper_id": c.paper_id, "topic": row["topic"],
                         "page_start": c.page_start, "page_end": c.page_end} for c in chunks]
            self.vector.upsert([c.chunk_id for c in chunks], vecs, payloads)

            n_docs += 1
            n_chunks += len(chunks)
            n_pages_total += n_pages

        self.mongo.record_run({
            "run_id": run_id, "n_docs": n_docs, "n_chunks": n_chunks, "n_pages": n_pages_total,
            "skipped": skipped, "embedder": self.s.embedder, "embed_model": self.s.embed_model,
            "chunk_size_chars": self.s.chunk_size_chars, "chunk_overlap_chars": self.s.chunk_overlap_chars,
        })
        # build the graph over the docs we actually ingested
        ingested = papers[papers["paper_id"].isin(
            {d["_id"] for d in self.mongo.documents.find({}, {"_id": 1})})]
        graph_stats = self.graph.load(ingested.to_dict("records"))
        self.searcher = None  # force BM25 rebuild on next search
        return {"run_id": run_id, "n_docs": n_docs, "n_chunks": n_chunks,
                "n_pages": n_pages_total, "skipped": skipped, "graph": graph_stats}

    # ---- search ----
    def get_searcher(self) -> HybridSearcher:
        if self.searcher is None:
            self.searcher = HybridSearcher(
                self.mongo, self.vector, self.embedder,
                default_lambda=self.s.hybrid_lambda, pool=self.s.candidate_pool)
            self.searcher.build_bm25()
        return self.searcher

    def search(self, query: str, top_k: int = 5, hybrid_lambda: Optional[float] = None):
        return self.get_searcher().search(query, top_k=top_k, hybrid_lambda=hybrid_lambda)

    def stats(self) -> Dict:
        return {"mongo": self.mongo.stats(), "qdrant_vectors": self.vector.count(),
                "graph": self.graph.stats(),
                "settings": {"embedder": self.s.embedder, "hybrid_lambda": self.s.hybrid_lambda}}


def build_pipeline(build_embedder: bool = True) -> Pipeline:
    return Pipeline(SETTINGS, build_embedder=build_embedder)
