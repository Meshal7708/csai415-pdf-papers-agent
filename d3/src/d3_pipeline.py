"""D3 pipeline: rehydrate every store from the offline cache and wire them into
the GraphRAG executor.

Single entry point shared by the evaluator, ablation, API, notebook and tests:

    p = build_d3_pipeline()
    ans = p.ask("how is concept drift detected in streaming data?")

It rehydrates Mongo (mongomock) from D2's jsonl cache, rebuilds an in-memory
Qdrant from D3's embedding npz, constructs the bge query embedder + D2 hybrid
searcher, derives paper-level embeddings for SIMILAR_TO edges, loads the
enriched graph (+ real CITES), and hands all of it to `GraphRAGExecutor`.

If real services are configured (MONGO_URI / QDRANT_URL / NEO4J_URI) the same
code targets them instead — the fallbacks are transparent.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import SETTINGS
from embedder import get_embedder
from hybrid_search import HybridSearcher
from stores.mongo_store import MongoStore
from vector_index import build_vector_store, load_embeddings
from graph_store_d3 import get_graph_store, topk_similar
from graphrag import GraphRAGExecutor

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "D2" / "data" / "cache"
CITES = ROOT / "data" / "cites.json"


def _rehydrate_mongo() -> MongoStore:
    m = MongoStore(SETTINGS.mongo_uri, SETTINGS.mongo_db)
    if m.documents.count_documents({}) > 0:
        return m                                   # real Mongo already seeded
    # Coerce ids to str everywhere so the stores, graph and gold set align
    # (the cache stores arXiv ids as floats; the graph + gold use strings).
    docs, chunks, sd, sc = [], [], set(), set()
    for l in open(CACHE / "documents.jsonl"):
        d = json.loads(l)
        d["_id"] = str(d["_id"])
        if d["_id"] not in sd:
            sd.add(d["_id"]); docs.append(d)
    for l in open(CACHE / "chunks.jsonl"):
        c = json.loads(l)
        c["_id"] = str(c["_id"]); c["paper_id"] = str(c["paper_id"])
        if c["_id"] not in sc:
            sc.add(c["_id"]); chunks.append(c)
    if docs:
        m.documents.insert_many(docs)
    if chunks:
        m.chunks.insert_many(chunks)
    return m


def _paper_vectors(mongo) -> (List[str], np.ndarray):
    """Mean-pool each paper's chunk vectors -> one vector per paper (for SIMILAR_TO)."""
    ids, vecs = load_embeddings()
    by_paper: Dict[str, List[np.ndarray]] = defaultdict(list)
    for cid, v in zip(ids, vecs):
        by_paper[cid.split("::")[0]].append(v)
    paper_ids = sorted(by_paper)
    mat = np.vstack([np.mean(by_paper[p], axis=0) for p in paper_ids]).astype(np.float32)
    return paper_ids, mat


class D3Pipeline:
    def __init__(self):
        self.mongo = _rehydrate_mongo()
        self.vector = build_vector_store(self.mongo)
        self.embedder = get_embedder(SETTINGS.embedder, SETTINGS.embed_model, SETTINGS.embed_dim)
        self.searcher = HybridSearcher(self.mongo, self.vector, self.embedder,
                                       default_lambda=SETTINGS.hybrid_lambda,
                                       pool=SETTINGS.candidate_pool)
        self.searcher.build_bm25()

        # documents carry authors/topic/year/venue -> build the graph from Mongo
        docs = [{"paper_id": str(d["_id"]), "title": d["title"], "authors": d["authors"],
                 "venue": d.get("venue", "arXiv"), "year": d.get("year", 0),
                 "topic": d["topic"]} for d in self.mongo.documents.find({})]
        pids, pmat = _paper_vectors(self.mongo)
        similar = topk_similar(pids, pmat, k=8, min_score=0.30)
        cites = [tuple(e) for e in json.loads(CITES.read_text())["edges"]] if CITES.exists() else []
        self.graph = get_graph_store(SETTINGS)
        self.graph_stats = self.graph.load(docs, similar=similar, cites=cites)

        self.executor = GraphRAGExecutor(self.searcher, self.graph, self.mongo)

    def ask(self, question: str, **kw):
        return self.executor.answer(question, **kw)

    def stats(self) -> Dict:
        return {"mongo": self.mongo.stats(), "qdrant_vectors": self.vector.count(),
                "graph": self.graph_stats}


def build_d3_pipeline() -> D3Pipeline:
    return D3Pipeline()
