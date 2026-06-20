"""Hybrid BM25 + dense retrieval over chunks, with citations.

Architecture (the realistic GraphRAG-ready shape):

  query
    ├─ dense side : embed(query) -> Qdrant ANN  -> candidate pool (P chunks)
    └─ lexical side: BM25 over all chunk texts   -> candidate pool (P chunks)
  union(pool)  ->  min-max normalise each signal  ->  fuse:  λ·bm25 + (1-λ)·dense
            ->  sort  ->  top-k chunks  ->  attach citation (title, pages, paper_id)

We fuse over the *union of the two candidate pools* rather than the whole
corpus: this is how production hybrid retrieval works (ANN-retrieve, then
re-score), keeps latency flat as the corpus grows, and still lets a chunk that
only one side surfaces win. λ is frozen from the D1 Optuna study (run card) but
is overridable per request — the online learner in D1 adapts exactly this knob.

BM25 is built once from Mongo's chunk texts and cached in-process.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def _tok(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.isalnum() or "-" in t]


def _minmax(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


@dataclass
class Citation:
    chunk_id: str
    paper_id: str
    title: str
    page_start: int
    page_end: int
    score: float
    text: str

    @property
    def page_range(self) -> str:
        return f"p.{self.page_start}" if self.page_start == self.page_end \
            else f"pp.{self.page_start}-{self.page_end}"

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id, "paper_id": self.paper_id, "title": self.title,
            "page_start": self.page_start, "page_end": self.page_end,
            "page_range": self.page_range, "score": round(self.score, 4),
            "snippet": self.text[:280] + ("…" if len(self.text) > 280 else ""),
        }


class HybridSearcher:
    def __init__(self, mongo, vector, embedder, default_lambda: float = 0.5, pool: int = 100):
        self.mongo = mongo
        self.vector = vector
        self.embedder = embedder
        self.default_lambda = default_lambda
        self.pool = pool
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: List[str] = []
        self._meta: Dict[str, Dict] = {}

    def build_bm25(self) -> None:
        """(Re)build the lexical index + the title lookup from Mongo."""
        chunks = self.mongo.all_chunks()
        self._chunk_ids = [c["_id"] for c in chunks]
        self._bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])
        titles = {}
        self._meta = {}
        for c in chunks:
            pid = c["paper_id"]
            if pid not in titles:
                doc = self.mongo.get_document(pid)
                titles[pid] = doc["title"] if doc else pid
            self._meta[c["_id"]] = {
                "paper_id": pid, "title": titles[pid],
                "page_start": c["page_start"], "page_end": c["page_end"],
                "text": c["text"],
            }

    def _bm25_pool(self, query: str) -> Dict[str, float]:
        if self._bm25 is None:
            self.build_bm25()
        scores = self._bm25.get_scores(_tok(query))
        order = np.argsort(-scores)[: self.pool]
        return {self._chunk_ids[i]: float(scores[i]) for i in order if scores[i] > 0}

    def _dense_pool(self, query: str, paper_ids=None) -> Dict[str, float]:
        qv = self.embedder.encode_query(query)
        hits = self.vector.search(qv, top_k=self.pool, paper_ids=paper_ids)
        return {cid: score for cid, score in hits}

    def search(
        self, query: str, top_k: int = 5, hybrid_lambda: Optional[float] = None,
        paper_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Citation], float]:
        """Return (citations, latency_ms). `paper_ids` pins the searchable set."""
        if self._bm25 is None:
            self.build_bm25()
        lam = self.default_lambda if hybrid_lambda is None else hybrid_lambda
        t0 = time.perf_counter()

        bm25 = self._bm25_pool(query)
        dense = self._dense_pool(query, paper_ids=paper_ids)
        bm25_n, dense_n = _minmax(bm25), _minmax(dense)

        cand = set(bm25_n) | set(dense_n)
        fused = {cid: lam * bm25_n.get(cid, 0.0) + (1 - lam) * dense_n.get(cid, 0.0)
                 for cid in cand}
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

        cites: List[Citation] = []
        for cid, sc in ranked:
            m = self._meta.get(cid) or {}
            cites.append(Citation(
                chunk_id=cid, paper_id=m.get("paper_id", "?"), title=m.get("title", "?"),
                page_start=m.get("page_start", 0), page_end=m.get("page_end", 0),
                score=sc, text=m.get("text", ""),
            ))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return cites, latency_ms
