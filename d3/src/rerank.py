"""Reranking stage (the executor's optional step 3 refinement).

Default `SemanticMMRReranker` reorders the fused candidate chunks with Maximal
Marginal Relevance: it rewards relevance to the query while penalising redundancy
against chunks already chosen, so the top-k that reaches the answerer covers more
of the question instead of repeating one passage. It scores against the same bge
vectors used everywhere (chunk embedding cache), so it adds no new model.

`CrossEncoderReranker` is an optional higher-fidelity path: if
`sentence-transformers` is installed it loads a ms-marco MiniLM cross-encoder.
The executor uses MMR by default (offline, deterministic) and can be handed the
cross-encoder when the dependency is present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from vector_index import load_embeddings


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class SemanticMMRReranker:
    kind = "semantic-mmr"

    def __init__(self, embedder, lam: float = 0.7):
        self.embedder = embedder
        self.lam = lam                              # relevance vs diversity tradeoff
        ids, vecs = load_embeddings()
        self.vec: Dict[str, np.ndarray] = {
            cid: _norm(v.astype(np.float32)) for cid, v in zip(ids, vecs)}

    def rerank(self, query: str, cites: List, top_k: int = 5,
               prior: dict = None, beta: float = 0.15) -> List:
        """Rerank by semantic relevance + MMR diversity, optionally adding a graph
        structural `prior` (paper_id -> [0,1] weight from the GraphRAG subgraph).
        The prior is the graph-augmented-fusion signal: evidence from papers the
        graph judged related to the seeds gets a modest boost."""
        if not cites:
            return cites
        prior = prior or {}
        q = _norm(self.embedder.encode_query(query))
        cand = [c for c in cites if c.chunk_id in self.vec]
        cand += [c for c in cites if c.chunk_id not in self.vec]   # keep unknowns at tail
        rel = {c.chunk_id: (float(q @ self.vec[c.chunk_id]) if c.chunk_id in self.vec else 0.0)
               + beta * float(prior.get(str(c.paper_id), 0.0))
               for c in cand}
        selected: List = []
        pool = list(cand)
        while pool and len(selected) < top_k:
            best, best_score = None, -1e9
            for c in pool:
                div = 0.0
                if selected and c.chunk_id in self.vec:
                    div = max(float(self.vec[c.chunk_id] @ self.vec[s.chunk_id])
                              for s in selected if s.chunk_id in self.vec)
                score = self.lam * rel[c.chunk_id] - (1 - self.lam) * div
                if score > best_score:
                    best, best_score = c, score
            selected.append(best)
            pool.remove(best)
        for c in selected:                          # expose the rerank relevance
            c.score = round(rel.get(c.chunk_id, c.score), 4)
        return selected


class CrossEncoderReranker:
    kind = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, cites: List, top_k: int = 5) -> List:
        if not cites:
            return cites
        scores = self.model.predict([(query, c.text) for c in cites])
        order = np.argsort(-np.asarray(scores))[:top_k]
        out = []
        for i in order:
            c = cites[i]
            c.score = float(scores[i])
            out.append(c)
        return out
