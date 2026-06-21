"""GraphRAG executor — the D3 core.

Four explicit stages (brief: "choose subgraph by Cypher; expand to supporting
chunks; hybrid blend and optional rerank; answer with citations and page
ranges"):

  1. SUBGRAPH   first-pass hybrid retrieval picks seed papers; the graph's
                weighted multi-signal query (cypher_queries_d3.WEIGHTED_SUBGRAPH)
                selects a ranked subgraph of related papers around the seeds.
  2. EXPAND     pull supporting chunks from the subgraph papers (dense search
                pinned to that paper set) and union them with the global hybrid
                candidates — graph expansion can only *add* recall.
  3. BLEND+RERANK   fuse the pools and rerank (semantic MMR) to a precise top-k.
  4. ANSWER     compose a grounded answer with numbered [n] citations carrying
                paper_id + page range.

`mode` selects the ablation arm:
    vector_only   — dense ANN only (λ=0), no graph, no rerank
    hybrid        — BM25+dense fusion, no graph expansion
    graph_hybrid  — the full pipeline above (default)

`answer()` returns the answer, the citations, a trace (seeds, subgraph, pinned
set) and the end-to-end latency, so the notebook and evaluator can inspect every
stage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from answerer import get_answerer
from rerank import SemanticMMRReranker


def _dedupe_papers(cites) -> List[str]:
    seen, out = set(), []
    for c in cites:
        if c.paper_id not in seen:
            seen.add(c.paper_id); out.append(c.paper_id)
    return out


def _merge(primary: List, extra: List) -> List:
    by_id = {c.chunk_id: c for c in primary}
    for c in extra:
        by_id.setdefault(c.chunk_id, c)
    return list(by_id.values())


@dataclass
class GraphRAGResult:
    answer: str
    citations: List[dict]
    mode: str
    latency_ms: float
    trace: Dict = field(default_factory=dict)
    contexts: List[str] = field(default_factory=list)
    retrieved_papers: List[str] = field(default_factory=list)   # top-k context papers

    def to_dict(self) -> Dict:
        return {"answer": self.answer, "citations": self.citations, "mode": self.mode,
                "latency_ms": round(self.latency_ms, 2), "trace": self.trace}


class GraphRAGExecutor:
    def __init__(self, searcher, graph, mongo, reranker=None, answerer=None,
                 n_seeds: int = 6, expand_k: int = 6, pool: int = 30):
        self.searcher = searcher
        self.graph = graph
        self.mongo = mongo
        self.reranker = reranker or SemanticMMRReranker(searcher.embedder)
        self.answerer = answerer or get_answerer()
        self.n_seeds = n_seeds
        self.expand_k = expand_k
        self.pool = pool

    # -------------------------------------------------- retrieval per mode ---
    def _vector(self, q):
        cites, _ = self.searcher.search(q, top_k=self.pool, hybrid_lambda=0.0)
        return cites, {}

    def _hybrid(self, q):
        cites, _ = self.searcher.search(q, top_k=self.pool, hybrid_lambda=None)
        return cites, {}

    def _graph_hybrid(self, q):
        # 1. SUBGRAPH — seeds from a first-pass hybrid retrieval
        first, _ = self.searcher.search(q, top_k=self.pool, hybrid_lambda=None)
        seeds = _dedupe_papers(first)[: self.n_seeds]
        subgraph = self.graph.weighted_subgraph(seeds, limit=self.expand_k)
        expanded = [r["id"] for r in subgraph]

        # 2. EXPAND — supporting chunks from the subgraph papers (pinned dense)
        extra = []
        if expanded:
            extra, _ = self.searcher.search(q, top_k=self.pool, hybrid_lambda=0.0,
                                            paper_ids=expanded)
        pool = _merge(first, extra)
        # graph structural prior for the rerank: seeds + subgraph papers, weighted
        prior = {str(s): 1.0 for s in seeds}
        if subgraph:
            top = max(r["score"] for r in subgraph) or 1.0
            for r in subgraph:
                prior[str(r["id"])] = max(prior.get(str(r["id"]), 0.0), r["score"] / top)
        trace = {"seeds": seeds,
                 "subgraph": subgraph,
                 "expanded_papers": expanded,
                 "pinned_set_size": len(set(seeds) | set(expanded)),
                 "candidate_chunks": len(pool),
                 "graph_prior": prior}
        return pool, trace

    # ------------------------------------------------------------- answer ----
    def answer(self, question: str, mode: str = "graph_hybrid", top_k: int = 5,
               rerank: bool = True, safety_filter=None) -> GraphRAGResult:
        t0 = time.perf_counter()
        if mode == "vector_only":
            pool, trace = self._vector(question); rerank = False
        elif mode == "hybrid":
            pool, trace = self._hybrid(question)
        elif mode == "graph_hybrid":
            pool, trace = self._graph_hybrid(question)
        else:
            raise ValueError(f"unknown mode: {mode}")

        # 2b. SAFETY (optional) — drop untrusted/poisoned chunks before blending
        if safety_filter is not None:
            pool, sret = safety_filter(question, pool)
            trace["safety"] = sret

        # 3. BLEND + RERANK (graph prior applied only in graph_hybrid mode)
        if rerank:
            prior = trace.get("graph_prior") if mode == "graph_hybrid" else None
            top = self.reranker.rerank(question, pool, top_k=top_k, prior=prior)
            trace["rerank"] = self.reranker.kind
        else:
            top = pool[:top_k]

        # 4. ANSWER with citations + page ranges
        ans = self.answerer.answer(question, top)
        latency = (time.perf_counter() - t0) * 1000.0
        return GraphRAGResult(answer=ans.text, citations=ans.citations, mode=mode,
                              latency_ms=latency, trace=trace, contexts=ans.contexts,
                              retrieved_papers=_dedupe_papers(top))
