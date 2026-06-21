"""D3 FastAPI — the GraphRAG agent endpoint surface.

    GET  /healthz                 store backends + readiness
    GET  /stats                   document/chunk/vector/graph counts
    GET  /ask?q=&mode=&k=         GraphRAG answer with citations + page ranges
                                  + the 4-stage trace (seeds, subgraph, expansion)
    GET  /search?q=&k=&lam=       raw hybrid retrieval (carried from D2)
    GET  /graph/subgraph?seeds=   weighted multi-signal subgraph (Cypher) for seeds
    POST /evaluate                run the gold-set evaluation for a mode

The pipeline (stores + embedder + graph + executor) is built once on startup and
honours the same env vars as everything else (real services if configured, the
offline cache otherwise). Run:  `uvicorn api.main:app --reload`
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Query

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d3_pipeline import build_d3_pipeline          # noqa: E402

app = FastAPI(title="D3 GraphRAG Agent", version="1.0")
_PIPE = {"p": None}


def pipe():
    if _PIPE["p"] is None:
        _PIPE["p"] = build_d3_pipeline()
    return _PIPE["p"]


@app.get("/healthz")
def healthz():
    p = pipe()
    return {"ok": True, "backends": {
        "mongo": p.mongo.backend, "qdrant": p.vector.backend,
        "graph": p.graph.backend}}


@app.get("/stats")
def stats():
    return pipe().stats()


@app.get("/ask")
def ask(q: str = Query(..., min_length=3), mode: str = "graph_hybrid",
        k: int = 5, rerank: bool = True):
    r = pipe().executor.answer(q, mode=mode, top_k=k, rerank=rerank)
    return {"question": q, "mode": r.mode, "answer": r.answer,
            "citations": r.citations, "latency_ms": round(r.latency_ms, 2),
            "trace": {kk: vv for kk, vv in r.trace.items() if kk != "graph_prior"}}


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 5, lam: float = None):
    cites, ms = pipe().searcher.search(q, top_k=k, hybrid_lambda=lam)
    return {"query": q, "latency_ms": round(ms, 2),
            "results": [c.to_dict() for c in cites]}


@app.get("/graph/subgraph")
def subgraph(seeds: str = Query(..., description="comma-separated paper_ids"),
             limit: int = 8):
    seed_ids = [s.strip() for s in seeds.split(",") if s.strip()]
    return {"seeds": seed_ids, "subgraph": pipe().graph.weighted_subgraph(seed_ids, limit=limit)}


@app.post("/evaluate")
def evaluate_endpoint(mode: str = "graph_hybrid", k: int = 5):
    from evaluate import evaluate
    ev = evaluate(pipe(), mode=mode, top_k=k)
    return {"mode": mode, "overall": ev["overall"], "by_type": ev["by_type"]}
