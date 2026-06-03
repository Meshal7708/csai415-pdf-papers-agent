"""FastAPI surface for the retrieval stack.

Endpoints (D2 scope; /ask + /feedback are stubbed for D3/D1-integration):
    GET  /healthz                 -> liveness + which store backends are live
    GET  /stats                   -> document/chunk/vector/graph counts
    GET  /search?q=&k=&lambda=     -> hybrid retrieval with citations + latency
    POST /ingest  {paper_id, pdf_path}  -> ingest a single PDF on the fly
    GET  /graph/related?paper_id=  -> GraphRAG 2-hop expansion (shared topic)

Run:  uvicorn api.main:app --reload   (from the D2/ dir, with PYTHONPATH=src)
The pipeline is built lazily on first use so the process starts instantly.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import build_pipeline  # noqa: E402

app = FastAPI(title="PDF-Papers Retrieval Stack (D2)", version="0.2.0")
_PIPE = None


def pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = build_pipeline(build_embedder=True)
        _PIPE.get_searcher()  # warm BM25 + titles
    return _PIPE


class IngestReq(BaseModel):
    paper_id: str
    pdf_path: str
    title: str = ""
    authors: str = ""
    venue: str = "arXiv"
    year: int = 0
    topic: str = "unknown"


@app.get("/healthz")
def healthz():
    p = pipe()
    return {"status": "ok", "backends": {
        "mongo": p.mongo.backend, "qdrant": p.vector.backend, "graph": p.graph.backend}}


@app.get("/stats")
def stats():
    return pipe().stats()


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 5, lam: float | None = None):
    cites, latency = pipe().search(q, top_k=k, hybrid_lambda=lam)
    return {
        "query": q, "k": k, "hybrid_lambda": lam if lam is not None else pipe().s.hybrid_lambda,
        "latency_ms": round(latency, 2),
        "results": [{**c.to_dict(),
                     "citation": f'{c.title} ({c.paper_id}), {c.page_range}'} for c in cites],
    }


@app.post("/ingest")
def ingest(req: IngestReq):
    import pandas as pd
    if not Path(req.pdf_path).exists():
        raise HTTPException(404, f"pdf not found: {req.pdf_path}")
    df = pd.DataFrame([req.model_dump()])
    df["pdf_url"] = ""
    out = pipe().ingest_corpus(df, recreate_vectors=False)
    return {"ingested": req.paper_id, **out}


@app.get("/graph/related")
def graph_related(paper_id: str, limit: int = 10):
    rows = pipe().graph.related_via_topic(paper_id, limit=limit)
    return {"paper_id": paper_id, "related": [{"id": i, "title": t} for i, t in rows]}
