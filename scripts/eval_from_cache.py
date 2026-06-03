"""Evaluate using the resumable on-disk cache (sandbox path).

Rehydrates Mongo (mongomock) from data/cache/*.jsonl, attaches the on-disk
Qdrant at data/qdrant_db, builds the hybrid searcher + graph, then writes the
same artefacts as `eval_search.py`:
    results/search_metrics.{json,md}, results/examples.md, results/graph_examples.md

On a laptop you'd just run `eval_search.py` against the docker services.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "src"))
from config import SETTINGS                        # noqa: E402


# Same binary-relevance definitions as D1's src/gold.py, but robust when fewer
# than k distinct papers are retrieved (D1's nDCG assumed len(retrieved) >= k,
# which breaks here because chunks cluster by paper -> deduped lists can be < k).
def recall_at_k(retrieved, relevant, k):
    rel = set(relevant)
    if not rel:
        return 0.0
    return len([p for p in retrieved[:k] if p in rel]) / len(rel)


def ndcg_at_k(retrieved, relevant, k):
    rel = set(relevant)
    if not rel:
        return 0.0
    gains = np.array([1.0 if p in rel else 0.0 for p in retrieved[:k]])
    if gains.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))      # full length-k discount vector
    dcg = float((gains * discounts[: len(gains)]).sum())
    ideal_n = min(len(rel), k)
    idcg = float(discounts[:ideal_n].sum())
    return dcg / idcg if idcg > 0 else 0.0
from embedder import get_embedder                  # noqa: E402
from hybrid_search import HybridSearcher           # noqa: E402
from stores.mongo_store import MongoStore          # noqa: E402
from stores.vector_store import VectorStore        # noqa: E402
from stores.graph_store import NetworkxGraphStore  # noqa: E402

import os
CACHE = Path(os.environ.get("CACHE_DIR") or (ROOT / "data" / "cache"))
QDB = SETTINGS.qdrant_path or str(ROOT / "data" / "qdrant_db")
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
K, POOL = 5, 20


def rehydrate_mongo() -> MongoStore:
    m = MongoStore("", "papers_ai")               # mongomock
    # dedupe by _id (append-only cache can repeat a paper across resumed batches)
    docs, chunks, seen_d, seen_c = [], [], set(), set()
    for l in open(CACHE / "documents.jsonl"):
        d = json.loads(l)
        if d["_id"] not in seen_d:
            seen_d.add(d["_id"]); docs.append(d)
    for l in open(CACHE / "chunks.jsonl"):
        c = json.loads(l)
        if c["_id"] not in seen_c:
            seen_c.add(c["_id"]); chunks.append(c)
    if docs:
        m.documents.insert_many(docs)
    if chunks:
        m.chunks.insert_many(chunks)
    return m


def dedupe(cites):
    seen, order = set(), []
    for c in cites:
        if c.paper_id not in seen:
            seen.add(c.paper_id); order.append(c.paper_id)
    return order


def eval_run(searcher, gold, lam):
    rs, ns, lat, by = [], [], [], {}
    for _, row in gold.iterrows():
        cites, ms = searcher.search(row["query"], top_k=POOL, hybrid_lambda=lam)
        papers = dedupe(cites)
        r = recall_at_k(papers, row["relevant"], K); n = ndcg_at_k(papers, row["relevant"], K)
        rs.append(r); ns.append(n); lat.append(ms)
        by.setdefault(row["query_type"], []).append((r, n))
    return {"recall@5": float(np.mean(rs)), "ndcg@5": float(np.mean(ns)),
            "p95_latency_ms": float(np.percentile(lat, 95)),
            "mean_latency_ms": float(np.mean(lat)), "n_queries": int(len(gold)),
            "by_type": {t: {"recall@5": float(np.mean([x[0] for x in v])),
                            "ndcg@5": float(np.mean([x[1] for x in v])), "n": len(v)}
                        for t, v in by.items()}}


def main():
    mongo = rehydrate_mongo()
    vec = VectorStore(SETTINGS.qdrant_url, SETTINGS.qdrant_collection, SETTINGS.embed_dim, path=QDB)
    emb = get_embedder(SETTINGS.embedder, SETTINGS.embed_model, SETTINGS.embed_dim)
    searcher = HybridSearcher(mongo, vec, emb, default_lambda=SETTINGS.hybrid_lambda, pool=SETTINGS.candidate_pool)
    searcher.build_bm25()

    papers = pd.read_csv(ROOT / "data" / "papers.csv", dtype={"paper_id": str})
    graph = NetworkxGraphStore()
    in_corpus = {d["_id"] for d in mongo.documents.find({}, {"_id": 1})}
    graph.load(papers[papers["paper_id"].isin(in_corpus)].rename(columns={"topics": "topic"}).to_dict("records"))

    n_chunks = mongo.chunks.count_documents({})
    print(f"{len(in_corpus)} papers · {n_chunks} chunks · qdrant={vec.count()} vectors")

    gold = pd.read_parquet(ROOT.parent / "data" / "gold.parquet")
    gold["relevant"] = gold["relevant"].apply(lambda xs: [p for p in xs if p in in_corpus])
    gold = gold[gold["relevant"].map(len) > 0].reset_index(drop=True)

    metrics = {"hybrid": eval_run(searcher, gold, SETTINGS.hybrid_lambda),
               "bm25_only": eval_run(searcher, gold, 1.0),
               "dense_only": eval_run(searcher, gold, 0.0),
               "config": {"embedder": SETTINGS.embedder, "hybrid_lambda": SETTINGS.hybrid_lambda,
                          "k": K, "search_pool": POOL, "n_papers": len(in_corpus), "n_chunks": n_chunks,
                          "max_pages": SETTINGS.max_pages,
                          "backends": {"mongo": mongo.backend, "qdrant": vec.backend, "graph": graph.backend}}}
    (RES / "search_metrics.json").write_text(json.dumps(metrics, indent=2))

    def fmt(m): return f"| {m['recall@5']:.3f} | {m['ndcg@5']:.3f} | {m['mean_latency_ms']:.1f} | {m['p95_latency_ms']:.1f} |"
    cfg = metrics["config"]
    md = ["# /search evaluation (D2)", "",
          f"Corpus: **{cfg['n_papers']} papers · {cfg['n_chunks']} chunks** (max {cfg['max_pages']} pages/PDF) · "
          f"embedder=`bge-small-en-v1.5` (384-d, fastembed/ONNX) · backends "
          f"mongo=`{mongo.backend}`, qdrant=`{vec.backend}`, graph=`{graph.backend}`",
          f"Gold: {metrics['hybrid']['n_queries']} queries (relevance restricted to in-corpus papers)", "",
          "| Retriever | Recall@5 | nDCG@5 | mean ms | p95 ms |", "|---|---:|---:|---:|---:|",
          f"| BM25 only (λ=1.0) {fmt(metrics['bm25_only'])}",
          f"| Dense only (λ=0.0) {fmt(metrics['dense_only'])}",
          f"| **Hybrid (λ={SETTINGS.hybrid_lambda})** {fmt(metrics['hybrid'])}", "",
          "## Hybrid by query type", "", "| Type | Recall@5 | nDCG@5 | n |", "|---|---:|---:|---:|"]
    for t, v in metrics["hybrid"]["by_type"].items():
        md.append(f"| {t} | {v['recall@5']:.3f} | {v['ndcg@5']:.3f} | {v['n']} |")
    (RES / "search_metrics.md").write_text("\n".join(md))

    ex = ["# Top-k examples with citations (hybrid, bge-small-en)", ""]
    for q in ["retrieval augmented generation with citations",
              "concept drift detection in streaming data",
              "vision transformer self-supervised pretraining",
              "hyperparameter optimization with Optuna"]:
        cites, ms = searcher.search(q, top_k=3)
        ex.append(f"### Query: *{q}*  ({ms:.1f} ms)")
        for i, c in enumerate(cites, 1):
            ex.append(f"{i}. **{c.title}** (`{c.paper_id}`), {c.page_range} — score {c.score:.3f}  \n"
                      f"   > {c.text[:200].strip()}…")
        ex.append("")
    (RES / "examples.md").write_text("\n".join(ex))

    g = graph
    gd = ["# Knowledge graph — example queries (D2)", "",
          f"Backend: `{g.backend}` · {json.dumps(g.stats())}", "",
          "These mirror the parameterised Cypher in `src/cypher_queries.py`.", ""]
    gd.append("**1. papers_by_topic('rag')** →")
    for pid, t in g.papers_by_topic("rag", 5): gd.append(f"   - {pid}: {t}")
    author = papers.iloc[0]["authors"].split(",")[0].strip()
    gd.append(f"\n**2. papers_by_author('{author}')** →")
    for pid, t in g.papers_by_author(author, 5): gd.append(f"   - {pid}: {t}")
    gd.append(f"\n**3. coauthors('{author}')** →")
    for a, c in g.coauthors(author, 5): gd.append(f"   - {a} ({c} shared)")
    yr = int(papers.iloc[0]["year"])
    gd.append(f"\n**4. papers_by_venue_year('arXiv', {yr})** →")
    for pid, t in g.papers_by_venue_year("arXiv", yr, 5): gd.append(f"   - {pid}: {t}")
    seed = papers.iloc[0]["paper_id"]
    gd.append(f"\n**5. related_via_topic('{seed}')  [GraphRAG 2-hop]** →")
    for pid, t in g.related_via_topic(seed, 5): gd.append(f"   - {pid}: {t}")
    (RES / "graph_examples.md").write_text("\n".join(gd))

    print(json.dumps(metrics, indent=2))
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
