"""Evaluate the hybrid /search stack: Recall@k, nDCG@k, latency p95 + examples.

Metric functions are imported from D1 (`src/gold.py`) so D2's numbers are
directly comparable to D1's. Differences from D1:

* Retrieval is over **chunks**; we dedupe each result list to a ranked **paper**
  list (first occurrence wins) before scoring, because the gold set is
  paper-level.
* The D2 corpus is a curated subset, so each query's gold `relevant` set is
  intersected with the papers actually ingested; queries with no in-corpus
  relevant paper are dropped.

We report overall + per-query-type metrics, a vector-only / BM25-only / hybrid
comparison (mini-ablation, foreshadowing D3), and a few top-k examples with
page-range citations. Outputs: results/search_metrics.{json,md}, examples.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "src"))   # D1 src for gold metrics
from gold import recall_at_k, ndcg_at_k         # noqa: E402  (reuse D1 metrics)
from pipeline import build_pipeline             # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
GOLD = ROOT.parent / "data" / "gold.parquet"
K = 5
POOL = 20  # how many chunks we pull before deduping to papers


def dedupe_papers(cites):
    seen, order = set(), []
    for c in cites:
        if c.paper_id not in seen:
            seen.add(c.paper_id)
            order.append(c.paper_id)
    return order


def run(pipe, gold, lam):
    rs, ns, lat = [], [], []
    by_type = {}
    for _, row in gold.iterrows():
        cites, ms = pipe.search(row["query"], top_k=POOL, hybrid_lambda=lam)
        papers = dedupe_papers(cites)
        r = recall_at_k(papers, row["relevant"], K)
        n = ndcg_at_k(papers, row["relevant"], K)
        rs.append(r); ns.append(n); lat.append(ms)
        by_type.setdefault(row["query_type"], []).append((r, n))
    out = {f"recall@{K}": float(np.mean(rs)), f"ndcg@{K}": float(np.mean(ns)),
           "p95_latency_ms": float(np.percentile(lat, 95)),
           "mean_latency_ms": float(np.mean(lat)), "n_queries": len(gold)}
    out["by_type"] = {t: {f"recall@{K}": float(np.mean([x[0] for x in v])),
                          f"ndcg@{K}": float(np.mean([x[1] for x in v])), "n": len(v)}
                      for t, v in by_type.items()}
    return out


def main():
    pipe = build_pipeline(build_embedder=True)
    pipe.get_searcher()
    in_corpus = {d["_id"] for d in pipe.mongo.documents.find({}, {"_id": 1})}
    print(f"{len(in_corpus)} papers in corpus")

    gold = pd.read_parquet(GOLD)
    gold["relevant"] = gold["relevant"].apply(lambda xs: [p for p in xs if p in in_corpus])
    gold = gold[gold["relevant"].map(len) > 0].reset_index(drop=True)
    print(f"{len(gold)} gold queries with in-corpus relevance")

    metrics = {
        "hybrid": run(pipe, gold, pipe.s.hybrid_lambda),
        "bm25_only": run(pipe, gold, 1.0),
        "dense_only": run(pipe, gold, 0.0),
        "config": {"embedder": pipe.s.embedder, "hybrid_lambda": pipe.s.hybrid_lambda,
                   "k": K, "search_pool": POOL,
                   "backends": {"mongo": pipe.mongo.backend, "qdrant": pipe.vector.backend,
                                "graph": pipe.graph.backend}},
    }
    (RESULTS / "search_metrics.json").write_text(json.dumps(metrics, indent=2))

    # markdown table
    def fmt(m):
        return (f"| {m['recall@5']:.3f} | {m['ndcg@5']:.3f} | "
                f"{m['mean_latency_ms']:.1f} | {m['p95_latency_ms']:.1f} |")
    md = ["# /search evaluation (D2)", "",
          f"Corpus: {len(in_corpus)} papers · {pipe.vector.count()} chunks · "
          f"embedder=`{pipe.s.embedder}` · backends "
          f"mongo=`{pipe.mongo.backend}`, qdrant=`{pipe.vector.backend}`, graph=`{pipe.graph.backend}`",
          f"Gold: {len(gold)} queries (relevance restricted to in-corpus papers)", "",
          "| Retriever | Recall@5 | nDCG@5 | mean ms | p95 ms |",
          "|---|---:|---:|---:|---:|",
          f"| BM25 only (λ=1.0) {fmt(metrics['bm25_only'])}",
          f"| Dense only (λ=0.0) {fmt(metrics['dense_only'])}",
          f"| **Hybrid (λ={pipe.s.hybrid_lambda})** {fmt(metrics['hybrid'])}", "",
          "## Hybrid by query type", "",
          "| Type | Recall@5 | nDCG@5 | n |", "|---|---:|---:|---:|"]
    for t, v in metrics["hybrid"]["by_type"].items():
        md.append(f"| {t} | {v['recall@5']:.3f} | {v['ndcg@5']:.3f} | {v['n']} |")
    (RESULTS / "search_metrics.md").write_text("\n".join(md))

    # top-k examples with citations
    examples = ["# Top-k examples with citations (hybrid)", ""]
    demo = ["retrieval augmented generation with citations",
            "concept drift detection in streaming data",
            "vision transformer self-supervised pretraining"]
    for q in demo:
        cites, ms = pipe.search(q, top_k=3)
        examples.append(f"### Query: *{q}*  ({ms:.1f} ms)")
        for i, c in enumerate(cites, 1):
            examples.append(f"{i}. **{c.title}** ({c.paper_id}), {c.page_range} "
                            f"— score {c.score:.3f}\n   > {c.text[:200].strip()}…")
        examples.append("")
    (RESULTS / "examples.md").write_text("\n".join(examples))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
