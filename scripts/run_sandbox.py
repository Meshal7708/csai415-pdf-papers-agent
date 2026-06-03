"""One-process sandbox run: ingest -> evaluate -> graph demo -> write all results.

On a laptop you'd run `seed_stores.py` (persists to the docker services) and then
`eval_search.py` (reads them back) as two steps. In the no-Docker sandbox the
stores are in-process and ephemeral, so this script does everything in a single
process and writes the artefacts the report consumes:

    results/ingest_summary.json
    results/search_metrics.{json,md}
    results/examples.md
    results/graph_examples.md
    results/RUN_DONE            <- completion marker for polling
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "src"))
from gold import recall_at_k, ndcg_at_k          # noqa: E402
from pipeline import build_pipeline              # noqa: E402

RES = ROOT / "results"; RES.mkdir(exist_ok=True)
K, POOL = 5, 20


def dedupe(cites):
    seen, order = set(), []
    for c in cites:
        if c.paper_id not in seen:
            seen.add(c.paper_id); order.append(c.paper_id)
    return order


def eval_run(pipe, gold, lam):
    rs, ns, lat, by = [], [], [], {}
    for _, row in gold.iterrows():
        cites, ms = pipe.search(row["query"], top_k=POOL, hybrid_lambda=lam)
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
    t0 = time.time()
    papers = pd.read_csv(ROOT / "data" / "papers.csv"); papers["topic"] = papers["topics"]
    pipe = build_pipeline(build_embedder=True)
    summary = pipe.ingest_corpus(papers, recreate_vectors=True)
    summary["ingest_seconds"] = round(time.time() - t0, 1)
    (RES / "ingest_summary.json").write_text(json.dumps({**summary, "stats": pipe.stats()},
                                                        indent=2, default=str))
    pipe.get_searcher()

    in_corpus = {d["_id"] for d in pipe.mongo.documents.find({}, {"_id": 1})}
    gold = pd.read_parquet(ROOT.parent / "data" / "gold.parquet")
    gold["relevant"] = gold["relevant"].apply(lambda xs: [p for p in xs if p in in_corpus])
    gold = gold[gold["relevant"].map(len) > 0].reset_index(drop=True)

    metrics = {"hybrid": eval_run(pipe, gold, pipe.s.hybrid_lambda),
               "bm25_only": eval_run(pipe, gold, 1.0),
               "dense_only": eval_run(pipe, gold, 0.0),
               "config": {"embedder": pipe.s.embedder, "hybrid_lambda": pipe.s.hybrid_lambda,
                          "k": K, "search_pool": POOL, "n_papers": len(in_corpus),
                          "n_chunks": pipe.vector.count(),
                          "backends": {"mongo": pipe.mongo.backend, "qdrant": pipe.vector.backend,
                                       "graph": pipe.graph.backend}}}
    (RES / "search_metrics.json").write_text(json.dumps(metrics, indent=2))

    def fmt(m): return f"| {m['recall@5']:.3f} | {m['ndcg@5']:.3f} | {m['mean_latency_ms']:.1f} | {m['p95_latency_ms']:.1f} |"
    cfg = metrics["config"]
    md = ["# /search evaluation (D2)", "",
          f"Corpus: **{cfg['n_papers']} papers · {cfg['n_chunks']} chunks** · embedder=`{pipe.s.embedder}` "
          f"(bge-small-en-v1.5, 384-d) · backends mongo=`{pipe.mongo.backend}`, "
          f"qdrant=`{pipe.vector.backend}`, graph=`{pipe.graph.backend}`",
          f"Gold: {metrics['hybrid']['n_queries']} queries (relevance restricted to in-corpus papers)", "",
          "| Retriever | Recall@5 | nDCG@5 | mean ms | p95 ms |", "|---|---:|---:|---:|---:|",
          f"| BM25 only (λ=1.0) {fmt(metrics['bm25_only'])}",
          f"| Dense only (λ=0.0) {fmt(metrics['dense_only'])}",
          f"| **Hybrid (λ={pipe.s.hybrid_lambda})** {fmt(metrics['hybrid'])}", "",
          "## Hybrid by query type", "", "| Type | Recall@5 | nDCG@5 | n |", "|---|---:|---:|---:|"]
    for t, v in metrics["hybrid"]["by_type"].items():
        md.append(f"| {t} | {v['recall@5']:.3f} | {v['ndcg@5']:.3f} | {v['n']} |")
    (RES / "search_metrics.md").write_text("\n".join(md))

    # top-k examples with page-range citations
    ex = ["# Top-k examples with citations (hybrid, bge-small-en)", ""]
    for q in ["retrieval augmented generation with citations",
              "concept drift detection in streaming data",
              "vision transformer self-supervised pretraining",
              "hyperparameter optimization with Optuna"]:
        cites, ms = pipe.search(q, top_k=3)
        ex.append(f"### Query: *{q}*  ({ms:.1f} ms)")
        for i, c in enumerate(cites, 1):
            ex.append(f"{i}. **{c.title}** (`{c.paper_id}`), {c.page_range} — score {c.score:.3f}  \n"
                      f"   > {c.text[:200].strip()}…")
        ex.append("")
    (RES / "examples.md").write_text("\n".join(ex))

    # graph + 5 example queries (networkx mirrors the Cypher in cypher_queries.py)
    g = pipe.graph
    gd = ["# Knowledge graph — example queries (D2)", "",
          f"Backend: `{g.backend}` · {json.dumps(g.stats())}", "",
          "These mirror the parameterised Cypher in `src/cypher_queries.py`.", ""]
    topic = "rag"
    gd.append(f"**1. papers_by_topic('{topic}')** →")
    for pid, t in g.papers_by_topic(topic, 5): gd.append(f"   - {pid}: {t}")
    some_author = papers.iloc[0]["authors"].split(",")[0].strip()
    gd.append(f"\n**2. papers_by_author('{some_author}')** →")
    for pid, t in g.papers_by_author(some_author, 5): gd.append(f"   - {pid}: {t}")
    gd.append(f"\n**3. coauthors('{some_author}')** →")
    for a, c in g.coauthors(some_author, 5): gd.append(f"   - {a} ({c} shared)")
    gd.append(f"\n**4. papers_by_venue_year('arXiv', {int(papers.iloc[0]['year'])})** →")
    for pid, t in g.papers_by_venue_year("arXiv", int(papers.iloc[0]["year"]), 5): gd.append(f"   - {pid}: {t}")
    seed = papers.iloc[0]["paper_id"]
    gd.append(f"\n**5. related_via_topic('{seed}')  [GraphRAG 2-hop]** →")
    for pid, t in g.related_via_topic(seed, 5): gd.append(f"   - {pid}: {t}")
    (RES / "graph_examples.md").write_text("\n".join(gd))

    (RES / "RUN_DONE").write_text(f"ok in {round(time.time()-t0,1)}s")
    print("RUN COMPLETE", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
