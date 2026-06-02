"""Baseline retrieval metrics for D1.

Three baselines, all using a fixed retriever config (k=10, cosine,
embedder=bge, l2_normalize=True). The only thing that varies is `hybrid_lambda`:
    BM25-only          → λ = 1.0
    Dense-only         → λ = 0.0
    Naive 0.5/0.5      → λ = 0.5

Output: results/baselines.json
"""

from __future__ import annotations               # postponed-evaluation type hints
import json                                      # dump baseline metrics as JSON
from pathlib import Path                         # cross-platform paths

import pandas as pd                              # load corpus / gold parquet files

from retriever import HybridConfig, HybridRetriever  # the retriever under test
from gold import evaluate                             # mean Recall@k + NDCG@k + p95 latency

ROOT = Path(__file__).resolve().parents[1]       # D1/
DATA = ROOT / "data"                             # input dir (corpus + gold parquet)
RESULTS = ROOT / "results"                       # output dir (baselines.json lands here)
RESULTS.mkdir(parents=True, exist_ok=True)       # create results/ if missing


def main():
    # Load the two persisted artifacts produced by build_corpus.py and gold.py
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    gold = pd.read_parquet(DATA / "gold.parquet")

    # Fixed retriever configuration — only `hybrid_lambda` varies between baselines.
    # Using a single fitted retriever for all three lambdas keeps the comparison apples-to-apples.
    cfg = HybridConfig(k=10, metric="cosine", svd_dim=128,
                       l2_normalize=True, hybrid_lambda=0.5)


    out = {"config_fixed": cfg.__dict__, "baselines": {}}    # output dict mirrors the JSON schema
    for name, lam in [("bm25_only", 1.0),                    # λ=1.0 → BM25-only (lexical)
                      ("dense_only", 0.0),                   # λ=0.0 → dense-only (TF-IDF+SVD)
                      ("naive_hybrid_0.5", 0.5)]:            # λ=0.5 → equal mix
        # Override λ at search time so we don't need to refit; gold.evaluate handles batching.
        m = evaluate(retriever, gold, k=5, top_k_search=20, hybrid_lambda=lam)
        out["baselines"][name] = m
        # Single-line summary for the CLI log
        print(f"{name:20s}  recall@5={m['recall@5']:.3f}  "
              f"ndcg@5={m['ndcg@5']:.3f}  p95_ms={m['p95_latency_ms']:.2f}")

    # Persist baselines so the AutoML and report scripts can compare against them.
    (RESULTS / "baselines.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'baselines.json'}")


if __name__ == "__main__":
    main()
