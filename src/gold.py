"""Gold Q/A evaluation set + ranking metrics for D1.

i generate three query types per topic:
  • broad queries     — relevant_set = all papers in that topic
  • targeted queries  — relevant_set = topic-papers whose text contains the term
  • title queries     — paraphrased title; relevant_set = {that single paper}
    (these make Recall@5 a meaningful metric — broad-query gold sets have ~25
     papers per topic and structurally cap Recall@5 at 5/25 = 0.20)

Output: data/gold.parquet with columns
  query_id, query, topic, query_type, relevant (list[paper_id])
"""

from __future__ import annotations               # postponed-evaluation type hints
from pathlib import Path                         # path manipulation
from typing import Iterable, List, Sequence      # type hints used in metric signatures

import numpy as np                               # NDCG uses log2 / array math
import pandas as pd                              # gold-set DataFrame, parquet I/O

ROOT = Path(__file__).resolve().parents[1]       # D1/
DATA = ROOT / "data"                             # D1/data/

# Broad queries — designed to land squarely inside one topic.
# These mimic real questions a user might type at the search bar.
BROAD_QUERIES = {
    "transformers": [
        "long-context attention mechanism",
        "memory-efficient transformer architecture",
        "sparse attention for language models",
        "rotary positional embedding for decoder models",
    ],
    "rag": [
        "retrieval augmented generation with citations",
        "graphrag subgraph expansion",
        "hybrid bm25 dense retrieval",
        "cross-encoder reranking for question answering",
    ],
    "online_learning": [
        "concept drift detection in streaming data",
        "prequential evaluation for online classifiers",
        "adaptive random forests for streams",
        "ADWIN-based drift handling",
    ],
    "vision": [
        "vision transformer self-supervised pretraining",
        "diffusion model for image generation",
        "object detection with DETR",
        "open-vocabulary segmentation",
    ],
    "rl_agents": [
        "tool-using llm agent with ReAct",
        "reinforcement learning for browser automation",
        "PPO actor-critic policy optimization",
        "multi-agent coordination with sparse rewards",
    ],
    "automl": [
        "hyperparameter optimization with Optuna",
        "FLAML automl pipeline search",
        "BOHB multi-fidelity tuning",
        "kNN hyperparameter search for retrieval",
    ],
}

# Method/dataset queries — narrower; gold = topic-papers that mention the term.
# These are picked from the methods/datasets that the corpus generator actually
# emits, so each term is guaranteed to appear in at least one paper.
# TARGETED_TERMS = {
#     "transformers": ["FlashAttention", "grouped-query attention", "PG-19"],
#     "rag": ["HNSW index", "Cypher subgraph expansion", "HotpotQA"],
#     "online_learning": ["ADWIN", "Hoeffding", "NYC Taxi"],
#     "vision": ["DINO", "MAE", "COCO"],
#     "rl_agents": ["WebShop", "self-play", "GSM8K"],
#     "automl": ["BOHB", "median pruner", "FLAML"],
# }
TARGETED_TERMS = {
"transformers":    ["attention", "transformer", "token"],
    "rag":             ["retrieval", "reranking", "passage"],
    "online_learning": ["drift", "stream", "incremental"],
    "vision":          ["segmentation", "detection", "image"],
    "rl_agents":       ["policy", "reward", "agent"],
    "automl":          ["hyperparameter", "search", "optimization"],
}


def _papers_in_topic(corpus: pd.DataFrame, topic: str) -> List[str]:
    """Return all paper_ids in a topic — used as the relevant set for broad queries."""
    return corpus.loc[corpus["topic"] == topic, "paper_id"].tolist()


def _papers_with_term(corpus: pd.DataFrame, topic: str, term: str) -> List[str]:
    """Topic-papers whose text contains the given term (case-insensitive substring)."""
    mask = (corpus["topic"] == topic) & corpus["text"].str.contains(
        term, case=False, regex=False                  # plain substring match (no regex surprises)
    )
    return corpus.loc[mask, "paper_id"].tolist()


def _title_to_query(title: str) -> str:
    """Lightly noise a title so the retriever can't trivially memorize it."""
    words = title.lower().split()                # lowercase + split on whitespace
    # drop short connectors so the query doesn't look identical to the title
    drop = {"for", "with", "via", "the", "a", "of", "on", "in", "to", "and", "under"}
    return " ".join(w for w in words if w not in drop)


def build_gold(corpus: pd.DataFrame, n_title_per_topic: int = 6,
               seed: int = 7) -> pd.DataFrame:
    """Build a 78-query gold set: 24 broad + 18 targeted + 36 title."""
    import random
    rng = random.Random(seed)                    # local seeded RNG (separate from corpus seed for diversity)

    rows = []                                    # accumulator for per-query records
    qid = 0                                      # running query id (Q000, Q001, ...)

    # ---- broad queries: gold = full topic ----
    for topic, queries in BROAD_QUERIES.items():
        relevant = _papers_in_topic(corpus, topic)   # 25 papers per topic
        for q in queries:
            rows.append({
                "query_id": f"Q{qid:03d}",
                "query": q,
                "topic": topic,
                "query_type": "broad",
                "relevant": relevant,
            })
            qid += 1

    # ---- targeted queries: gold = topic-papers that mention the term ----
    for topic, terms in TARGETED_TERMS.items():
        for term in terms:
            relevant = _papers_with_term(corpus, topic, term)
            if not relevant:                     # skip if term doesn't appear in any topic-paper
                continue
            rows.append({
                "query_id": f"Q{qid:03d}",
                "query": f"{term} {topic.replace('_', ' ')}",   # combine the term + a topic hint
                "topic": topic,
                "query_type": "targeted",
                "relevant": relevant,
            })
            qid += 1

    # ---- Title queries — single-paper gold; gives Recall@5 a tight denominator. ----
    for topic in BROAD_QUERIES.keys():
        topic_df = corpus[corpus["topic"] == topic]                    # filter corpus by topic
        sampled = topic_df.sample(                                     # pick n papers per topic
            n=min(n_title_per_topic, len(topic_df)),
            random_state=seed,                                         # deterministic sampling
        )
        for _, row in sampled.iterrows():                              # iterate sampled papers
            rows.append({
                "query_id": f"Q{qid:03d}",
                "query": _title_to_query(row["title"]),                # noised title becomes the query
                "topic": topic,
                "query_type": "title",
                "relevant": [row["paper_id"]],                         # gold = exactly one paper
            })
            qid += 1

    return pd.DataFrame(rows)


# ---------- metrics ----------
def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k retrieved list."""
    rel = set(relevant)                          # O(1) membership test
    if not rel:                                  # avoid 0/0 if the gold set is empty
        return 0.0
    top = retrieved[:k]                          # only the first k retrieved items count
    return len([p for p in top if p in rel]) / len(rel)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalised discounted cumulative gain at k (binary relevance)."""
    rel = set(relevant)                          # O(1) membership test
    if not rel:                                  # avoid 0/0
        return 0.0
    # gains[i] = 1 if retrieved[i] is relevant else 0 — first k positions only
    gains = np.asarray([1.0 if p in rel else 0.0 for p in retrieved[:k]])
    if gains.sum() == 0:                         # short-circuit: no hits → NDCG = 0
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))   # 1/log2(rank+1) — standard discount factor
    dcg = float((gains * discounts).sum())                    # discounted cumulative gain
    ideal_n = min(len(rel), k)                                # how many positions could be filled with relevant items
    ideal_gains = np.ones(ideal_n)                            # an "ideal" ranking has all relevant at the top
    ideal_dcg = float((ideal_gains * discounts[:ideal_n]).sum())
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0          # normalise → ∈ [0,1]


def evaluate(retriever, gold: pd.DataFrame, k: int = 5,
             top_k_search: int = 20, hybrid_lambda: float | None = None):
    """Return dict with mean recall@k, ndcg@k, p95 latency (ms)."""
    queries = gold["query"].tolist()             # list of query strings
    results, latencies = retriever.search_many(  # batch search
        queries, top_k=top_k_search, hybrid_lambda=hybrid_lambda
    )
    rs, ns = [], []                              # per-query metric accumulators
    for results_q, relevant in zip(results, gold["relevant"]):
        retrieved_ids = [pid for pid, _ in results_q]              # drop scores, keep ids
        rs.append(recall_at_k(retrieved_ids, relevant, k))
        ns.append(ndcg_at_k(retrieved_ids, relevant, k))
    return {                                                       # aggregate metrics across queries
        f"recall@{k}": float(np.mean(rs)),
        f"ndcg@{k}": float(np.mean(ns)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "mean_latency_ms": float(np.mean(latencies)),
        "n_queries": len(queries),
    }


if __name__ == "__main__":
    corpus = pd.read_parquet(DATA / "corpus.parquet")              # load the corpus written by build_corpus.py
    gold = build_gold(corpus)                                      # build the gold set
    gold.to_parquet(DATA / "gold.parquet", index=False)            # persist to parquet
    print(f"Wrote {len(gold)} gold queries to {DATA / 'gold.parquet'}")
    print(gold.groupby(["topic", "query_type"]).size().to_string())  # 4 broad + 3 targeted + 6 title per topic
