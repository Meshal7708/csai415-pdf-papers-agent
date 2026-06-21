"""RAGAS-equivalent evaluation, computed fully offline with bge embeddings.

We don't call a judge LLM (no API needed, fully reproducible), but we compute the
same quantities RAGAS reports, using sentence embeddings as the semantic scorer:

* faithfulness      — fraction of the answer's sentences whose content is
                      supported by the retrieved context (max cosine to any
                      context chunk >= tau). High = the answer is grounded.
* answer_relevance  — cosine(answer, question): does the answer address the ask?
* answer_correctness— cosine(answer, gold reference answer).
* context_recall@k  — did the retrieved citations include the gold papers?
* latency p95       — end-to-end /ask latency.

`evaluate()` runs the gold set through one executor mode and aggregates overall +
by query_type. Used by the ablation and the notebook.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold_qa.json"
TAU = 0.60                                          # support threshold for faithfulness


def _sents(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").replace("\n", " "))
    return [s.strip(" []0123456789") for s in parts if len(s.strip()) > 20]


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Scorer:
    """Embeds arbitrary text with the same bge model (document mode) and caches."""
    def __init__(self, embedder):
        self.embedder = embedder
        self._cache: Dict[str, np.ndarray] = {}

    def emb(self, text: str) -> np.ndarray:
        if text not in self._cache:
            v = self.embedder.encode_documents([text or " "])[0]
            self._cache[text] = _norm(v)
        return self._cache[text]

    def cos(self, a: str, b: str) -> float:
        return float(self.emb(a) @ self.emb(b))

    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        claims = _sents(answer)
        if not claims or not contexts:
            return 0.0
        ctx = [self.emb(c) for c in contexts]
        supported = 0
        for cl in claims:
            e = self.emb(cl)
            if max(float(e @ c) for c in ctx) >= TAU:
                supported += 1
        return supported / len(claims)


def load_gold() -> List[Dict]:
    return json.loads(GOLD.read_text())["items"]


def evaluate(pipeline, mode: str = "graph_hybrid", top_k: int = 5,
             rerank: bool = True, scorer: Scorer = None) -> Dict:
    gold = load_gold()
    scorer = scorer or Scorer(pipeline.embedder)
    rows, by_type = [], {}
    for item in gold:
        res = pipeline.executor.answer(item["question"], mode=mode, top_k=top_k, rerank=rerank)
        ans, contexts = res.answer, res.contexts
        # context-recall = did the retrieved top-k context cover the gold papers
        retrieved = {str(p) for p in res.retrieved_papers}
        rel = set(item["relevant"])
        recall = len(retrieved & rel) / len(rel) if rel else 0.0
        row = {
            "id": item["id"], "type": item["query_type"],
            "faithfulness": scorer.faithfulness(ans, contexts),
            "answer_relevance": scorer.cos(ans, item["question"]),
            "answer_correctness": scorer.cos(ans, item["reference_answer"]),
            "context_recall": recall,
            "latency_ms": res.latency_ms,
        }
        rows.append(row)
        by_type.setdefault(item["query_type"], []).append(row)

    def agg(rs):
        return {
            "faithfulness": float(np.mean([r["faithfulness"] for r in rs])),
            "answer_relevance": float(np.mean([r["answer_relevance"] for r in rs])),
            "answer_correctness": float(np.mean([r["answer_correctness"] for r in rs])),
            "context_recall@%d" % top_k: float(np.mean([r["context_recall"] for r in rs])),
            "p95_latency_ms": float(np.percentile([r["latency_ms"] for r in rs], 95)),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rs])),
            "n": len(rs),
        }

    out = {"mode": mode, "top_k": top_k, "overall": agg(rows),
           "by_type": {t: agg(rs) for t, rs in by_type.items()},
           "per_query": rows}
    return out
