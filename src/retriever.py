"""Hybrid BM25 + dense retriever with tunable hyperparameters.

Search space (set by Optuna in src/automl.py):
    k             : int   — top-k returned by the dense kNN side
    metric        : str   — {"cosine", "dot", "euclidean"}
    svd_dim       : int   — 0 disables SVD; otherwise TruncatedSVD components
    l2_normalize  : bool  — L2-normalize dense vectors before scoring
    hybrid_lambda : float — final = lam * bm25 + (1 - lam) * dense (both min-max'd)

Lexical side: BM25Okapi from rank_bm25.
Dense side  : TF-IDF (sklearn) → optional TruncatedSVD → optional L2 norm → kNN.

NOTE on choice of dense embedder
--------------------------------
The brief lists `sentence-transformers/bge-small-en` as the *suggested* dense
encoder, and we will use it in D2 once full ingestion is wired up. For D1, the
sandbox is fully offline (no network), so we use TF-IDF + TruncatedSVD as the
dense vectors. This:
  • makes the SVD-dim Optuna axis directly meaningful (LSA components),
  • keeps the notebook reproducible and CPU-only,
  • leaves the HybridRetriever interface unchanged when we swap to BGE in D2
    (the dense matrix becomes shape (N, 384) instead of (N, svd_dim)).
"""

from __future__ import annotations               # postponed-evaluation type hints

import time                                      # used to measure per-query latency
from dataclasses import dataclass, field         # @dataclass for the config struct
from typing import Iterable, List, Sequence, Tuple   # type hints

import numpy as np                               # numerical operations on the dense matrix
import pandas as pd                              # corpus DataFrame
from rank_bm25 import BM25Okapi                  # classic BM25 implementation (no external service required)
from sklearn.decomposition import TruncatedSVD   # LSA-style dense projection
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF features (the dense base)
from sklearn.preprocessing import normalize      # row-wise L2 norm, used optionally before scoring


# ---------- helpers ----------
def _tokenize(text: str) -> List[str]:
    """Cheap, deterministic tokenizer for BM25."""
    # Lowercase, split on whitespace, drop tokens that aren't alphanumeric or hyphenated.
    return [t for t in text.lower().split() if t.isalnum() or "-" in t]


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Per-query min-max scaling to [0, 1]; safe on constant vectors."""
    s_min = float(scores.min())                  # smallest raw score
    s_max = float(scores.max())                  # largest raw score
    if s_max - s_min < 1e-12:                    # guard against division-by-zero on degenerate inputs
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - s_min) / (s_max - s_min)    # standard min-max normalisation


# ---------- retriever ----------
@dataclass                                       # auto-generated __init__ / __repr__
class HybridConfig:
    k: int = 10                                  # how many candidates the dense kNN side returns (search space [3,30])
    metric: str = "cosine"                       # cosine | dot | euclidean — search space {3 enums}
    svd_dim: int = 128                           # 0 = no SVD; search space {0, 64, 128, 256}
    l2_normalize: bool = True                    # whether to L2-normalize dense vectors before scoring
    hybrid_lambda: float = 0.5                   # 1.0 = BM25-only, 0.0 = dense-only — search space [0,1]
    embedder: str = "tfidf"                      # "tfidf" (old LSA path) or "bge" (semantic embeddings)
    embed_model: str = "BAAI/bge-small-en-v1.5"  # used only when embedder == "bge"

class HybridRetriever:
    """BM25 + (TF-IDF[+SVD]) hybrid retriever, fit once over a corpus."""

    def __init__(self, cfg: HybridConfig):
        self.cfg = cfg                           # immutable config struct
        self.bm25: BM25Okapi | None = None       # populated by fit()
        self.tfidf: TfidfVectorizer | None = None   # populated by fit()
        self.svd: TruncatedSVD | None = None     # populated by fit() (or stays None if svd_dim==0)
        self.dense_matrix: np.ndarray | None = None # (N_docs, dim) dense matrix
        self.paper_ids: List[str] = []           # row-index ↔ paper_id
        self.embed_model = None                  # SentenceTransformer, populated by fit() on the bge path


    # ---- fitting ----
    def fit(self, df: pd.DataFrame) -> "HybridRetriever":
        texts = df["text"].tolist()              # 'text' is the combined title+abstract field
        self.paper_ids = df["paper_id"].tolist() # parallel list, used to map row → id later

        # BM25 — index over the tokenised corpus
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts])

        if self.cfg.embedder == "bge":
            # Dense side = real semantic embeddings (bge-small-en-v1.5, 384-dim).
            # This is the D2 upgrade: a genuine neural encoder replaces TF-IDF/SVD,
            # so the dense signal bridges vocabulary gaps instead of staying bag-of-words.
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(self.cfg.embed_model)
            self.tfidf = None                    # unused on the bge path
            self.svd = None
            dense = self.embed_model.encode(
                texts, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True,
            )                                    # (N_docs, 384)
        else:
            # TF-IDF — unigrams + bigrams, drop very common terms (>95% of docs)
            self.tfidf = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95
            )
            X = self.tfidf.fit_transform(texts)      # sparse matrix (N_docs, V) where V = vocab size

            # Optional SVD (LSA) — projects the sparse TF-IDF matrix into a dense low-dim space
            if self.cfg.svd_dim and self.cfg.svd_dim > 0:
                # cap n_components so SVD doesn't fail on small corpora
                n_components = min(self.cfg.svd_dim, X.shape[1] - 1, X.shape[0] - 1)
                self.svd = TruncatedSVD(n_components=n_components, random_state=42)
                dense = self.svd.fit_transform(X)    # (N_docs, n_components)
            else:
                self.svd = None                      # explicitly None — used as a flag at search time
                dense = X.toarray()                  # if no SVD, materialise the TF-IDF matrix as dense


        # Optional L2 norm — makes cosine equivalent to dot product
        if self.cfg.l2_normalize:
            dense = normalize(dense, norm="l2", axis=1)   # axis=1 → row-wise (per-document)

        # Cast to float32 to halve memory footprint without hurting retrieval quality
        self.dense_matrix = dense.astype(np.float32, copy=False)
        return self                              # fluent API: HybridRetriever(cfg).fit(df)

    # ---- scoring ----
    def _dense_scores(self, query_text: str) -> np.ndarray:
        """Return per-document dense scores for one query (higher = more relevant)."""
        assert self.dense_matrix is not None
        if self.cfg.embedder == "bge":
            # Same model as fit(); the query instruction helps bge on short queries.
            q_text = "Represent this sentence for searching relevant passages: " + query_text
            qd = self.embed_model.encode([q_text], convert_to_numpy=True)   # (1, 384)
        else:
            q = self.tfidf.transform([query_text])    # (1, V) sparse vector for the query
            if self.svd is not None:                  # if we used SVD at fit time, project the query the same way
                qd = self.svd.transform(q)
            else:
                qd = q.toarray()                      # otherwise materialise the sparse query as dense
        if self.cfg.l2_normalize:                 # apply the same L2 norm we used at fit time
            qd = normalize(qd, norm="l2", axis=1)
        qd = qd.astype(np.float32, copy=False)    # match dtype of the doc matrix to enable fast BLAS

        m = self.cfg.metric                       # local alias for readability
        if m == "cosine":
            # If both vectors are L2-normed, dot product == cosine similarity (cheaper).
            if self.cfg.l2_normalize:
                return (self.dense_matrix @ qd[0]).astype(np.float64)
            # Otherwise, normalise on the fly — slow path, but fully correct.
            a = normalize(self.dense_matrix, norm="l2", axis=1)
            b = normalize(qd, norm="l2", axis=1)[0]
            return (a @ b).astype(np.float64)
        if m == "dot":                            # raw inner product (no norm)
            return (self.dense_matrix @ qd[0]).astype(np.float64)
        if m == "euclidean":
            # Smaller distance → more relevant; we negate so "higher is better" matches the other metrics.
            d = np.linalg.norm(self.dense_matrix - qd[0], axis=1)
            return -d.astype(np.float64)
        raise ValueError(f"unknown metric: {m}")  # defensive — should never trigger if the config is valid

    def _bm25_scores(self, query_text: str) -> np.ndarray:
        """Return per-document BM25 scores for one query."""
        assert self.bm25 is not None
        return np.asarray(self.bm25.get_scores(_tokenize(query_text)), dtype=np.float64)

    # ---- public API ----
    def search(
        self,
        query: str,
        top_k: int | None = None,
        hybrid_lambda: float | None = None,
    ) -> List[Tuple[str, float]]:
        """Return [(paper_id, score), ...] sorted desc."""
        top_k = top_k or self.cfg.k                                       # arg overrides config
        lam = self.cfg.hybrid_lambda if hybrid_lambda is None else hybrid_lambda

        bm25 = _minmax(self._bm25_scores(query))                          # min-max so BM25 ∈ [0,1]
        dens = _minmax(self._dense_scores(query))                         # min-max so dense ∈ [0,1] (comparable scale)
        fused = lam * bm25 + (1.0 - lam) * dens                           # the fusion equation: λ controls the mix

        idx = np.argsort(-fused)[:top_k]                                  # top-k indices (negative for descending sort)
        return [(self.paper_ids[i], float(fused[i])) for i in idx]        # map indices → (paper_id, score)

    def search_many(
        self,
        queries: Sequence[str],
        top_k: int | None = None,
        hybrid_lambda: float | None = None,
    ) -> Tuple[List[List[Tuple[str, float]]], List[float]]:
        """Batch search; also returns per-query latency (ms)."""
        results = []                              # list of result-lists (one per query)
        latencies = []                            # parallel list of per-query latencies in ms
        for q in queries:
            t0 = time.perf_counter()              # high-resolution monotonic clock
            r = self.search(q, top_k=top_k, hybrid_lambda=hybrid_lambda)
            latencies.append((time.perf_counter() - t0) * 1000.0)         # convert s → ms
            results.append(r)
        return results, latencies


# ---------- smoke test ----------
if __name__ == "__main__":
    from pathlib import Path
    df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "corpus.parquet")
    r = HybridRetriever(HybridConfig(k=5)).fit(df)                        # fit on the 150-paper corpus
    out, lat = r.search_many(                                             # three sanity-check queries
        ["graph retrieval over scientific papers",
         "online learning with concept drift",
         "automl hyperparameter search"],
        top_k=5,
    )
    for q, results in zip(["graph retrieval", "online drift", "automl"], out):
        print(q, "→", [pid for pid, _ in results])                        # expect topic-aligned paper_ids
    print(f"avg latency: {np.mean(lat):.2f}ms  p95: {np.percentile(lat, 95):.2f}ms")
