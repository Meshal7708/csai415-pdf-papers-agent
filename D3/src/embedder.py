"""Dense embedders for the hybrid retriever.

`BGEEmbedder` is the real one used everywhere (BAAI/bge-small-en-v1.5, 384-dim).
It runs the model through **fastembed** (ONNX runtime) rather than
sentence-transformers/torch: same weights, ~10x smaller install, CPU-only, and
it shares Qdrant's tokenisation. bge is asymmetric — documents are embedded
as-is while queries get the model's search-instruction prefix (fastembed's
`query_embed` applies this automatically), which measurably helps short queries.

`HashEmbedder` is a deterministic, dependency-free stand-in used only by the
smoke tests so CI needs no model download.

Both expose the same interface:
    encode_documents(list[str]) -> np.ndarray (N, dim)  float32
    encode_query(str)          -> np.ndarray (dim,)     float32
    dim: int
"""
from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # lazy import
        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH") or None
        self.model = TextEmbedding(model_name, cache_dir=cache_dir)
        self.dim = 384

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        v = np.vstack(list(self.model.embed(texts, batch_size=32)))
        return v.astype(np.float32, copy=False)

    def encode_query(self, text: str) -> np.ndarray:
        v = next(iter(self.model.query_embed(text)))
        return np.asarray(v, dtype=np.float32)


class HashEmbedder:
    """Hashing trick -> L2-normalised vector. Offline, deterministic, no weights.

    Quality is far below bge; this exists purely so tests run without network.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def get_embedder(kind: str = "bge", model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384):
    if kind == "bge":
        return BGEEmbedder(model_name)
    if kind == "hash":
        return HashEmbedder(dim)
    raise ValueError(f"unknown embedder kind: {kind}")
