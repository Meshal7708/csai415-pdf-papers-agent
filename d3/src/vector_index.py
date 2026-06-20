"""Rebuild an in-process Qdrant index from the D3 embedding cache.

D2 persisted embeddings inside Qdrant's on-disk format, which is tied to a
specific qdrant-client build. D3 instead keeps the vectors in a portable
`data/embeddings.npz` (chunk_id -> bge-small-en-v1.5 vector) and rehydrates a
fresh in-memory Qdrant collection from it in well under a second. Every D3 entry
point (executor, evaluator, ablation, API, notebook, tests) shares this one
deterministic vector source, so numbers reproduce bit-for-bit.

The returned object is a D2 `VectorStore` whose `.search()` the existing
`HybridSearcher` already knows how to call — nothing downstream changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from stores.vector_store import VectorStore

EMB = Path(__file__).resolve().parents[1] / "data" / "embeddings.npz"


def load_embeddings(path: Path = EMB):
    z = np.load(path, allow_pickle=True)
    ids = [str(x) for x in z["chunk_ids"]]
    vecs = z["vectors"].astype(np.float32)            # float16 cache -> float32 for ANN
    return ids, vecs


def build_vector_store(mongo, collection: str = "chunks", dim: int = 384) -> VectorStore:
    """In-memory Qdrant populated from the npz, with payloads pulled from Mongo."""
    ids, vecs = load_embeddings()
    meta: Dict[str, Dict] = {c["_id"]: c for c in mongo.all_chunks()}
    vs = VectorStore(url="", collection=collection, dim=dim, path="")   # :memory:
    vs.recreate()
    payloads: List[Dict] = []
    keep_ids, keep_vecs = [], []
    for cid, v in zip(ids, vecs):
        m = meta.get(cid)
        if not m:
            continue
        payloads.append({"paper_id": m["paper_id"], "topic": m.get("topic", ""),
                         "page_start": m["page_start"], "page_end": m["page_end"]})
        keep_ids.append(cid)
        keep_vecs.append(v)
    vs.upsert(keep_ids, np.vstack(keep_vecs), payloads)
    return vs
