"""Qdrant vector store for chunk embeddings.

Qdrant requires point IDs to be ints or UUIDs, but our natural chunk id is a
string (`<paper_id>::<idx>`). We derive a stable UUIDv5 from that string for the
point id and keep the human-readable `chunk_id` in the payload, so we can always
map an ANN hit back to its Mongo record.

Service mode  : `QdrantClient(url=QDRANT_URL)` talks to the docker container.
Sandbox mode  : `QdrantClient(location=":memory:")` runs the whole thing
                in-process — real ANN search, no server. Same API.

A payload index on `paper_id`/`topic` is created so the store can do filtered
search (e.g. restrict to a pinned source — used by the D3 safety work).
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")  # fixed namespace -> stable ids


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class VectorStore:
    def __init__(self, url: str = "", collection: str = "chunks", dim: int = 384, path: str = ""):
        if url:
            self.client = QdrantClient(url=url)
            self.backend = "qdrant-server"
        elif path:
            self.client = QdrantClient(path=path)   # on-disk, persists across processes
            self.backend = "qdrant-disk"
        else:
            self.client = QdrantClient(location=":memory:")
            self.backend = "qdrant-memory"
        self.collection = collection
        self.dim = dim

    def recreate(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )
        self._make_indexes()

    def ensure_collection(self) -> None:
        """Create the collection only if absent — used by resumable seeding."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )
            self._make_indexes()

    def _make_indexes(self) -> None:
        # payload indexes enable fast filtered search / source pinning
        for field in ("paper_id", "topic"):
            try:
                self.client.create_payload_index(
                    self.collection, field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def upsert(self, chunk_ids: Sequence[str], vectors: np.ndarray, payloads: List[Dict]) -> int:
        points = [
            qm.PointStruct(
                id=_point_id(cid),
                vector=vectors[i].tolist(),
                payload={**payloads[i], "chunk_id": cid},
            )
            for i, cid in enumerate(chunk_ids)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self, query_vector: np.ndarray, top_k: int = 100,
        paper_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return [(chunk_id, score), ...]. Optional `paper_ids` pins the source set."""
        flt = None
        if paper_ids:
            flt = qm.Filter(must=[qm.FieldCondition(
                key="paper_id", match=qm.MatchAny(any=list(paper_ids)))])
        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k, query_filter=flt, with_payload=True,
        )
        return [(h.payload["chunk_id"], float(h.score)) for h in resp.points]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
