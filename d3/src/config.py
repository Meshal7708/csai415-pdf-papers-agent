"""Central configuration for the D2 retrieval stack.

Everything is environment-driven so the *same* code runs in two modes:

* **Service mode** — `docker compose up` starts MongoDB, Qdrant and Neo4j and
  the env vars below point at them. This is what you run on your machine.
* **Embedded / sandbox mode** — if a service URL is absent (or unreachable),
  each store transparently falls back to an in-process equivalent
  (`mongomock`, Qdrant `:memory:`, an in-memory NetworkX graph). This is how we
  generate the metrics table without Docker, and how the pytest smoke tests run
  in CI.

The fallbacks expose the *same* method signatures as the real adapters, so no
downstream code (ingestion, search, API) branches on the backend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
RESULTS = ROOT / "results"
for _d in (DATA, PDF_DIR, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


@dataclass
class Settings:
    # ---- stores ----
    mongo_uri: str = field(default_factory=lambda: _env("MONGO_URI", ""))
    mongo_db: str = field(default_factory=lambda: _env("MONGO_DB", "papers_ai"))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", ""))
    qdrant_path: str = field(default_factory=lambda: _env("QDRANT_PATH", ""))  # on-disk local mode
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "chunks"))
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", ""))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "test1234"))

    # ---- embedder ----
    # "bge"  -> BAAI/bge-small-en-v1.5 (384-dim, the brief's suggested encoder)
    # "hash" -> deterministic offline embedder, only used for fast smoke tests
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "bge"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_dim: int = 384

    # ---- ingestion ----
    chunk_size_chars: int = 900      # ~180 tokens per chunk
    chunk_overlap_chars: int = 150   # ~17% overlap so sentences don't get split blind
    min_chunk_chars: int = 120       # drop tiny tail fragments
    # Cap pages per PDF: arXiv papers' substantive content is up front; the long
    # tail (references/appendices) bloats chunk counts with low-value text. 0 = no cap.
    max_pages: int = field(default_factory=lambda: int(_env("MAX_PAGES", "20")))

    # ---- retrieval ----
    # Frozen from the D1 Optuna winner (run_card.yaml). 1.0 = BM25 only, 0.0 = dense only.
    hybrid_lambda: float = field(default_factory=lambda: float(_env("HYBRID_LAMBDA", "0.5")))
    candidate_pool: int = 100        # ANN + BM25 pool size before fusion

    seed: int = 42

    def using_real_mongo(self) -> bool:
        return bool(self.mongo_uri)

    def using_real_qdrant(self) -> bool:
        return bool(self.qdrant_url)

    def using_real_neo4j(self) -> bool:
        return bool(self.neo4j_uri)


SETTINGS = Settings()
