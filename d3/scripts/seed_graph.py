"""Seed the enriched D3 graph (CITES + SIMILAR_TO) into the active backend.

Builds the full D3 pipeline, which derives SIMILAR_TO from paper embeddings and
loads the real CITES edges, then loads them into the graph store. With NEO4J_URI
set this populates Neo4j (so you can run the D3 Cypher in the browser); with no
services it builds the in-process NetworkX graph used by the sandbox/CI path.

Prereqs (offline path): scripts/build_embeddings.py and scripts/extract_cites.py
have produced data/embeddings.npz and data/cites.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d3_pipeline import build_d3_pipeline          # noqa: E402


def main():
    p = build_d3_pipeline()
    print(json.dumps({"graph": p.graph_stats, "stores": p.stats()}, indent=2, default=str))
    print("SEED_GRAPH_DONE")


if __name__ == "__main__":
    main()
