"""Build a reproducible bge embedding cache for the D3 corpus.

Reads the chunks the D2 ingestion already produced (data/cache/chunks.jsonl),
embeds every unique chunk with BAAI/bge-small-en-v1.5 (fastembed/ONNX, the same
encoder D2 used), and writes a single compact artefact:

    D3/data/embeddings.npz   ->  chunk_ids: (N,) str   vectors: (N, 384) float16

This decouples D3 from Qdrant's on-disk format (which is tied to a specific
qdrant-client build): every D3 entry point rebuilds an in-process Qdrant from
this npz in <1s, so the executor, evaluator, ablation, notebook and tests all
share one deterministic vector source. The job is resumable — re-run until it
prints ALL_DONE.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                 # .../D3
CACHE = ROOT.parent / "D2" / "data" / "cache" / "chunks.jsonl"
OUT = ROOT / "data" / "embeddings.npz"
OUT.parent.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FASTEMBED_CACHE_PATH",
                      str(ROOT.parent.parent / "outputs" / ".fastembed_cache"))
BATCH = 256
BUDGET_S = float(os.environ.get("EMBED_BUDGET_S", "600"))


def load_chunks():
    seen, ids, texts = set(), [], []
    for line in open(CACHE):
        c = json.loads(line)
        if c["_id"] in seen:
            continue
        seen.add(c["_id"])
        ids.append(str(c["_id"]))
        texts.append(c["text"])
    return ids, texts


def main():
    ids, texts = load_chunks()
    id2text = dict(zip(ids, texts))
    print(f"{len(ids)} unique chunks to embed", flush=True)

    done_ids, done_vecs = [], []
    if OUT.exists():
        z = np.load(OUT, allow_pickle=True)
        done_ids = list(z["chunk_ids"])
        done_vecs = list(z["vectors"])
        print(f"resuming: {len(done_ids)} already embedded", flush=True)

    done_set = set(done_ids)
    todo = [(i, id2text[i]) for i in ids if i not in done_set]
    if not todo:
        print("ALL_DONE", flush=True)
        return

    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5",
                          cache_dir=os.environ["FASTEMBED_CACHE_PATH"])

    t0 = time.time()
    for s in range(0, len(todo), BATCH):
        batch = todo[s:s + BATCH]
        vecs = list(model.embed([t for _, t in batch], batch_size=64))
        for (cid, _), v in zip(batch, vecs):
            done_ids.append(cid)
            done_vecs.append(np.asarray(v, dtype=np.float16))
        np.savez_compressed(OUT,
                            chunk_ids=np.array(done_ids, dtype=object),
                            vectors=np.vstack(done_vecs).astype(np.float16))
        print(f"  embedded {len(done_ids)}/{len(ids)} "
              f"({(time.time() - t0):.0f}s)", flush=True)
        if time.time() - t0 > BUDGET_S:
            print("BUDGET_HIT — re-run to continue", flush=True)
            return
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
