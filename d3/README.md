# D3 — GraphRAG Executor, Evaluation & Safety

CSAI415 · *PDF-Papers AI Agent: Hybrid Retrieval + GraphRAG with Online Learning and AutoML*

**Team:** Khalifa . Essam . Meshal . Ahmed . Mahmoud

D3 turns the D2 retrieval stack into a **GraphRAG agent**. A question runs a
first-pass hybrid retrieval to pick seed papers; the enriched knowledge graph
selects a related **subgraph by Cypher**; supporting **chunks are pulled** from
those papers; the pools are **blended and reranked**; and the answer is composed
with **grounded citations and page ranges**. It then evaluates faithfulness,
answer-relevance and latency on a gold Q/A set, ablates vector-only vs hybrid vs
graph-guided, and ships a **safety mitigation with before/after evidence**.

Everything runs **offline** from a committed embedding cache (mongomock ·
in-memory Qdrant · NetworkX) and targets real MongoDB/Qdrant/Neo4j when those
services are configured — the same dual-path design as D2.

## What's here

```
D3/
├── D3.ipynb                 ← collective notebook, executed end-to-end (the D2 fix)
├── build_notebook.py        ← regenerates + executes D3.ipynb (nbformat + nbclient)
├── build_report.js          ← regenerates D3_Report.docx from results/ (numbers stay in sync)
├── D3_Report.docx           ← the report
├── run_card.yaml            ← active config + corpus + headline results
├── requirements.txt
├── data/
│   ├── gold_qa.json         ← 12-question gold Q/A set (2 per topic)
│   ├── embeddings.npz       ← bge chunk vectors (committed → no re-embedding needed)
│   └── cites.json           ← real CITES edges extracted from the PDFs
├── src/
│   ├── graphrag.py          ← the GraphRAG executor (4 stages)
│   ├── graph_store_d3.py    ← enriched graph (CITES + SIMILAR_TO) — NetworkX + Neo4j
│   ├── cypher_queries_d3.py ← non-trivial Cypher (the D2 "simple queries" fix)
│   ├── answerer.py          ← extractive answerer (+ optional LLM)
│   ├── rerank.py            ← semantic-MMR reranker (+ optional cross-encoder)
│   ├── evaluate.py          ← RAGAS-equivalent metrics (offline)
│   ├── ablation.py          ← 3-mode comparison + budget sweep
│   ├── safety.py            ← provenance filter + source pinning + injection scrub
│   ├── d3_pipeline.py       ← wires stores + embedder + graph + executor
│   ├── vector_index.py      ← rebuilds in-memory Qdrant from embeddings.npz
│   └── (config, embedder, ingest, hybrid_search, stores/*  — reused from D2)
├── api/main.py              ← FastAPI: /ask /search /graph/subgraph /evaluate /stats /healthz
├── scripts/
│   ├── build_embeddings.py  ← build the bge embedding cache (resumable)
│   ├── extract_cites.py     ← extract real CITES edges from the PDFs
│   ├── seed_graph.py        ← load the enriched graph (Neo4j or NetworkX)
│   └── run_eval.py / run_safety.py
├── results/                 ← eval.json, ablation.md, graphrag_examples.md,
│                              safety.json, safety_before_after.md
├── diagram/                 ← graphrag_flow.{mmd,svg,png}
└── tests/test_smoke.py      ← end-to-end pytest (offline)
```

## Quick start

### No Docker (sandbox / CI path — what produced the committed numbers)

```bash
cd D3
pip install -r requirements.txt
export FASTEMBED_CACHE_PATH=$PWD/.fastembed_cache       # bge model cache

# embeddings.npz + cites.json are committed; rebuild them only if needed:
# python scripts/build_embeddings.py   # re-run until it prints ALL_DONE (resumable)
# python scripts/extract_cites.py

PYTHONPATH=src python scripts/run_eval.py      # -> results/eval.json, ablation.md, examples
PYTHONPATH=src python scripts/run_safety.py    # -> results/safety_before_after.md
PYTHONPATH=src python build_notebook.py        # executes D3.ipynb in place
PYTHONPATH=src pytest -q                        # 5 smoke tests

uvicorn api.main:app --reload                   # http://127.0.0.1:8000/docs
# e.g.  curl 'http://127.0.0.1:8000/ask?q=how+is+concept+drift+detected&mode=graph_hybrid'
```

### With Docker (service path)

```bash
cd ../D2 && docker compose up -d                # Mongo + Qdrant + Neo4j
cd ../D3 && set -a && source ../D2/.env && set +a
PYTHONPATH=src python scripts/seed_graph.py     # loads CITES + SIMILAR_TO into Neo4j
uvicorn api.main:app --reload
```

The same `src/` runs both ways; with `NEO4J_URI` set, the D3 Cypher in
`src/cypher_queries_d3.py` runs against Neo4j (paste them into the browser).

## Headline results

GraphRAG answers on the 12-question gold set (RAGAS-equivalent, bge-scored, offline):

| Mode | Faithfulness | Answer-rel. | Answer-corr. | Recall@5 | p95 ms |
|---|---:|---:|---:|---:|---:|
| Vector-only | 1.000 | 0.891 | 0.825 | 0.944 | 22.2 |
| Hybrid | 1.000 | 0.894 | 0.823 | 0.917 | 27.7 |
| **Graph-guided** | 1.000 | 0.893 | 0.822 | 0.917 | 78.7 |

Graph expansion's value shows under a constrained first-stage budget (the
imperfect-recall regime GraphRAG targets):

| First-stage budget | Vector-only | Hybrid | Graph-expanded |
|---|---:|---:|---:|
| top-5  | 0.944 | 0.944 | **1.000** |
| top-8  | 0.972 | 0.944 | **1.000** |
| top-20 | 1.000 | 1.000 | **1.000** |

**Safety.** A poisoned passage (ranked #1, carrying an injected instruction) is
retrieved and cited **before** mitigation; the provenance filter drops it as an
untrusted source **after**, and the answer reverts to trusted, cited evidence —
attack **blocked**. Full evidence in `results/safety_before_after.md`.

**Graph.** 365 nodes / **908 edges** (vs D2's 425): adds 480 `SIMILAR_TO`
semantic edges and 3 real `CITES` edges, enabling the non-trivial Cypher.

## Notes & decisions

- **Honest ablation.** On this small, topically-clean corpus the three modes
  converge on answer quality at full budget; graph expansion mainly adds
  latency. We report that plainly and isolate where the graph *does* help
  (constrained-budget recall) rather than overclaiming.
- **Grounded answerer.** The default answerer is extractive, so every sentence
  is lifted from a retrieved chunk — faithfulness is high *by construction* and
  the evaluation measures it. An LLM answerer is an optional drop-in
  (`OPENAI_API_KEY`) with the same interface.
- **Two D2 fixes.** A single executed **collective notebook**, and a much richer
  graph with **non-trivial Cypher** (weighted multi-signal subgraph, two-hop
  semantic expansion, shortest author paths, co-citation, PageRank authority).
- **Reproducibility.** One seed (42); `embeddings.npz` committed so results
  rebuild with no re-embedding; the report reads `results/*.json` so its numbers
  cannot drift from the artefacts.


A single seed (`42`) is shared across modules so each of us reproduces the
other's results bit-for-bit.
