# /search evaluation (D2)

Corpus: **60 papers · 4155 chunks** (max 20 pages/PDF) · embedder=`bge-small-en-v1.5` (384-d, fastembed/ONNX) · backends mongo=`mongomock`, qdrant=`qdrant-disk`, graph=`networkx`
Gold: 59 queries (relevance restricted to in-corpus papers)

| Retriever | Recall@5 | nDCG@5 | mean ms | p95 ms |
|---|---:|---:|---:|---:|
| BM25 only (λ=1.0) | 0.624 | 0.860 | 17.9 | 28.0 |
| Dense only (λ=0.0) | 0.615 | 0.829 | 18.2 | 29.2 |
| **Hybrid (λ=0.5)** | 0.611 | 0.841 | 18.1 | 28.0 |

## Hybrid by query type

| Type | Recall@5 | nDCG@5 | n |
|---|---:|---:|---:|
| broad | 0.379 | 0.779 | 24 |
| targeted | 0.526 | 0.761 | 17 |
| title | 1.000 | 1.000 | 18 |