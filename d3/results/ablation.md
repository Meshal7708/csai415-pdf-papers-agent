# D3 ablation — vector-only vs hybrid vs graph-guided

Gold set: 12 Q/A (2 per topic). Metrics are RAGAS-equivalent (bge-scored, offline). Higher is better except latency.

| Mode | Faithfulness | Answer-rel. | Answer-corr. | Recall@5 | p95 ms |
|---|---:|---:|---:|---:|---:|
| Vector-only | 1.000 | 0.891 | 0.825 | 0.944 | 22.2 |
| Hybrid | 1.000 | 0.894 | 0.823 | 0.917 | 27.7 |
| **Graph-guided** | 1.000 | 0.893 | 0.822 | 0.917 | 78.7 |

## Graph-guided vs hybrid (relative)

- **faithfulness**: 1.000 → 1.000  (+0.0%)
- **answer_relevance**: 0.894 → 0.893  (-0.1%)
- **answer_correctness**: 0.823 → 0.822  (-0.1%)
- **context_recall@5**: 0.917 → 0.917  (+0.0%)

At the full first-stage budget the corpus is small enough that vector, hybrid and graph-guided converge on answer quality — graph expansion adds latency without changing the top-k answer. Its value shows where it is designed to: recovering recall when the first-stage budget is tight (next table).

## Retrieval coverage vs first-stage budget (context-recall)

Graph expansion = hybrid seeds → weighted subgraph (Cypher) → pinned expansion. Lower budget = more imperfect first-stage recall.

| First-stage budget | Vector-only | Hybrid | Graph-expanded |
|---|---:|---:|---:|
| top-5 | 0.944 | 0.944 | **1.000** |
| top-8 | 0.972 | 0.944 | **1.000** |
| top-20 | 1.000 | 1.000 | **1.000** |

At top-5, graph expansion lifts context-recall 0.944 → 1.000; the gain closes as the budget grows and the small corpus saturates.