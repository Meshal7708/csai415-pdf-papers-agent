"""Build (and execute in-place) the collective D3 notebook.

D2 lost marks for having no single collective notebook — the work lived in
scripts. This regenerates `D3.ipynb` from the `src/` modules and executes it with
nbclient so every output (graph stats, Cypher results, a worked GraphRAG answer,
the evaluation/ablation tables, the safety before/after) is embedded. One
pipeline is built once and reused across cells, so it executes quickly.

    FASTEMBED_CACHE_PATH=... python3 build_notebook.py
"""
from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "D3.ipynb"


def md(c): return nbf.v4.new_markdown_cell(c)
def code(c): return nbf.v4.new_code_cell(c)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# D3 — GraphRAG Executor, Evaluation & Safety\n\n"
        "**CSAI415 · PDF-Papers AI Agent** · Team: Khalifa · Essam\n\n"
        "This is the collective notebook for D3. It runs the whole deliverable end to "
        "end on the offline cache (mongomock · in-memory Qdrant · NetworkX), so it is "
        "reproducible with no Docker:\n\n"
        "1. **GraphRAG executor** — choose subgraph by Cypher → expand to supporting "
        "chunks → hybrid blend + rerank → answer with citations & page ranges.\n"
        "2. **Enriched graph** — `CITES` + `SIMILAR_TO` edges and non-trivial Cypher "
        "(the D2 *simple-queries* fix).\n"
        "3. **Evaluation** — gold Q/A, faithfulness, answer-relevance, latency p95.\n"
        "4. **Ablation** — vector-only vs hybrid vs graph-guided + a budget sweep.\n"
        "5. **Safety** — retrieval-poisoning / prompt-injection mitigation, before vs after."))

    cells.append(md("## 0. Build the pipeline (stores + embedder + enriched graph + executor)"))
    cells.append(code(
        "import sys, json, warnings; warnings.filterwarnings('ignore')\n"
        "sys.path.insert(0, 'src')\n"
        "from d3_pipeline import build_d3_pipeline\n"
        "P = build_d3_pipeline()\n"
        "print(json.dumps(P.stats(), indent=2, default=str))"))

    cells.append(md(
        "The graph is a **superset of D2's**: same Author/Paper/Topic/Venue schema "
        "plus `CITES` (real references, sparse by design across six disjoint topics) "
        "and `SIMILAR_TO` (top-k semantic neighbours over bge paper embeddings). "
        "That richer schema is what makes the queries below non-trivial."))

    cells.append(md("## 1. Enriched graph — non-trivial Cypher (the D2 'simple queries' fix)\n\n"
                    "Each call below mirrors a parameterised query in "
                    "`src/cypher_queries_d3.py`. The NetworkX backend runs them here; "
                    "the same methods hit Neo4j when `NEO4J_URI` is set."))
    cells.append(code(
        "g = P.graph\n"
        "seeds = ['2407.05375', '2311.06396']  # two concept-drift papers\n"
        "print('weighted_subgraph (multi-signal expansion):')\n"
        "for r in g.weighted_subgraph(seeds, limit=5):\n"
        "    print('  ', r['id'], r['via'], round(r['score'],3), '-', r['title'][:48])\n"
        "print('\\nsemantic_neighbours of 2407.05375:')\n"
        "for r in g.semantic_neighbours('2407.05375', limit=4):\n"
        "    print('  ', r['id'], round(r['score'],3), '-', r['title'][:48])\n"
        "print('\\ntwo_hop_similar (variable-length):')\n"
        "for r in g.two_hop_similar('2407.05375', limit=4):\n"
        "    print('  ', r['id'], round(r['strength'],3), '-', r['title'][:48])\n"
        "print('\\npagerank_authority (centrality over SIMILAR_TO):')\n"
        "for r in g.pagerank_authority(limit=5):\n"
        "    print('  ', r['id'], round(r['score'],4), '-', r['title'][:48])"))

    cells.append(md("## 2. GraphRAG executor — the four stages on one question\n\n"
                    "Subgraph (Cypher) → expand to supporting chunks → blend + rerank "
                    "→ grounded answer with `[n]` citations carrying page ranges."))
    cells.append(code(
        "r = P.ask('how is concept drift detected in streaming data?', mode='graph_hybrid')\n"
        "print('seeds        :', r.trace['seeds'])\n"
        "print('subgraph     :', [(s['id'], '+'.join(s['via'])) for s in r.trace['subgraph'][:5]])\n"
        "print('pinned set   :', r.trace['pinned_set_size'], 'papers /',\n"
        "      r.trace['candidate_chunks'], 'candidate chunks · rerank =', r.trace['rerank'])\n"
        "print('latency_ms   : %.1f' % r.latency_ms)\n"
        "print('\\nANSWER:\\n', r.answer)\n"
        "print('\\nCITATIONS:')\n"
        "for c in r.citations:\n"
        "    print('  ', c['marker'], c['title'][:50], '(%s),' % c['paper_id'], c['page_range'])"))

    cells.append(md("## 3. Evaluation — RAGAS-equivalent metrics on the gold Q/A set\n\n"
                    "Faithfulness (answer grounded in retrieved context), answer-relevance, "
                    "answer-correctness vs the gold reference, context-recall, latency p95 — "
                    "all bge-scored, fully offline."))
    cells.append(code(
        "from evaluate import evaluate\n"
        "ev = evaluate(P, mode='graph_hybrid', top_k=5)\n"
        "o = ev['overall']\n"
        "print('graph_hybrid overall:')\n"
        "for k in ['faithfulness','answer_relevance','answer_correctness','context_recall@5','p95_latency_ms']:\n"
        "    print('  %-20s %.3f' % (k, o[k]))\n"
        "print('\\nby query type:')\n"
        "for t, v in ev['by_type'].items():\n"
        "    print('  %-9s recall=%.3f  ans_rel=%.3f  n=%d' % (t, v['context_recall@5'], v['answer_relevance'], v['n']))"))

    cells.append(md("## 4. Ablation — vector-only vs hybrid vs graph-guided\n\n"
                    "Full tables are produced by `scripts/run_eval.py` (3-mode quality + a "
                    "first-stage-budget sweep) and loaded here so the notebook stays fast."))
    cells.append(code(
        "ab = json.load(open('results/eval.json'))\n"
        "print('Answer-quality ablation (top_k=5):')\n"
        "print('  mode           faith  ans_rel ans_corr recall  p95ms')\n"
        "for m in ['vector_only','hybrid','graph_hybrid']:\n"
        "    o = ab[m]['overall']\n"
        "    print('  %-13s %.3f  %.3f  %.3f   %.3f  %.1f' % (m, o['faithfulness'],\n"
        "          o['answer_relevance'], o['answer_correctness'], o['context_recall@5'], o['p95_latency_ms']))\n"
        "print('\\nContext-recall vs first-stage budget (graph value shows when recall is imperfect):')\n"
        "for fs, b in ab['recall_vs_budget'].items():\n"
        "    print('  top-%-2s  vector=%.3f  hybrid=%.3f  graph=%.3f' % (fs, b['vector_only'], b['hybrid'], b['graph_expanded']))"))
    cells.append(md(
        "At the full budget the small, topically-clean corpus saturates and the three "
        "modes converge on answer quality (graph adds latency). Graph expansion earns "
        "its keep where it is designed to — **recovering context-recall when the "
        "first-stage budget is tight** (top-5 hybrid: 0.972 → 1.000)."))

    cells.append(md("## 5. Safety — retrieval poisoning / prompt injection (before vs after)\n\n"
                    "`scripts/run_safety.py` injects a poisoned passage (clone of the top "
                    "hit + an injected instruction) and answers with the mitigation off, "
                    "then on. Summary loaded here; full evidence in "
                    "`results/safety_before_after.md`."))
    cells.append(code(
        "s = json.load(open('results/safety.json'))\n"
        "print('threat    :', s['threat'])\n"
        "print('mitigation:', s['mitigation'])\n"
        "print('BEFORE    : attack_succeeded =', s['before']['signals']['attack_succeeded'],\n"
        "      '| poison cited =', s['before']['signals']['poison_in_citations'])\n"
        "print('AFTER     : attack_succeeded =', s['after']['signals']['attack_succeeded'],\n"
        "      '| dropped   =', s['after']['filter_report']['n_dropped'],\n"
        "      '(%s)' % ', '.join(d['reason'] for d in s['after']['filter_report']['dropped']))\n"
        "print('RESULT    : attack blocked =', s['result']['blocked'])"))

    cells.append(md(
        "## Summary — D3 rubric coverage\n\n"
        "- **GraphRAG pipeline (8%)** — §2: Cypher subgraph selection → chunk "
        "expansion → hybrid blend → semantic-MMR rerank → grounded answers with "
        "page-range citations.\n"
        "- **Evaluation (5%)** — §3–4: faithfulness, answer-relevance/correctness, "
        "context-recall, latency p95; thorough ablation with a budget sweep.\n"
        "- **Safety (2%)** — §5: provenance filter + source pinning + injection "
        "scrubbing, with before/after evidence and documented limits.\n\n"
        "Plus the two D2 fixes: this **collective notebook**, and a much richer graph "
        "with **non-trivial Cypher** (§1)."))

    nb["cells"] = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def main():
    nb = build()
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(nb, OUT)
    print("NOTEBOOK_WRITTEN", OUT)


if __name__ == "__main__":
    main()
