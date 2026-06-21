"""Run the D3 evaluation + ablation and write results artefacts.

    results/eval.json            full metrics for all three modes
    results/ablation.md          the comparison table + graph-vs-hybrid lift
    results/graphrag_examples.md  worked examples with the 4-stage trace
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d3_pipeline import build_d3_pipeline          # noqa: E402
from ablation import run_ablation, ablation_markdown, recall_vs_budget  # noqa: E402

RES = ROOT / "results"; RES.mkdir(exist_ok=True)
TOP_K = 5


def write_examples(pipe, questions):
    out = ["# GraphRAG executor — worked examples", "",
           "Each shows the four stages: seed papers → weighted subgraph "
           "(Cypher) → expanded chunk pool → grounded answer with page-range "
           "citations.", ""]
    for q in questions:
        r = pipe.ask(q, mode="graph_hybrid")
        out.append(f"## Q: *{q}*")
        out.append(f"- **Seeds:** {', '.join(map(str, r.trace['seeds']))}")
        sg = "; ".join(f"{s['id']} ({'+'.join(s['via'])}, {s['score']})"
                       for s in r.trace["subgraph"][:5])
        out.append(f"- **Weighted subgraph:** {sg}")
        out.append(f"- **Pinned set:** {r.trace['pinned_set_size']} papers · "
                   f"{r.trace['candidate_chunks']} candidate chunks · "
                   f"rerank={r.trace.get('rerank')}")
        out.append(f"- **Latency:** {r.latency_ms:.1f} ms")
        out.append(f"\n**Answer.** {r.answer}\n")
        out.append("**Citations:** " + "; ".join(
            f"{c['marker']} {c['title']} ({c['paper_id']}), {c['page_range']}"
            for c in r.citations))
        out.append("")
    (RES / "graphrag_examples.md").write_text("\n".join(out))


def main():
    pipe = build_d3_pipeline()
    print("pipeline:", json.dumps(pipe.stats(), default=str)[:200], flush=True)

    results = run_ablation(pipe, top_k=TOP_K)
    budget = recall_vs_budget(pipe, budgets=(5, 8, 20))
    payload = {m: r for m, r in results.items()}
    payload["recall_vs_budget"] = budget
    (RES / "eval.json").write_text(json.dumps(payload, indent=2, default=str))
    md = ablation_markdown(results, top_k=TOP_K, budget=budget)
    (RES / "ablation.md").write_text(md)
    print(md, flush=True)

    write_examples(pipe, [
        "how is concept drift detected in streaming data?",
        "what is the goal of neural architecture search in AutoML?",
        "how do vision transformers reduce computation through token sampling?",
    ])
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
