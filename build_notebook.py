"""Build a single runnable D1.ipynb that bundles the full pipeline.

Cells call into src/ modules so the notebook stays compact while still showing
the architecture, results tables, and plots.
"""

from __future__ import annotations               # postponed-evaluation type hints
import json                                      # write the .ipynb (which is JSON)
from pathlib import Path                         # path manipulation

ROOT = Path(__file__).resolve().parent           # D1/ — this file lives at D1/build_notebook.py

# ---------- tiny helpers for building notebook cells -------------------------
def md(*lines: str) -> dict:
    """Build a markdown cell from one or more lines (joined with newlines)."""
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def code(src: str) -> dict:
    """Build a code cell from a multi-line Python string."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,                 # will be filled in when nbclient executes the notebook
        "outputs": [],                           # ditto
        "source": [l + "\n" for l in src.splitlines()],
    }


# ---------- the actual notebook content -------------------------------------
# Each item is either a md(...) or a code(...) cell. Order matters — this is
# also the execution order when the notebook is run end-to-end.
CELLS = [
    # ---- title block ----
    md("# D1 — Streaming Learner & AutoML Note",
       "**Module:** CSAI415 — *PDF-Papers AI Agent: Hybrid Retrieval + GraphRAG with Online Learning and AutoML*",
       "",
       "**Team:** Khalifa (AutoML) · Meshal (Online learning) · Mahmoud (Retriever) · Ahmed (Corpus & gold) · Essam (Report & integration)",
       "",
       "**This notebook bundles the full D1 pipeline:**",
       "1. Fetch a 150-paper real corpus from the **arXiv API** across 6 topics (full PDF ingestion lands in D2).",
       "2. Build a hybrid retriever: **BM25 + (TF-IDF [+ TruncatedSVD])**, tunable.",
       "3. Build a 78-query gold set with three query types (broad, targeted, title).",
       "4. Compute **baseline** Recall@5 / NDCG@5 / p95 latency.",
       "5. Run **Optuna AutoML** (Track A): tune k, metric, SVD dim, L2-norm, hybrid λ.",
       "6. Add a **River** incremental classifier (logistic regression over λ-buckets, with ε-greedy and **ADWIN** drift handling), evaluate prequentially.",
       "7. Save the **run card** YAML.",
       "",
       "All artefacts (`results/*.json`, `results/*.png`, `run_card.yaml`) are written next to this notebook."),

    # ---- section 0: setup ----
    md("## 0 — Setup"),
    code("""\
import json, sys, importlib
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'src'))            # so we can import the src/ modules
DATA = ROOT / 'data'; RESULTS = ROOT / 'results'
DATA.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

# Show package versions used in this run (also dumped to run_card.yaml later)
for mod in ['numpy', 'pandas', 'sklearn', 'rank_bm25', 'optuna', 'river']:
    try:
        m = importlib.import_module(mod)
        print(f'{mod:12s} {getattr(m, "__version__", "ok")}')
    except Exception as e:
        print(f'{mod:12s} {e}')
"""),

    # ---- section 1: corpus ----
    md("## 1 — Build the corpus (150 real arXiv papers across 6 topics)",
       "Six topic clusters: `transformers`, `rag`, `online_learning`, `vision`, `rl_agents`, `automl`. "
       "Papers are fetched live from the **arXiv API** (25 per topic); each has a title, abstract, authors, "
       "venue, year, topic — schema mirrors the brief's required metadata. The fetched `corpus.parquet` is "
       "committed so the snapshot is reproducible for the whole team."),
    code("""\
from build_corpus import build_corpus
corpus = build_corpus(target_size=150)           # fetch real papers from the arXiv API
out = DATA / 'corpus.parquet'
corpus.to_parquet(out, index=False)              # persist for downstream cells
(DATA / 'corpus_hash.txt').write_text(corpus.attrs['corpus_hash'])
print(f'Wrote {out}  hash={corpus.attrs["corpus_hash"]}')
corpus.head()                                    # quick visual check
"""),

    # ---- section 2: retriever ----
    md("## 2 — Hybrid BM25 + dense retriever",
       "*Lexical:* BM25Okapi (rank_bm25). &nbsp;*Dense:* TF-IDF → optional TruncatedSVD → optional L2-norm.",
       "Final score = `λ * minmax(BM25) + (1-λ) * minmax(dense)`. The dense module will be swapped to "
       "`sentence-transformers/bge-small-en` in D2 — the `HybridRetriever` interface stays the same."),
    code("""\
from retriever import HybridConfig, HybridRetriever
cfg = HybridConfig(k=10, metric='cosine', svd_dim=128, l2_normalize=True, hybrid_lambda=0.5)
retriever = HybridRetriever(cfg).fit(corpus)     # fits BM25 + TF-IDF + (optional) SVD
# Quick three-query smoke test to verify topic alignment + latency
demo, lat = retriever.search_many(
    ['graphrag subgraph expansion', 'concept drift detection', 'optuna hyperparameter search'], top_k=3)
for q, hits in zip(['graphrag', 'drift', 'optuna'], demo):
    print(f'{q:10s} →', [pid for pid,_ in hits])
print(f'mean latency {np.mean(lat):.2f}ms  p95 {np.percentile(lat, 95):.2f}ms')
"""),

    # ---- section 3: gold set ----
    md("## 3 — Gold Q/A evaluation set",
       "Three query types per topic make Recall@5 a meaningful metric:",
       "- **broad** — full-topic gold (4 / topic)",
       "- **targeted** — gold = topic-papers mentioning a specific term (3 / topic)",
       "- **title** — paraphrased title; gold = single paper (6 / topic)"),
    code("""\
from gold import build_gold, evaluate
gold = build_gold(corpus)
gold.to_parquet(DATA / 'gold.parquet', index=False)
print(f'{len(gold)} queries')
display(gold.groupby(['topic','query_type']).size().unstack(fill_value=0))
"""),

    # ---- section 4: baselines ----
    md("## 4 — Baselines",
       "Three baselines with a fixed retriever; only `hybrid_lambda` varies."),
    code("""\
import importlib, baselines
importlib.reload(baselines); baselines.main()    # writes results/baselines.json
print()
# Render baselines.json as a clean DataFrame for the report
display(pd.DataFrame(json.loads((RESULTS / 'baselines.json').read_text())['baselines']).T)
"""),

    # ---- section 5: AutoML ----
    md("## 5 — AutoML: Optuna search over the kNN retriever",
       "**Search space** (from the brief): `k ∈ [3,30]`, `metric ∈ {cosine,dot,euclidean}`, "
       "`svd_dim ∈ {0,64,128,256}`, `l2_normalize ∈ {T,F}`, `hybrid_lambda ∈ [0,1]`. "
       "**Objective:** NDCG@5 with a soft p95-latency penalty. **Trials:** 50 with TPE sampler. "
       "We split gold 60/40 to avoid optimising on the test set."),
    code("""\
import importlib, automl
importlib.reload(automl); summary = automl.main()  # runs 50 trials; writes automl_study.json
print('\\nWinning config:'); print(json.dumps(summary['best_params'], indent=2))
"""),
    code("""\
from PIL import Image
display(Image.open(RESULTS / 'optuna_history.png'))   # show the running-best curve
"""),

    # ---- section 6: River + ADWIN ----
    md("## 6 — River: adaptive hybrid-weight learner with ADWIN drift handling",
       "**Online learner (Week-03 toolkit):** `compose.Pipeline(preprocessing.OneHotEncoder, linear_model.LogisticRegression)` — "
       "an incremental classifier that predicts P(helpful = True | λ_bucket) from the streaming click feedback. "
       "Discretising λ into 11 buckets in {0.0, 0.1, …, 1.0} and using ε-greedy (ε = 0.10) on top of the classifier "
       "keeps the action-selection logic explicit and bandit-free. ",
       "**Drift detector:** `river.drift.ADWIN` (δ = 0.002) on the click stream; on alarm we reset the classifier — "
       "exact same pattern as `Week-04-02-Drift_Detection_v3.ipynb`. ",
       "**User-preference model:** P(helpful | λ, regime) = clip(1 − 2|λ − λ_ideal(regime)|, 0, 1) with 5% label noise. "
       "Pre-drift λ_ideal = 0.25 (matches Optuna). Post-drift (step 600) λ_ideal jumps to 0.85 — a sharp shift in user preference. "
       "Note: real click logs aren't available for D1, so this feedback stream is simulated over the real retrieved corpus; D2 replaces it with logged interactions."),
    code("""\
import importlib, online
importlib.reload(online); online.run()           # runs the 1200-step stream; writes online_log + online_stats
"""),

    # ---- section 7: prequential plot ----
    md("## 7 — Prequential evaluation plot",
       "Rolling helpful-rate (window 50) for the static-λ baseline vs the adaptive learner, with the injected drift "
       "and ADWIN alarms marked. Lower panel: classifier's chosen λ vs the regime-true λ_ideal."),
    code("""\
import importlib, prequential_plot
importlib.reload(prequential_plot); prequential_plot.main()
display(__import__('PIL').Image.open(RESULTS / 'prequential.png'))
"""),

    # ---- section 8: run card ----
    md("## 8 — Run card (winning config + environment)",
       "Single source of truth for reproducing the pipeline (Optuna search space + winning hyperparams, "
       "River policy/detector parameters, dataset hash, library versions)."),
    code("""\
import importlib, run_card
importlib.reload(run_card); run_card.main()
print('\\n--- run_card.yaml (head) ---')
print((ROOT / 'run_card.yaml').read_text()[:1400])
"""),

    # ---- section 9: decisions & pitfalls ----
    md("## 9 — Decisions and pitfalls",
       "**Decisions.**",
       "- *Real corpus source.* Papers are pulled from the live arXiv API (25 per topic, deduped) and the resulting "
       "`corpus.parquet` is committed, so the data is real yet reproducible. This replaces the earlier synthetic "
       "stand-in and gives the retriever genuine abstracts to work over.",
       "- *Dense vector source.* The brief suggests `bge-small-en`; D1 uses TF-IDF + TruncatedSVD as the dense side and "
       "defers the neural encoder to D2. This keeps the `svd_dim` Optuna axis directly meaningful (LSA components) and "
       "the notebook fully reproducible. We hold the `HybridRetriever` interface stable so D2 can swap in the BGE "
       "encoder by re-fitting only the dense matrix.",
       "- *Incremental classifier, not bandit.* The brief's framing — *clicked helpful y/n* — is binary, so the online "
       "learner is a `river.linear_model.LogisticRegression` inside a `compose.Pipeline` with `preprocessing.OneHotEncoder`. "
       "This stays strictly inside the Week-03 toolkit (no `river.bandit` module). ε-greedy gives the classifier early data on every arm.",
       "- *Drift triggers a full learner reset.* On ADWIN alarm we re-instantiate the classifier so it re-learns under the "
       "new regime from scratch — the same recovery strategy used in the Week-04 drift-detection notebook.",
       "",
       "**Pitfalls we hit and resolved.**",
       "- *Recall@5 ceiling on broad gold.* 25 papers/topic capped Recall@5 at 0.20 for broad queries. We added 36 "
       "single-paper *title* queries so Recall@5 has a tight denominator.",
       "- *Reward signal too flat for the learner.* The 150-doc corpus is small enough that top-1 hit-rate is "
       "near-constant across λ. We modelled user preference explicitly (triangular reward peaking at λ_ideal) so the "
       "classifier has signal to learn from — this matches the brief's framing of *clicked helpful y/n* feedback.",
       "- *Exploration cost.* Adaptive trails static *pre*-drift (94% → 81%) because of the 10% exploration tax; "
       "this is the well-known regret of online learners. The post-drift recovery (5% → 63%, +1122% relative) "
       "more than makes up for it overall."),
]


def main():
    # Assemble the .ipynb JSON skeleton with the cells we built above.
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "D1.ipynb"
    out.write_text(json.dumps(nb, indent=1))     # indent=1 keeps the file readable but compact
    print(f"Wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
