# D1 — Streaming Learner & AutoML Note

CSAI415 · *PDF-Papers AI Agent: Hybrid Retrieval + GraphRAG with Online Learning and AutoML*

**Team:** Khalifa · Meshal · Mahmoud · Ahmed · Essam

This folder is the complete D1 deliverable: a hybrid BM25 + dense retriever, an Optuna AutoML search (Track A), and a River incremental classifier (logistic regression over λ-buckets, with ε-greedy action selection and ADWIN drift handling) — evaluated prequentially over a 1 200-step feedback stream.

## Files

```
D1/
├── D1.ipynb              ← runnable notebook (already executed; outputs included)
├── run_card.yaml         ← winning AutoML config + River params + env versions + dataset hash
├── README.md             ← this file
├── build_notebook.py     ← regenerates D1.ipynb from src/ modules
├── build_report.js       ← regenerates ../D1_Report.docx (the 2-page report, in the parent folder)
├── data/
│   ├── corpus.parquet    ← 150 real arXiv papers across 6 topics (fetched via the arXiv API)
│   ├── corpus_hash.txt   ← stable sha256-prefix of the corpus
│   └── gold.parquet      ← 78-query gold set (broad / targeted / title)
├── results/
│   ├── baselines.json
│   ├── automl_study.json
│   ├── optuna_history.png
│   ├── online_log.parquet
│   ├── online_stats.json
│   └── prequential.png   ← embedded in the report; rolling helpful-rate + chosen λ
└── src/
    ├── build_corpus.py   ← procedural corpus generator
    ├── retriever.py      ← HybridRetriever (BM25 + TF-IDF/SVD)
    ├── gold.py           ← gold-set builder + Recall@k / NDCG@k
    ├── baselines.py
    ├── automl.py         ← Optuna study (50 trials, TPE, NDCG@5 objective)
    ├── online.py         ← River incremental classifier (LogisticRegression) + ε-greedy + ADWIN
    ├── prequential_plot.py
    └── run_card.py
```

The 2-page report `D1_Report.docx` lives one level up (next to the project brief) for easy submission.

## How to reproduce

```bash
# from this folder
pip install --break-system-packages \
    rank_bm25 optuna river scikit-learn pyarrow pyyaml matplotlib \
    nbformat nbclient pillow arxiv ipykernel

# rebuild every artefact end-to-end
PYTHONPATH=src python3 src/build_corpus.py
PYTHONPATH=src python3 src/gold.py
PYTHONPATH=src python3 src/baselines.py
PYTHONPATH=src python3 src/automl.py
PYTHONPATH=src python3 src/online.py
PYTHONPATH=src python3 src/prequential_plot.py
PYTHONPATH=src python3 src/run_card.py

# regenerate the notebook (executed in-place by nbclient)
python3 build_notebook.py
PYTHONPATH=src python3 -c "import nbformat; from nbclient import NotebookClient; nb=nbformat.read('D1.ipynb',4); NotebookClient(nb,timeout=120,kernel_name='python3').execute(); nbformat.write(nb,'D1.ipynb')"

# regenerate the report (requires Node.js and: npm install -g docx)
NODE_PATH="$HOME/.npm-global/lib/node_modules" node build_report.js
```

A single seed (`42`) is used across the learning components: gold split, Optuna sampler, the incremental classifier, ADWIN. The arXiv corpus itself is pinned by committing `corpus.parquet` (the API fetch is reproducible from the committed snapshot).

## Headline results

| Metric | Best baseline (naive 0.5) | AutoML winner (full gold) |
|---|---:|---:|
| Recall@5 | 0.580 | **0.584** |
| NDCG@5  | 0.895 | **0.896** |
| p95 latency | 0.68 ms | 0.41 ms |

Online learner (1 200-step stream, drift at step 600):

| Slice | Static (frozen Optuna λ) | Adaptive (River classifier + ADWIN) | Relative lift |
|---|---:|---:|---:|
| Pre-drift  | 0.68 | 0.81 | (exploration tax) |
| Post-drift | 0.13 | 0.63 | **+399 %** |
| Overall    | 0.40 | 0.72 | **+80.2 %** |

ADWIN fires at step **639** (~ 39 steps after the injected drift) and resets the classifier, which re-learns under the new regime.

## Notes

- **Dense embedder.** D1 uses TF-IDF + TruncatedSVD instead of `bge-small-en` (the neural encoder is deferred to D2). The `HybridRetriever` interface holds stable so D2 swaps the dense matrix in without touching downstream code.
- **Reward model.** Click-helpfulness is modelled as `clip(1 − 2|λ − λ_ideal|, 0, 1)` with 5 % label flip; the regime-true `λ_ideal` jumps from 0.25 to 0.85 at step 600. This makes the static-vs-adaptive contrast visible despite the small corpus where retrieval-driven feedback would be near-flat across λ.
- **Targets.** p95 latency ≤ 2 s ✓ · online lift > +5 % vs static on a temporal slice ✓ (post-drift +399 %). Recall@5 = 0.58 on the real arXiv corpus — just under the 0.60 target; closing that gap (stronger dense encoder, larger corpus) is a D2 item.
- **In-scope toolkit.** Optuna with TPE sampler (Week 02), `river.compose / preprocessing / linear_model / drift` (Weeks 03–04), `rank_bm25` (Week 05 RAG notebook), TF-IDF + TruncatedSVD (sklearn). No `river.bandit`, MongoDB, Qdrant, PEFT/QLoRA, FastAPI — those belong to D2–D4.

## Team & contributions

| Member  | Owns                              | Primary files                                                  |
|---------|-----------------------------------|----------------------------------------------------------------|
| Khalifa | AutoML track (Optuna)             | `src/automl.py`, `src/baselines.py`                            |
| Meshal  | Online learning & drift (River)   | `src/online.py`, `src/prequential_plot.py`                     |
| Mahmoud | Hybrid retriever                  | `src/retriever.py`                                             |
| Ahmed   | Corpus & gold set + ranking metrics | `src/build_corpus.py`, `src/gold.py`                         |
| Essam   | Report, notebook, run card        | `build_notebook.py`, `build_report.js`, `src/run_card.py`, `README.md` |

**How we worked.** The table above lists primary ownership. All five members reviewed each other's PRs, contributed commits to integration fixes, and were present for the AutoML/online-learning design discussions. A single seed (`42`) is shared across all modules so any member can reproduce any other member's results bit-for-bit.

### Per-member details

- **Khalifa — AutoML track.** Defined the search space (`k ∈ [3, 30]`, `metric ∈ {cosine, dot, euclidean}`, `svd_dim ∈ {0, 64, 128, 256}`, `l2_normalize ∈ {T, F}`, `hybrid_lambda ∈ [0, 1]`), the latency-penalised NDCG@5 objective, the 60/40 train/val split, ran the 50-trial TPE study, produced `automl_study.json` and `optuna_history.png`. Owns the *AutoML design* rubric (6 %).
- **Meshal — Online learning & drift.** Built the River incremental classifier (`compose.Pipeline(OneHotEncoder, LogisticRegression)`), ε-greedy action selection over 11 λ-buckets, ADWIN drift detector with classifier reset, the 1 200-step simulated feedback stream, and the two-panel prequential plot. Owns the *Online learning* rubric (6 %).
- **Mahmoud — Hybrid retriever.** Implemented `HybridRetriever`: BM25Okapi on the lexical side, TF-IDF + optional TruncatedSVD on the dense side, the cosine/dot/euclidean metric switch, optional L2 normalisation, min-max score fusion, batched search with per-query latency timing. Exposed every Optuna axis as a parameter on `HybridConfig`.
- **Ahmed — Corpus & gold set.** Built the 150-paper real arXiv corpus (25 papers/topic fetched via the arXiv API across six topic clusters); built the 78-query gold set (4 broad + 3 targeted + 6 title per topic); implemented `recall_at_k` and `ndcg_at_k` from scratch and the per-retriever `evaluate` helper.
- **Essam — Report, notebook & run card.** Assembled `D1.ipynb` from the `src/` modules; wrote the 2-page DOCX report with the embedded prequential figure and the baseline-vs-AutoML / winning-config / online-results tables; produced `run_card.yaml` capturing the winning configuration, environment versions, dataset hash, and ADWIN alarms; wrote this README and the reproducibility instructions. Owns the *Report quality* rubric (3 %).
