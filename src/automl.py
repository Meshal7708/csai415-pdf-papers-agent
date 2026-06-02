"""Optuna AutoML search for the kNN retriever (D1 Track A).

Search space (from the project brief):
    k             : int    [3, 30]
    metric        : enum   {cosine, dot, euclidean}
    svd_dim       : enum   {0, 64, 128, 256}
    l2_normalize  : bool
    hybrid_lambda : float  [0.0, 1.0]

Objective: maximize NDCG@5 with a soft latency penalty.
    score = NDCG@5 - 0.05 * max(0, p95_ms - 50)
We also report Recall@5 for the final winning config.

We split the gold set into a train/val split (60/40) so Optuna optimizes on
a held-out set and we report final metrics on the full gold set with the
chosen config — this is the AutoML pattern from the FLAML / HPO tutorials.

Output: results/automl_study.json + results/optuna_history.png
"""

from __future__ import annotations               # postponed-evaluation type hints
import json                                      # dump the study summary as JSON
from pathlib import Path                         # cross-platform paths

import numpy as np                               # array math + percentile + permutation
import optuna                                    # the AutoML framework
import pandas as pd                              # corpus / gold I/O

from retriever import HybridConfig, HybridRetriever   # candidate retriever under tuning
from gold import evaluate                             # objective metric helper

ROOT = Path(__file__).resolve().parents[1]       # D1/
DATA = ROOT / "data"                             # input dir
RESULTS = ROOT / "results"                       # output dir
RESULTS.mkdir(parents=True, exist_ok=True)       # ensure results/ exists

SEED = 42                                        # global seed — same one used everywhere in D1
N_TRIALS = 50                                    # how many configurations Optuna evaluates


def split_gold(gold: pd.DataFrame, frac: float = 0.6, seed: int = SEED):
    """Deterministic train/val split of the gold set (default 60/40)."""
    rng = np.random.default_rng(seed)            # numpy's modern, seedable RNG
    perm = rng.permutation(len(gold))            # random permutation of row indices
    cut = int(len(gold) * frac)                  # split index
    # iloc → integer-location indexing; reset_index keeps row ids contiguous
    return gold.iloc[perm[:cut]].reset_index(drop=True), gold.iloc[perm[cut:]].reset_index(drop=True)


def make_objective(corpus: pd.DataFrame, gold_train: pd.DataFrame,
                   gold_val: pd.DataFrame):
    """Build the Optuna objective function.

    A retriever has to be REFIT whenever {svd_dim, l2, metric} changes — those
    axes change the dense matrix. But k and hybrid_lambda are applied at
    search time, so they don't need a refit. We cache fitted retrievers by
    their structural axes to keep the 50-trial study fast.
    """
    cache: dict[tuple, HybridRetriever] = {}     # (svd_dim, l2, metric) → fitted retriever

    def _retriever_for(svd_dim: int, l2: bool, metric: str) -> HybridRetriever:
        key = (svd_dim, l2, metric)              # structural key
        if key not in cache:
            cfg = HybridConfig(k=10, metric=metric, svd_dim=svd_dim,
                               l2_normalize=l2, hybrid_lambda=0.5,
                               embedder="bge")        # D2: semantic dense side; k & λ don't matter here
            cache[key] = HybridRetriever(cfg).fit(corpus)                 # one fit per unique key
        return cache[key]

    def objective(trial: optuna.Trial) -> float:
        # Each suggest_* call tells Optuna "this is a tunable axis with this distribution"
        k = trial.suggest_int("k", 3, 30)                                  # int axis [3, 30]
        metric = trial.suggest_categorical("metric", ["cosine", "dot", "euclidean"])
        svd_dim = trial.suggest_categorical("svd_dim", [0, 64, 128, 256])
        l2 = trial.suggest_categorical("l2_normalize", [True, False])
        lam = trial.suggest_float("hybrid_lambda", 0.0, 1.0)               # continuous axis [0, 1]

        retriever = _retriever_for(svd_dim, l2, metric)                    # cached fit
        # k is applied at search time via top_k_search — no refit needed.
        m = evaluate(retriever, gold_train, k=5,
                     top_k_search=max(k, 5), hybrid_lambda=lam)
        # Objective: maximise NDCG@5; penalise p95 latency above 50ms (soft constraint).
        score = m["ndcg@5"] - 0.05 * max(0.0, m["p95_latency_ms"] - 50.0)
        # Stash useful info on the trial for later inspection (study JSON).
        trial.set_user_attr("recall@5", m["recall@5"])
        trial.set_user_attr("ndcg@5", m["ndcg@5"])
        trial.set_user_attr("p95_ms", m["p95_latency_ms"])
        return score                                                       # Optuna maximises this

    return objective, cache


def main():
    # Load the two persisted artefacts
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    gold = pd.read_parquet(DATA / "gold.parquet")
    gold_train, gold_val = split_gold(gold, frac=0.6, seed=SEED)           # 60% to tune on, 40% held out
    print(f"Gold split: train={len(gold_train)}  val={len(gold_val)}")

    # TPE = Tree-structured Parzen Estimator — Bayesian Optimization for mixed search spaces.
    sampler = optuna.samplers.TPESampler(seed=SEED)
    optuna.logging.set_verbosity(optuna.logging.WARNING)                   # quiet down Optuna's per-trial logs
    study = optuna.create_study(direction="maximize", sampler=sampler)     # we want highest score
    objective, cache = make_objective(corpus, gold_train, gold_val)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)  # run the search

    best = study.best_trial                                                # highest-scoring trial
    print("\n=== Best trial (on train split) ===")
    print(f"  score    = {best.value:.4f}")
    print(f"  params   = {best.params}")
    print(f"  recall@5 = {best.user_attrs['recall@5']:.3f}")
    print(f"  ndcg@5   = {best.user_attrs['ndcg@5']:.3f}")
    print(f"  p95_ms   = {best.user_attrs['p95_ms']:.2f}")

    # Re-evaluate the winning config on the held-out val split, then on the full gold set,
    # to detect optimistic bias in the training-split score.
    p = best.params
    cfg = HybridConfig(k=p["k"], metric=p["metric"], svd_dim=p["svd_dim"],
                       l2_normalize=p["l2_normalize"],
                       hybrid_lambda=p["hybrid_lambda"],
                       embedder="bge")           # D2: semantic dense side (bge-small-en)
    retriever = HybridRetriever(cfg).fit(corpus)
    m_val = evaluate(retriever, gold_val, k=5, top_k_search=max(cfg.k, 5))
    m_full = evaluate(retriever, gold, k=5, top_k_search=max(cfg.k, 5))
    print(f"\n  val  : recall@5={m_val['recall@5']:.3f}  "
          f"ndcg@5={m_val['ndcg@5']:.3f}  p95_ms={m_val['p95_latency_ms']:.2f}")
    print(f"  full : recall@5={m_full['recall@5']:.3f}  "
          f"ndcg@5={m_full['ndcg@5']:.3f}  p95_ms={m_full['p95_latency_ms']:.2f}")

    # Build the JSON summary — everything the run-card and report need.
    study_summary = {
        "n_trials": N_TRIALS,
        "seed": SEED,
        "search_space": {                                                  # exactly mirrors the brief's spec
            "k": [3, 30],
            "metric": ["cosine", "dot", "euclidean"],
            "svd_dim": [0, 64, 128, 256],
            "l2_normalize": [True, False],
            "hybrid_lambda": [0.0, 1.0],
        },
        "objective": "NDCG@5 - 0.05 * max(0, p95_ms - 50)",                # human-readable objective string
        "best_params": best.params,
        "best_score": float(best.value),
        "metrics": {
            "train": {                                                     # what Optuna saw
                "recall@5": float(best.user_attrs["recall@5"]),
                "ndcg@5": float(best.user_attrs["ndcg@5"]),
                "p95_ms": float(best.user_attrs["p95_ms"]),
            },
            "val": m_val,                                                  # held-out check
            "full": m_full,                                                # final reportable numbers
        },
        "trials": [                                                        # every trial — for plotting / debugging
            {"number": t.number, "value": t.value, "params": t.params,
             **{k: v for k, v in t.user_attrs.items()}}
            for t in study.trials
        ],
    }
    (RESULTS / "automl_study.json").write_text(json.dumps(study_summary, indent=2))
    print(f"\nWrote {RESULTS / 'automl_study.json'}")

    # Optuna history plot — running-best curve over trials.
    try:
        import matplotlib
        matplotlib.use("Agg")                                              # headless backend (no display)
        import matplotlib.pyplot as plt
        values = [t.value for t in study.trials if t.value is not None]    # drop any failed trials
        running_best = np.maximum.accumulate(values)                       # cumulative max — the headline curve
        fig, ax = plt.subplots(figsize=(7, 3.5))                           # short wide figure
        ax.plot(values, "o", color="#888", alpha=0.6, label="trial score") # individual trial scores
        ax.plot(running_best, color="#0a7", lw=2, label="running best")    # the only line that matters most
        ax.set_xlabel("trial")
        ax.set_ylabel("NDCG@5 (latency-penalised)")
        ax.set_title("Optuna AutoML search — kNN retriever")
        ax.legend()
        ax.grid(alpha=0.3)                                                 # subtle grid
        fig.tight_layout()
        fig.savefig(RESULTS / "optuna_history.png", dpi=140)
        plt.close(fig)                                                     # release memory
        print(f"Wrote {RESULTS / 'optuna_history.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")                                      # don't kill the run if plotting fails

    return study_summary


if __name__ == "__main__":
    main()
