"""River incremental classifier for adaptive hybrid-weight learning, with ADWIN drift handling.

Setup
-----
We model the live system as a stream of (query, click-was-helpful) feedback.
The retriever runs at the chosen `hybrid_lambda` for every step; the one knob
we adapt online is that lambda.

Online learner (Week-03 scope)
------------------------------
We discretise lambda into 11 buckets in {0.0, 0.1, …, 1.0} and learn a single
incremental classifier that predicts the probability of a *helpful* click for
each bucket:

    features = {"lambda_bucket": "<arm_id>"}       # one categorical feature
    target   = helpful ∈ {True, False}              # binary click outcome
    model    = compose.Pipeline(
                   preprocessing.OneHotEncoder(),
                   linear_model.LogisticRegression()
               )

At each step we use ε-greedy action selection on top of the classifier:
explore (random arm) with probability ε, otherwise exploit the bucket with the
highest predicted P(helpful = True). This stays inside the Week-03 toolkit
(`river.compose`, `river.preprocessing`, `river.linear_model`) — no bandit
module is used.

User-preference model
---------------------
Real users prefer different fusion weights at different times: maybe today the
corpus drifts toward keyword-heavy queries (BM25 wins), maybe tomorrow toward
semantic paraphrases (dense wins). We simulate this with a triangular reward
function that peaks at an *ideal* lambda for the current regime:

    P(helpful) = clip( 1 - 2 * |lambda - lambda_ideal(regime)| , 0, 1 )

Pre-drift  (steps   0 –  599):  lambda_ideal = 0.25  (matches Optuna's pick)
Post-drift (steps 600 – 1199):  lambda_ideal = 0.85  (sudden shift to lexical)

ADWIN watches the click stream; when it fires we reset the classifier so it
re-learns under the new regime — same pattern as Week-04-02-Drift_Detection_v3.

Outputs
-------
results/online_log.parquet  — per-step records (lambda chosen, reward, etc.)
results/online_stats.json   — summary stats and ADWIN alarm step indices.
The prequential plot is produced by src/prequential_plot.py.
"""

from __future__ import annotations               # postponed-evaluation type hints
import json                                      # JSON I/O for the summary stats
import random                                    # seeded RNG for the stream + reward noise
from pathlib import Path                         # cross-platform paths

import numpy as np                               # arms grid + array math
import pandas as pd                              # log dataframe + parquet I/O
# Week-03 primitives only — no river.bandit here.
from river import compose, drift, linear_model, preprocessing, stats

from retriever import HybridConfig, HybridRetriever   # the retriever

ROOT = Path(__file__).resolve().parents[1]       # D1/
DATA = ROOT / "data"                             # input dir
RESULTS = ROOT / "results"                       # output dir
RESULTS.mkdir(parents=True, exist_ok=True)       # ensure results/ exists

SEED = 42                                        # global seed
N_STEPS = 1200                                   # length of the simulated feedback stream
DRIFT_AT = 600                                   # synthetic concept drift injected here
ARMS = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]  # 11 lambda values evenly spaced in [0,1]
EPSILON = 0.10                                   # exploration rate on top of the classifier
ADWIN_DELTA = 0.002                              # ADWIN confidence parameter (matches Week-04 default)

# Ideal hybrid_lambda per regime. Pre-drift matches Optuna's pick; post-drift
# shifts to a heavily-lexical preference.
IDEAL_LAMBDA_PRE = 0.25
IDEAL_LAMBDA_POST = 0.85
NOISE_FLIP = 0.05                                # 5 % label flip on each click


def make_stream(gold: pd.DataFrame, n_steps: int, drift_at: int, seed: int):
    """Yield (step, qid, query, relevant, regime, lambda_ideal) tuples."""
    rng = random.Random(seed)                    # local seeded RNG for the stream
    rows = gold.to_dict("records")               # list of dicts for O(1) random.choice

    for step in range(n_steps):
        regime = "pre" if step < drift_at else "post"                      # which side of the drift point
        lambda_ideal = IDEAL_LAMBDA_PRE if regime == "pre" else IDEAL_LAMBDA_POST
        row = rng.choice(rows)                                             # sample a query uniformly
        yield (step, row["query_id"], row["query"], list(row["relevant"]),
               regime, lambda_ideal)


def click_reward(chosen_lambda: float, lambda_ideal: float, rng) -> float:
    """Simulated 'clicked helpful y/n'.

    Probability of helpful = clip(1 - 2|λ - λ_ideal|, 0, 1).
    Then we flip the label with NOISE_FLIP probability.
    Returns 0.0 or 1.0.
    """
    p = max(0.0, 1.0 - 2.0 * abs(chosen_lambda - lambda_ideal))            # triangular peak at λ_ideal
    helpful = 1.0 if rng.random() < p else 0.0                             # Bernoulli(p)
    if rng.random() < NOISE_FLIP:                                          # 5 % label noise → flip outcome
        helpful = 1.0 - helpful
    return helpful


def _new_classifier():
    """Factory for a fresh incremental classifier.

    A `compose.Pipeline` stitches together a OneHotEncoder (categorical → sparse
    binary features) and a LogisticRegression (incremental). Both pieces are
    the exact `river.*` modules imported in `Week-03-00-Introduction.ipynb`.
    """
    return compose.Pipeline(
        preprocessing.OneHotEncoder(),                                     # bucket-id → one-hot
        linear_model.LogisticRegression(),                                 # incremental logistic regression
    )


def _choose_arm(clf, rng: random.Random) -> int:
    """ε-greedy action selection on top of the classifier."""
    # Explore: pick a random arm with probability ε. This guarantees every arm
    # eventually gets some data (otherwise an under-sampled arm could never have
    # its predicted P(helpful) updated).
    if rng.random() < EPSILON:
        return rng.randrange(len(ARMS))
    # Exploit: predict P(helpful=True) for each arm, pick the argmax.
    scores = []
    for arm_id in range(len(ARMS)):
        x = {"lambda_bucket": str(arm_id)}                                 # one categorical feature
        proba = clf.predict_proba_one(x)                                   # {True: p, False: 1-p}
        # Before any learn_one, predict_proba_one returns an empty dict — handle that gracefully
        scores.append(proba.get(True, 0.5))
    return int(np.argmax(scores))


def run() -> dict:
    # ---- load inputs ----------------------------------------------------
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    gold = pd.read_parquet(DATA / "gold.parquet")
    automl = json.loads((RESULTS / "automl_study.json").read_text())       # so we can read the static λ
    p = automl["best_params"]
    static_lambda = float(p["hybrid_lambda"])                              # frozen Optuna λ — the baseline

    # Fit the retriever once with the AutoML-winning structural axes.
    cfg = HybridConfig(k=p["k"], metric=p["metric"], svd_dim=p["svd_dim"],
                       l2_normalize=p["l2_normalize"], hybrid_lambda=static_lambda,
                       embedder="bge")           # D2: semantic dense side (bge-small-en) instead of TF-IDF/SVD
    retriever = HybridRetriever(cfg).fit(corpus)

    rng = random.Random(SEED)                                              # one RNG for exploration + reward noise

    # ---- River components ----------------------------------------------
    clf = _new_classifier()                                                # incremental classifier (Week-03)
    adwin = drift.ADWIN(delta=ADWIN_DELTA)                                 # drift detector on the click stream

    # ---- bookkeeping ---------------------------------------------------
    log = []                                                               # per-step records
    adwin_alarms: list[int] = []                                           # step indices where ADWIN fired
    classifier_resets: list[int] = []                                      # step indices where we re-instantiated

    static_mean = stats.Mean()                                             # running mean of static reward (Week-03 stats)
    adaptive_mean = stats.Mean()                                           # running mean of adaptive reward

    # ---- main loop -----------------------------------------------------
    for step, qid, query, relevant, regime, lambda_ideal in make_stream(
            gold, N_STEPS, DRIFT_AT, SEED):
        # ---- adaptive: classifier picks the best λ-bucket (ε-greedy) -------
        arm_id = _choose_arm(clf, rng)                                     # ε-greedy on top of P(helpful | bucket)
        chosen_lambda = ARMS[arm_id]                                       # translate arm index → λ value
        retriever.search(query, top_k=5, hybrid_lambda=chosen_lambda)      # actually do the retrieval (latency realism)
        r_adaptive = click_reward(chosen_lambda, lambda_ideal, rng)        # simulate user click (0/1)

        # ---- static (Optuna lambda, frozen) --------------------------------
        retriever.search(query, top_k=5, hybrid_lambda=static_lambda)      # also actually retrieve for fairness
        r_static = click_reward(static_lambda, lambda_ideal, rng)          # same reward model, fixed λ

        # ---- update classifier + ADWIN on the adaptive reward stream -------
        clf.learn_one({"lambda_bucket": str(arm_id)}, bool(r_adaptive))    # incremental fit
        adwin.update(r_adaptive)                                           # ADWIN watches the click stream

        if adwin.drift_detected:                                           # ADWIN flagged a change-point
            adwin_alarms.append(step)
            clf = _new_classifier()                                        # full reset → re-learn under new regime
            classifier_resets.append(step)

        static_mean.update(r_static)                                       # update running means
        adaptive_mean.update(r_adaptive)

        log.append({                                                       # one row per step
            "step": step, "qid": qid, "regime": regime,
            "lambda_ideal": lambda_ideal,
            "chosen_lambda": chosen_lambda,
            "r_adaptive": r_adaptive, "r_static": r_static,
            "static_mean": static_mean.get(),
            "adaptive_mean": adaptive_mean.get(),
        })

    # ---- summary stats --------------------------------------------------
    df = pd.DataFrame(log)
    pre = df[df["regime"] == "pre"]
    post = df[df["regime"] == "post"]
    summary = {
        "n_steps": N_STEPS,
        "drift_at": DRIFT_AT,
        "epsilon": EPSILON,
        "adwin_delta": ADWIN_DELTA,
        "arms": ARMS,
        "static_lambda": static_lambda,
        "learner": "river.compose.Pipeline(OneHotEncoder + LogisticRegression) + ε-greedy",
        "overall": {
            "static_helpful_rate": float(df["r_static"].mean()),
            "adaptive_helpful_rate": float(df["r_adaptive"].mean()),
            "relative_lift_pct": float(                                    # (adaptive / static - 1) * 100
                (df["r_adaptive"].mean() / df["r_static"].mean() - 1.0) * 100
                if df["r_static"].mean() > 0 else 0.0),
        },
        "pre_drift": {                                                     # slice metrics on the pre side
            "static": float(pre["r_static"].mean()),
            "adaptive": float(pre["r_adaptive"].mean()),
        },
        "post_drift": {                                                    # the "after the drift" slice
            "static": float(post["r_static"].mean()),
            "adaptive": float(post["r_adaptive"].mean()),
            "relative_lift_pct": float(
                (post["r_adaptive"].mean() / post["r_static"].mean() - 1.0) * 100
                if post["r_static"].mean() > 0 else 0.0),
        },
        "adwin_alarms": adwin_alarms,
        "n_classifier_resets": len(classifier_resets),
        # Distribution of λ chosen by the classifier, broken down by regime — a sanity check
        "lambda_distribution_pre": (
            df.loc[df["regime"] == "pre", "chosen_lambda"]
              .value_counts(normalize=True).round(3).to_dict()),
        "lambda_distribution_post": (
            df.loc[df["regime"] == "post", "chosen_lambda"]
              .value_counts(normalize=True).round(3).to_dict()),
    }

    # Persist per-step log + summary for the plotting script and the report.
    df.to_parquet(RESULTS / "online_log.parquet", index=False)
    (RESULTS / "online_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS / 'online_log.parquet'} and online_stats.json")
    return summary


if __name__ == "__main__":
    run()
