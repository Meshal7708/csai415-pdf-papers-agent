"""Build run_card.yaml for the winning AutoML config + River component.

The run card is a single source of truth for reproducing D1's pipeline.
"""

from __future__ import annotations               # postponed-evaluation type hints
import hashlib                                   # not used directly; kept for future extension
import json                                      # read result JSON files
import platform                                  # capture OS / kernel info
import sys                                       # capture Python version
from datetime import datetime, timezone          # timestamp the run card
from pathlib import Path                         # cross-platform paths

import yaml                                      # YAML output (human-readable run card)

ROOT = Path(__file__).resolve().parents[1]       # D1/
DATA = ROOT / "data"                             # input dir (corpus hash etc.)
RESULTS = ROOT / "results"                       # input dir (study + stats JSON)


def _versions() -> dict:
    """Collect installed-package versions for reproducibility."""
    import importlib
    out = {}
    # The exact packages D1 depends on; record their installed versions
    for mod in ["numpy", "pandas", "scikit-learn", "rank_bm25", "optuna",
                "river", "matplotlib", "pyarrow"]:
        try:
            # scikit-learn imports as `sklearn`; the import name & PyPI name differ
            m = importlib.import_module(mod.replace("-", "_"))
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "not-installed"           # don't crash if something is missing
    out["python"] = sys.version.split()[0]       # "3.10.12" etc.
    out["platform"] = platform.platform()        # OS/kernel string
    return out


def main():
    # ---- load all the upstream artefacts --------------------------------
    automl = json.loads((RESULTS / "automl_study.json").read_text())
    online = json.loads((RESULTS / "online_stats.json").read_text())
    base = json.loads((RESULTS / "baselines.json").read_text())
    corpus_hash = (DATA / "corpus_hash.txt").read_text().strip()

    # ---- assemble the run-card dict ------------------------------------
    card = {
        "run_id": f"D1-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",   # unique per run
        "deliverable": "D1 — Streaming Learner & AutoML Note",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": 42,
        "team": {                                                          # primary ownership per member
            "Khalifa": "AutoML track (Optuna): src/automl.py, src/baselines.py",
            "Meshal":  "Online learning & drift (River): src/online.py, src/prequential_plot.py",
            "Mahmoud": "Hybrid retriever: src/retriever.py",
            "Ahmed":   "Corpus & gold set + ranking metrics: src/build_corpus.py, src/gold.py",
            "Essam":   "Report, notebook, run card: build_notebook.py, build_report.js, src/run_card.py, README.md",
        },
        "dataset": {
            "name": "arxiv-api-150",
            "n_papers": 150,
            "topics": 6,
            "corpus_hash": corpus_hash,                                    # fingerprint of the actual corpus
            "gold_queries": 78,
        },
        "automl": {
            "framework": "Optuna",
            "n_trials": automl["n_trials"],
            "search_space": automl["search_space"],                        # exactly what was searched
            "objective": automl["objective"],                              # human-readable objective string
            "winning_config": automl["best_params"],                       # the hyperparams to deploy
            "metrics": {                                                   # side-by-side baseline vs AutoML
                "baseline_naive_hybrid": base["baselines"]["naive_hybrid_0.5"],
                "best_on_train_split": automl["metrics"]["train"],
                "best_on_val_split":   automl["metrics"]["val"],
                "best_on_full_gold":   automl["metrics"]["full"],
            },
        },
        "online_learning": {
            "framework": "River",
            "task": "adaptive hybrid weight from feedback (clicked-helpful y/n)",
            "policy": {
                "kind": online["learner"],                                 # classifier pipeline + ε-greedy
                "epsilon": online["epsilon"],
                "arms_lambda": online["arms"],                             # the 11 discrete λ values
            },
            "drift_detector": {
                "kind": "ADWIN",
                "delta": online["adwin_delta"],
                "alarms_at_steps": online["adwin_alarms"],                 # where ADWIN fired
                "n_classifier_resets": online["n_classifier_resets"],
            },
            "stream": {                                                    # how the synthetic drift was constructed
                "n_steps": online["n_steps"],
                "drift_at": online["drift_at"],
                "lambda_ideal_pre": 0.25,
                "lambda_ideal_post": 0.85,
                "noise_flip_rate": 0.05,
            },
            "results": {                                                   # headline numbers for the report
                "static_helpful_rate_overall": online["overall"]["static_helpful_rate"],
                "adaptive_helpful_rate_overall": online["overall"]["adaptive_helpful_rate"],
                "relative_lift_overall_pct": online["overall"]["relative_lift_pct"],
                "post_drift_static": online["post_drift"]["static"],
                "post_drift_adaptive": online["post_drift"]["adaptive"],
                "post_drift_relative_lift_pct": online["post_drift"]["relative_lift_pct"],
            },
        },
        "artifacts": {                                                     # pointer to every output file
            "notebook": "D1.ipynb",
            "report": "D1_Report.docx",
            "prequential_plot": "results/prequential.png",
            "optuna_history_plot": "results/optuna_history.png",
            "baselines_json": "results/baselines.json",
            "automl_study_json": "results/automl_study.json",
            "online_log_parquet": "results/online_log.parquet",
        },
        "environment": _versions(),                                        # library + python + OS versions
    }

    out = ROOT / "run_card.yaml"
    # sort_keys=False → preserve the human-readable ordering we built above
    out.write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))
    print(f"Wrote {out}")
    print("---")
    print(out.read_text()[:1200])                                          # show a preview in the CLI log


if __name__ == "__main__":
    main()
