"""Prequential metrics plot.

Two panels:
  (a) Rolling helpful-rate over the stream: static λ vs adaptive classifier.
      Drift point and ADWIN alarms marked.
  (b) Classifier's chosen λ over time, with the regime-true λ_ideal overlaid.
"""

from __future__ import annotations               # postponed-evaluation type hints
import json                                      # read online_stats.json
from pathlib import Path                         # cross-platform paths

import matplotlib                                # plotting backend
matplotlib.use("Agg")                            # headless backend — no display needed
import matplotlib.pyplot as plt                  # the plotting API
import numpy as np                               # arrays + rolling-mean helper
import pandas as pd                              # parquet I/O + rolling computation

ROOT = Path(__file__).resolve().parents[1]       # D1/
RESULTS = ROOT / "results"                       # input + output directory


def rolling_mean(arr: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling mean over `window` steps; pandas handles edge-of-array gracefully."""
    # min_periods=1 → start emitting values from step 0 instead of after `window` warm-up
    s = pd.Series(arr).rolling(window, min_periods=1).mean()
    return s.to_numpy()


def main():
    # ---- inputs ---------------------------------------------------------
    log = pd.read_parquet(RESULTS / "online_log.parquet")                  # per-step rewards + chosen λ
    stats = json.loads((RESULTS / "online_stats.json").read_text())        # summary including alarm step indices

    n = len(log)                                                           # stream length
    drift_at = stats["drift_at"]                                           # where the concept drift was injected
    alarms = stats["adwin_alarms"]                                         # list of step indices where ADWIN fired
    static_lambda = stats["static_lambda"]                                 # frozen Optuna λ — for the legend

    # Smooth the two reward streams so the eye can compare trends.
    rolling_static = rolling_mean(log["r_static"].to_numpy(), window=50)
    rolling_adaptive = rolling_mean(log["r_adaptive"].to_numpy(), window=50)

    # Two stacked panels — top panel taller (helpful-rate), bottom shorter (chosen λ).
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 2]})

    # ---- (a) helpful-rate ----------------------------------------------
    ax1.plot(rolling_static, color="#777", lw=1.6,
             label=f"Static  (λ={static_lambda:.2f}, Optuna)")
    ax1.plot(rolling_adaptive, color="#0a7", lw=1.8,
             label="Adaptive (River classifier + ADWIN)")
    # Vertical dashed line at the injected drift step
    ax1.axvline(drift_at, color="#c33", ls="--", lw=1.2, alpha=0.8,
                label=f"injected drift @ {drift_at}")
    # Dotted lines at every ADWIN alarm — only the first one gets a legend label
    for i, a in enumerate(alarms):
        ax1.axvline(a, color="#e69500", ls=":", lw=1.0, alpha=0.9,
                    label="ADWIN alarm" if i == 0 else None)
    ax1.set_ylabel("rolling helpful-rate (window=50)")
    ax1.set_title("Prequential evaluation — adaptive hybrid weight under concept drift")
    ax1.set_ylim(-0.02, 1.02)                                              # pin axes to [0,1] + a little headroom
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)

    # ---- (b) chosen lambda ---------------------------------------------
    # Scatter every per-step choice — transparency reveals density
    ax2.scatter(np.arange(n), log["chosen_lambda"].to_numpy(),
                s=3, alpha=0.30, color="#0a7", label="classifier-chosen λ")
    # True ideal λ per step — step function that jumps at drift
    ideal = np.where(np.arange(n) < drift_at, 0.25, 0.85)
    ax2.plot(ideal, color="#c33", lw=1.6, label="λ_ideal (regime)")
    # Rolling mean of the chosen λ → visible learning trajectory
    rolling_lam = rolling_mean(log["chosen_lambda"].to_numpy(), window=50)
    ax2.plot(rolling_lam, color="#024", lw=1.4, alpha=0.8,
             label="classifier λ (rolling mean)")
    # Mirror the drift line + alarm lines on the bottom panel
    ax2.axvline(drift_at, color="#c33", ls="--", lw=1.2, alpha=0.6)
    for a in alarms:
        ax2.axvline(a, color="#e69500", ls=":", lw=1.0, alpha=0.9)
    ax2.set_xlabel("step")
    ax2.set_ylabel("hybrid λ")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="center right", fontsize=9)

    fig.tight_layout()                                                     # avoid overlapping labels
    out = RESULTS / "prequential.png"
    fig.savefig(out, dpi=150)                                              # 150 DPI keeps the PNG readable when embedded
    plt.close(fig)                                                         # release memory
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
