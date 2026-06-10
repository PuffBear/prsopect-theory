"""
Set the MPE collapse threshold (theta) from the calibration runs.

Reads docs/exp_mpe/mpe_calibration_raw_data.csv (from exp_mpe_calibrate.py: trained
PPO at w=0.0 and w=1.0), computes the true-reward distribution for each extreme,
places theta between the "collapsed" (w=0, worst) and "coordinated" (w=1, best)
regimes, and writes docs/exp_mpe/mpe_collapse_threshold.json (read by
env/mpe_coop_nav.is_collapsed_mpe and the sweep analysis).

theta rule (pre-registered): midpoint of the two means,
    theta = (mean_true_reward[w=0] + mean_true_reward[w=1]) / 2
with an explicit separation/overlap check. If the two regimes overlap (so no clean
threshold exists), that is reported -- it would mean MPE does not show the collapse
transition cleanly, which is itself a result.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

CSV = "docs/exp_mpe/mpe_calibration_raw_data.csv"
OUT_JSON = "docs/exp_mpe/mpe_collapse_threshold.json"


def main():
    if not os.path.exists(CSV):
        print(f"No calibration data at {CSV} -- run exp_mpe_calibrate.py (40 runs) first.")
        sys.exit(0)
    df = pd.read_csv(CSV)
    g = df.groupby("w")["mean_reward"]
    stats = g.agg(["count", "mean", "std", "min", "max"])
    print("Calibration true-reward distribution by w:\n")
    print(stats.round(2).to_string())

    if not {0.0, 1.0}.issubset(set(df["w"].unique())):
        print("\nNeed both w=0.0 and w=1.0 present. Found:", sorted(df["w"].unique()))
        sys.exit(1)

    m0, m1 = stats.loc[0.0, "mean"], stats.loc[1.0, "mean"]
    theta = (m0 + m1) / 2.0

    # separation check: does the worst-case (w=1 min) sit above the best-case (w=0 max)?
    w0_max, w1_min = stats.loc[0.0, "max"], stats.loc[1.0, "max"]  # noqa
    w1_lo = stats.loc[1.0, "min"]
    clean = w1_lo > stats.loc[0.0, "max"]   # all w=1 runs beat all w=0 runs
    overlap = stats.loc[0.0, "max"] - stats.loc[1.0, "min"]

    print(f"\n  mean(true_reward | w=0, worst) = {m0:.2f}")
    print(f"  mean(true_reward | w=1, best ) = {m1:.2f}")
    print(f"  -> theta (midpoint)            = {theta:.2f}")
    print(f"  clean separation (all w=1 > all w=0): {clean}"
          + ("" if clean else f"  [overlap region = {overlap:.2f}; threshold is fuzzy]"))

    payload = {
        "threshold": round(float(theta), 3),
        "w0_mean": round(float(m0), 3), "w1_mean": round(float(m1), 3),
        "w0_max": round(float(stats.loc[0.0, "max"]), 3),
        "w1_min": round(float(stats.loc[1.0, "min"]), 3),
        "clean_separation": bool(clean),
        "n_seeds_per_w": int(stats.loc[0.0, "count"]),
        "rule": "midpoint of mean(true_reward) at w=0 (worst) and w=1 (best)",
        "source_csv": CSV,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT_JSON}:  threshold = {payload['threshold']}")
    print("  (env/mpe_coop_nav.is_collapsed_mpe and the sweep analysis read this file.)")


if __name__ == "__main__":
    main()
