"""
Set the MPE collapse threshold (v2, OCCUPANCY-based) from the calibration runs.

Reads docs/exp_mpe/mpe_calibration_v2_raw_data.csv (from exp_mpe_calibrate.py:
trained PPO at w=0.0 and w=1.0, null condition), computes the occupancy
distribution at each extreme, places theta_occ at the midpoint, and writes
docs/exp_mpe/mpe_collapse_threshold_v2.json (read by
env/mpe_coop_nav.is_collapsed_mpe_occupancy and the sweep analysis).

GATE (pre-registered, and this time ENFORCED): the main sweep may only be
submitted if clean_separation is true, i.e. every w=1 run achieves higher
occupancy than every w=0 run. The v1 calibration reported
clean_separation=false and the 1000-job sweep was submitted anyway --
that data was unusable. Do not repeat this.

History note (v1, r_global-based): trained w=0 scored -34 and trained w=1
scored -96 on episode r_global -- INVERTED relative to the design intent,
because (a) the r_global gap between clustering and spreading is small
(scripted probe: -48 vs -21) and (b) 600k single-copy training failed to
learn the w=1 end at all (below the -46 no-op baseline). v2 uses the
occupancy metric and the strengthened training config in exp_ppo_mpe.py.
"""
import os
import sys
import json
import pandas as pd

CSV = "docs/exp_mpe/mpe_calibration_v2_raw_data.csv"
OUT_JSON = "docs/exp_mpe/mpe_collapse_threshold_v2.json"


def main():
    if not os.path.exists(CSV):
        print(f"No calibration data at {CSV} -- run exp_mpe_calibrate.py (20 runs) first.")
        sys.exit(0)
    df = pd.read_csv(CSV)
    g = df.groupby("w")["mean_occupancy"]
    stats = g.agg(["count", "mean", "std", "min", "max"])
    print("Calibration occupancy distribution by w:\n")
    print(stats.round(3).to_string())
    print("\nSecondary metric (episode r_global) by w:\n")
    print(df.groupby("w")["mean_reward"].agg(["mean", "std", "min", "max"]).round(1).to_string())

    if not {0.0, 1.0}.issubset(set(df["w"].unique())):
        print("\nNeed both w=0.0 and w=1.0 present. Found:", sorted(df["w"].unique()))
        sys.exit(1)

    m0, m1 = stats.loc[0.0, "mean"], stats.loc[1.0, "mean"]
    theta = (m0 + m1) / 2.0
    clean = stats.loc[1.0, "min"] > stats.loc[0.0, "max"]
    overlap = stats.loc[0.0, "max"] - stats.loc[1.0, "min"]

    print(f"\n  mean(occupancy | w=0, collapse)     = {m0:.3f}")
    print(f"  mean(occupancy | w=1, coordination) = {m1:.3f}")
    print(f"  -> theta_occ (midpoint)             = {theta:.3f}")
    print(f"  clean separation: {clean}"
          + ("" if clean else f"  [overlap = {overlap:.3f}]"))

    if m1 <= m0:
        print("\n*** CALIBRATION FAILED: w=1 occupancy <= w=0 occupancy (inverted). ***")
        print("*** Do NOT submit the sweep. Fix training before proceeding.        ***")
    elif not clean:
        print("\n*** GATE NOT PASSED: distributions overlap. Inspect before submitting. ***")
    else:
        print("\nGate PASSED -- sweep may be submitted (qsub docs/slurm/run_ppo_mpe.pbs).")

    payload = {
        "threshold_occupancy": round(float(theta), 4),
        "w0_mean_occ": round(float(m0), 4), "w1_mean_occ": round(float(m1), 4),
        "w0_max_occ": round(float(stats.loc[0.0, "max"]), 4),
        "w1_min_occ": round(float(stats.loc[1.0, "min"]), 4),
        "clean_separation": bool(clean),
        "inverted": bool(m1 <= m0),
        "n_seeds_per_w": int(stats.loc[0.0, "count"]),
        "rule": "midpoint of mean per-step landmark occupancy at trained w=0 vs w=1 (null condition)",
        "source_csv": CSV,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT_JSON}:  threshold_occupancy = {payload['threshold_occupancy']}")


if __name__ == "__main__":
    main()
