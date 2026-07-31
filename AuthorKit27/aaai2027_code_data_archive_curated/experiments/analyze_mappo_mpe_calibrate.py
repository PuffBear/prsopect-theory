"""
Set the MAPPO-MPE collapse threshold (occupancy-based) from calibration runs.

Reads docs/exp_mpe/mappo_mpe_calibration_raw_data.csv (trained MAPPO at w=0 and
w=1, null condition), computes occupancy at each extreme, places theta_occ at the
midpoint, writes docs/exp_mpe/mappo_mpe_collapse_threshold.json.

GATE (enforced): the main MAPPO sweep may only be submitted if the endpoints
SEPARATE -- w=1 mean occupancy is clearly above w=0 (not inverted, and ideally
non-overlapping). Single-seed Mac smoke gave w=0~0.05-0.16 vs w=1~0.24-0.33; the
10-seed cluster distribution is the real arbiter.
"""
import os
import sys
import json
import pandas as pd

CSV = "docs/exp_mpe/mappo_mpe_calibration_raw_data.csv"
OUT_JSON = "docs/exp_mpe/mappo_mpe_collapse_threshold.json"
MIN_GAP = 0.10   # mean(w=1) - mean(w=0) must exceed this to pass the gate


def main():
    if not os.path.exists(CSV):
        print(f"No calibration data at {CSV} -- run exp_mappo_mpe_calibrate.py (20 runs) first.")
        sys.exit(0)
    df = pd.read_csv(CSV)
    stats = df.groupby("w")["mean_occupancy"].agg(["count", "mean", "std", "min", "max"])
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
    gap = m1 - m0
    overlap = stats.loc[0.0, "max"] - stats.loc[1.0, "min"]

    print(f"\n  mean(occupancy | w=0, collapse)     = {m0:.3f}")
    print(f"  mean(occupancy | w=1, coordination) = {m1:.3f}")
    print(f"  gap (w1 - w0)                       = {gap:+.3f}")
    print(f"  -> theta_occ (midpoint)             = {theta:.3f}")
    print(f"  clean (non-overlapping) separation: {clean}"
          + ("" if clean else f"  [overlap = {overlap:.3f}]"))

    if m1 <= m0:
        verdict = "FAILED_INVERTED"
        print("\n*** CALIBRATION FAILED: w=1 occupancy <= w=0 (inverted). Do NOT submit. ***")
    elif gap < MIN_GAP:
        verdict = "FAILED_WEAK"
        print(f"\n*** GATE NOT PASSED: gap {gap:.3f} < {MIN_GAP}. Too weak; inspect before submitting. ***")
    elif not clean:
        verdict = "PASSED_OVERLAP"
        print(f"\nGate PASSED (mean gap {gap:.3f} >= {MIN_GAP}) but distributions OVERLAP "
              f"(overlap {overlap:.3f}). Boundary is real but noisy -- OK to run the "
              f"sweep; expect a softer sigmoid.")
    else:
        verdict = "PASSED_CLEAN"
        print(f"\nGate PASSED -- clean separation. Submit qsub docs/slurm/run_mappo_mpe.pbs.")

    payload = {
        "threshold_occupancy": round(float(theta), 4),
        "w0_mean_occ": round(float(m0), 4), "w1_mean_occ": round(float(m1), 4),
        "gap": round(float(gap), 4),
        "w0_max_occ": round(float(stats.loc[0.0, "max"]), 4),
        "w1_min_occ": round(float(stats.loc[1.0, "min"]), 4),
        "clean_separation": bool(clean),
        "inverted": bool(m1 <= m0),
        "verdict": verdict,
        "n_seeds_per_w": int(stats.loc[0.0, "count"]),
        "rule": "midpoint of mean per-step landmark occupancy at trained w=0 vs w=1 "
                "(MAPPO, centralized critic + agent-id, 50-step horizon, null condition)",
        "source_csv": CSV,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    # Also write the file the sweep's collapse flag actually reads
    # (env.mpe_coop_nav.load_occupancy_threshold -> mpe_collapse_threshold_v2.json),
    # so calibration -> threshold -> sweep is consistent.
    SWEEP_JSON = "docs/exp_mpe/mpe_collapse_threshold_v2.json"
    with open(SWEEP_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT_JSON} and {SWEEP_JSON}")


if __name__ == "__main__":
    main()
