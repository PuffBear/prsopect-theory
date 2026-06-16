"""
Set the RWARE collapse threshold from the v2 calibration runs, with the
submission gate (mirrors analyze_mpe_calibrate.py).

Writes docs/exp_rware/rware_collapse_threshold.json (read by
env/rware_wrapper.is_collapsed_rware).
"""
import os
import sys
import json
import pandas as pd

CSV = "docs/exp_rware/rware_calibration_v2_raw_data.csv"
OUT_JSON = "docs/exp_rware/rware_collapse_threshold.json"


def main():
    if not os.path.exists(CSV):
        print(f"No calibration data at {CSV} -- run exp_rware_calibrate.py (20 runs) first.")
        sys.exit(0)
    df = pd.read_csv(CSV)
    stats = df.groupby("w")["mean_reward"].agg(["count", "mean", "std", "min", "max"])
    print("Calibration deliveries-per-episode by w:\n")
    print(stats.round(3).to_string())

    if not {0.0, 1.0}.issubset(set(df["w"].unique())):
        print("\nNeed both w=0.0 and w=1.0 present. Found:", sorted(df["w"].unique()))
        sys.exit(1)

    m0, m1 = stats.loc[0.0, "mean"], stats.loc[1.0, "mean"]
    theta = (m0 + m1) / 2.0
    clean = stats.loc[1.0, "min"] > stats.loc[0.0, "max"]

    print(f"\n  mean(deliveries | w=0) = {m0:.3f}")
    print(f"  mean(deliveries | w=1) = {m1:.3f}")
    print(f"  -> theta (midpoint)    = {theta:.3f}")
    print(f"  clean separation: {clean}")

    if m1 < 1.0:
        print("\n*** CALIBRATION FAILED: w=1 endpoint averages < 1 delivery/episode. ***")
        print("*** The sparse-reward dead-end is real at this horizon.              ***")
        print("*** Do NOT submit the sweep. Report back instead.                    ***")
    elif m1 <= m0:
        print("\n*** CALIBRATION FAILED: endpoints inverted. Do NOT submit the sweep. ***")
    elif not clean:
        print("\n*** GATE NOT PASSED: distributions overlap. Inspect before submitting. ***")
    else:
        print("\nGate PASSED -- sweep may be submitted (qsub docs/slurm/run_ppo_rware_v2.pbs).")

    payload = {
        "threshold": round(float(theta), 4),
        "w0_mean": round(float(m0), 4), "w1_mean": round(float(m1), 4),
        "w0_max": round(float(stats.loc[0.0, "max"]), 4),
        "w1_min": round(float(stats.loc[1.0, "min"]), 4),
        "clean_separation": bool(clean),
        "inverted": bool(m1 <= m0),
        "n_seeds_per_w": int(stats.loc[0.0, "count"]),
        "rule": "midpoint of mean deliveries/episode at trained w=0 vs w=1 (null condition)",
        "source_csv": CSV,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT_JSON}: threshold = {payload['threshold']}")


if __name__ == "__main__":
    main()
