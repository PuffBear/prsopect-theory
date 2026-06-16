"""
Analyze MPE experiments: PPO and MAPPO on Cooperative Navigation.

Reads:
  - docs/expPPO_mpe_figures/ppo_mpe_raw_data.csv
  - docs/expMAPPO_mpe_figures/mappo_mpe_raw_data.csv

Outputs (per algorithm):
  - Per-lambda logistic fits, w_crit with bootstrap CI
  - Collapse curves by lambda
  - w_crit vs lambda comparison (PPO vs MAPPO on MPE)
  - Cross-environment comparison (or-gym vs MPE)
  - mpe_results.md

NOTE: The MPE collapse definition may need threshold recalibration after
pilot runs. The is_collapsed_mpe threshold (-30.0) is a starting estimate.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import wilson_ci, fit_logistic, bootstrap_wcrit, _coerce, per_w_table, curve

PPO_MPE_CSV = "docs/expPPO_mpe_figures/ppo_mpe_raw_data.csv"
MAPPO_MPE_CSV = "docs/expMAPPO_mpe_figures/mappo_mpe_raw_data.csv"
OUTDIR = "docs/expMPE_combined_figures"
LAMBDAS = [1, 2, 3, 4, 5, 6, 7]
COLORS = {1: "tab:blue", 2: "tab:cyan", 3: "tab:orange", 4: "tab:olive",
          5: "tab:green", 6: "tab:purple", 7: "tab:red"}
ALGO_COLORS = {"PPO": "tab:blue", "MAPPO": "tab:red"}


def analyze_lambda(df, lam):
    sub = df[df["lambda_loss"] == lam]
    if len(sub) < 5:
        return None
    cells = per_w_table(sub)
    try:
        m, slope, pval, r2, w_crit = fit_logistic(sub)
        wc_med, wc_lo, wc_hi = bootstrap_wcrit(sub)
    except Exception as e:
        print(f"  lambda={lam}: logistic fit failed ({e}), skipping")
        return None
    return {"lambda": lam, "n": len(sub), "cells": cells, "model": m, "slope": slope,
            "pval": pval, "r2": r2, "w_crit": w_crit, "ci_lo": wc_lo, "ci_hi": wc_hi}


# --- Decomposition by CONDITION (the RQ3 design) ----------------------------
# The sweep grid is {null, curvature, lambda3}, NOT a lambda sweep. null and
# curvature BOTH have lambda=1 (they differ in alpha,beta), so grouping by
# lambda would merge them and destroy the curvature-vs-null contrast. The
# decomposition must group by the `condition` column.
CONDITIONS = ["null", "curvature", "lambda3"]


def analyze_condition(df, cond):
    sub = df[df["condition"] == cond]
    if len(sub) < 5:
        return None
    try:
        m, slope, pval, r2, w_crit = fit_logistic(sub)
        wc_med, wc_lo, wc_hi = bootstrap_wcrit(sub)
    except Exception as e:
        print(f"  condition={cond}: logistic fit failed ({e})")
        return None
    return {"condition": cond, "n": len(sub), "w_crit": w_crit,
            "ci_lo": wc_lo, "ci_hi": wc_hi, "slope": slope, "pval": pval, "r2": r2}


def _occupancy_threshold():
    """Calibrated occupancy theta the COLLAPSE flag must use. The sweep's stored
    `collapsed` column can be wrong if it was written against a stale threshold
    file, so we always recompute from raw occupancy here."""
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from env.mpe_coop_nav import load_occupancy_threshold
        return float(load_occupancy_threshold())
    except Exception:
        return 0.164  # MAPPO calibration midpoint (w=0 0.053 / w=1 0.276)


def _safe_read(csv_path):
    """Robust read: keep the literal string 'null' (pandas parses it as NaN by
    default, which would erase the null condition); skip malformed legacy rows;
    and RECOMPUTE the collapse flag from raw occupancy with the calibrated
    threshold (the stored `collapsed` column may use a stale threshold)."""
    df = pd.read_csv(csv_path, keep_default_na=False,
                     na_values=["", "NaN", "nan"], on_bad_lines="skip")
    if "condition" in df.columns:
        df["condition"] = df["condition"].replace({"": "null"}).fillna("null")
    if "mean_occupancy" in df.columns:
        theta = _occupancy_threshold()
        df["mean_occupancy"] = pd.to_numeric(df["mean_occupancy"], errors="coerce")
        df["collapsed"] = df["mean_occupancy"] < theta
    return df


def analyze_decomposition(csv_path, algo_name):
    """Per-CONDITION w_crit + excess vs null (the 3-way decomposition)."""
    if not os.path.exists(csv_path):
        return None
    df = _safe_read(csv_path)
    df = _coerce(df)
    if "condition" not in df.columns:
        print(f"[{algo_name}] no 'condition' column; cannot run decomposition.")
        return None
    res = {c: analyze_condition(df, c) for c in CONDITIONS}
    res = {c: r for c, r in res.items() if r is not None}
    if "null" not in res:
        print(f"[{algo_name}] null condition missing; cannot compute excess.")
        return res
    w0 = res["null"]["w_crit"]
    print(f"\n=== {algo_name} MPE decomposition (by condition) ===")
    lines = ["| condition | w_crit | 95% CI | excess vs null |",
             "|---|---|---|---|"]
    for c in CONDITIONS:
        if c not in res:
            continue
        r = res[c]
        exc = r["w_crit"] - w0
        print(f"  {c:10s}: w_crit={r['w_crit']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
              f"  excess={exc:+.3f}")
        lines.append(f"| {c} | {r['w_crit']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {exc:+.3f} |")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"{algo_name.lower()}_mpe_decomposition.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return res


def analyze_algo(csv_path, algo_name):
    """Run per-lambda analysis for one algorithm's CSV."""
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        return None, None
    try:
        df = _safe_read(csv_path)
        df["lambda_loss"] = df["lambda_loss"].astype(float).round().astype(int)
        df = _coerce(df)
    except Exception as e:
        print(f"[{algo_name}] could not read {csv_path} ({e}); skipping.")
        return None, None
    res = {}
    print(f"\n{'='*60}")
    print(f"  {algo_name} on MPE ({len(df)} runs)")
    print(f"{'='*60}")
    for lam in LAMBDAS:
        r = analyze_lambda(df, lam)
        if r is not None:
            res[lam] = r
            print(f"  lambda={lam}: w_crit={r['w_crit']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    return df, res


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    ppo_df, ppo_res = analyze_algo(PPO_MPE_CSV, "PPO")
    mappo_df, mappo_res = analyze_algo(MAPPO_MPE_CSV, "MAPPO")

    # Primary RQ3 output: w_crit per CONDITION (null / curvature / lambda3).
    analyze_decomposition(PPO_MPE_CSV, "PPO")
    analyze_decomposition(MAPPO_MPE_CSV, "MAPPO")

    if ppo_res is None and mappo_res is None:
        print("No data available for either PPO or MAPPO on MPE.")
        sys.exit(0)

    sns.set_style("whitegrid")

    # --- Figure 1: Per-algorithm collapse curves ------------------------
    for algo_name, res in [("PPO", ppo_res), ("MAPPO", mappo_res)]:
        if res is None or not res:
            continue
        wg = np.linspace(0.0, 1.0, 200)
        fig, ax = plt.subplots(figsize=(9, 6))
        for lam in LAMBDAS:
            if lam not in res:
                continue
            r = res[lam]; c = COLORS[lam]
            ax.plot(wg, curve(r["model"], wg), color=c, lw=2.2, label=f"lambda={lam}")
            ax.errorbar(r["cells"]["w"], r["cells"]["frac"],
                        yerr=[r["cells"]["frac"] - r["cells"]["lo"],
                              r["cells"]["hi"] - r["cells"]["frac"]],
                        fmt="o", color=c, capsize=3, alpha=0.6)
            ax.axvline(r["w_crit"], color=c, ls="--", lw=1.2)
        ax.set_xlim(0.0, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
        ax.set_title(f"{algo_name} on MPE Coop Nav: collapse by lambda (alpha=beta=0.88)")
        ax.legend(title="loss aversion", ncol=2)
        plt.tight_layout()
        fname = os.path.join(OUTDIR, f"{algo_name.lower()}_mpe_collapse_curves.png")
        plt.savefig(fname, dpi=150); print(f"\nSaved {fname}")

    # --- Figure 2: PPO vs MAPPO w_crit on MPE --------------------------
    if ppo_res and mappo_res:
        common_lams = sorted(set(ppo_res.keys()) & set(mappo_res.keys()))
        if common_lams:
            fig, ax = plt.subplots(figsize=(8, 5))
            for algo_name, res, marker in [("PPO", ppo_res, "s"), ("MAPPO", mappo_res, "o")]:
                xs = common_lams
                ys = [res[l]["w_crit"] for l in common_lams]
                lo = [res[l]["w_crit"] - res[l]["ci_lo"] for l in common_lams]
                hi = [res[l]["ci_hi"] - res[l]["w_crit"] for l in common_lams]
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt=f"{marker}-",
                            color=ALGO_COLORS[algo_name], capsize=5, lw=2,
                            markersize=8, label=algo_name)
            ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
            ax.set_xticks(LAMBDAS)
            ax.set_title("w_crit vs lambda: PPO vs MAPPO (MPE Cooperative Navigation)")
            ax.legend()
            plt.tight_layout()
            f2 = os.path.join(OUTDIR, "ppo_vs_mappo_mpe_wcrit.png")
            plt.savefig(f2, dpi=150); print(f"Saved {f2}")

    # --- Results markdown -----------------------------------------------
    md = ["# MPE Cooperative Navigation - Collapse Analysis\n"]
    for algo_name, res in [("PPO", ppo_res), ("MAPPO", mappo_res)]:
        if res is None or not res:
            continue
        md += [f"\n## {algo_name}\n",
               "| lambda | w_crit | CI_lo | CI_hi | slope | p | n |",
               "|---|---|---|---|---|---|---|"]
        for lam in LAMBDAS:
            if lam in res:
                r = res[lam]
                md.append(f"| {lam} | {r['w_crit']:.3f} | {r['ci_lo']:.3f} | {r['ci_hi']:.3f} | "
                          f"{r['slope']:.3f} | {r['pval']:.2e} | {r['n']} |")
    with open(os.path.join(OUTDIR, "mpe_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'mpe_results.md')}")


if __name__ == "__main__":
    main()
