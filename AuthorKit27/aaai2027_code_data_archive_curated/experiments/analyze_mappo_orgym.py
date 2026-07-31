"""
Analyze MAPPO on or-gym: w_crit(lambda) surface and comparison to PPO baseline.

Reads:   docs/expMAPPO_orgym_figures/mappo_orgym_raw_data.csv
Outputs:
  - Per-lambda logistic fits, w_crit with bootstrap CI
  - Comparison table: MAPPO vs PPO w_crit at each lambda
  - Figure: MAPPO collapse curves by lambda
  - Figure: PPO vs MAPPO w_crit comparison
  - mappo_orgym_results.md

Reuses the statistical machinery from analyze_expN.py (wilson_ci, fit_logistic,
bootstrap_wcrit, etc.)
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

MAPPO_CSV = "docs/expMAPPO_orgym_figures/mappo_orgym_raw_data.csv"
PPO_CSV = "docs/expB_reduced_figures/expB_reduced_raw_data.csv"
PPO_SWEEP_CSV = "docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv"
OUTDIR = "docs/expMAPPO_orgym_figures"
LAMBDAS = [1, 2, 3, 4, 5, 6, 7]
COLORS = {1: "tab:blue", 2: "tab:cyan", 3: "tab:orange", 4: "tab:olive",
          5: "tab:green", 6: "tab:purple", 7: "tab:red"}


def analyze_lambda(df, lam):
    sub = df[df["lambda_loss"] == lam]
    if len(sub) < 5:
        return None
    cells = per_w_table(sub)
    m, slope, pval, r2, w_crit = fit_logistic(sub)
    wc_med, wc_lo, wc_hi = bootstrap_wcrit(sub)
    return {"lambda": lam, "n": len(sub), "cells": cells, "model": m, "slope": slope,
            "pval": pval, "r2": r2, "w_crit": w_crit, "ci_lo": wc_lo, "ci_hi": wc_hi}


def load_ppo_combined():
    """Load and merge PPO data from Exp B-Reduced (lambda=1,3,7) and lambda sweep (lambda=2,4,5,6)."""
    dfs = []
    if os.path.exists(PPO_CSV):
        dfs.append(pd.read_csv(PPO_CSV))
    if os.path.exists(PPO_SWEEP_CSV):
        dfs.append(pd.read_csv(PPO_SWEEP_CSV))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["lambda_loss"] = df["lambda_loss"].astype(float).round().astype(int)
    return _coerce(df)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(MAPPO_CSV):
        print(f"No MAPPO data at {MAPPO_CSV}. Run the experiment first.")
        sys.exit(0)

    df = pd.read_csv(MAPPO_CSV)
    df["lambda_loss"] = df["lambda_loss"].astype(float).round().astype(int)
    df = _coerce(df)

    mappo_res = {}
    for lam in LAMBDAS:
        r = analyze_lambda(df, lam)
        if r is not None:
            mappo_res[lam] = r

    # --- Print MAPPO results --------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'lambda':>6} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | {'b1(slope)':>10} | {'p':>9} | {'n':>4}")
    print("-" * 72)
    for lam in LAMBDAS:
        if lam in mappo_res:
            r = mappo_res[lam]
            print(f"{lam:>6} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | {r['ci_hi']:>6.3f} | "
                  f"{r['slope']:>10.3f} | {r['pval']:>9.2e} | {r['n']:>4}")
    print("=" * 72)

    # --- Figure 1: MAPPO collapse curves by lambda ---------------------------
    sns.set_style("whitegrid")
    wg = np.linspace(0.3, 1.0, 200)
    fig, ax = plt.subplots(figsize=(9, 6))
    for lam in LAMBDAS:
        if lam not in mappo_res:
            continue
        r = mappo_res[lam]; c = COLORS[lam]
        ax.plot(wg, curve(r["model"], wg), color=c, lw=2.2, label=f"lambda={lam}")
        ax.errorbar(r["cells"]["w"], r["cells"]["frac"],
                    yerr=[r["cells"]["frac"] - r["cells"]["lo"],
                          r["cells"]["hi"] - r["cells"]["frac"]],
                    fmt="o", color=c, capsize=3, alpha=0.6)
        ax.axvline(r["w_crit"], color=c, ls="--", lw=1.2)
        ax.axvspan(r["ci_lo"], r["ci_hi"], color=c, alpha=0.08)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
    ax.set_title("MAPPO on or-gym: collapse transition by lambda (alpha=beta=0.88)")
    ax.legend(title="loss aversion", ncol=2)
    plt.tight_layout()
    f1 = os.path.join(OUTDIR, "mappo_orgym_collapse_curves.png")
    plt.savefig(f1, dpi=150); print(f"\nSaved {f1}")

    # --- Figure 2: PPO vs MAPPO w_crit comparison ----------------------
    ppo_df = load_ppo_combined()
    ppo_res = {}
    if ppo_df is not None:
        for lam in LAMBDAS:
            r = analyze_lambda(ppo_df, lam)
            if r is not None:
                ppo_res[lam] = r

    common_lams = sorted(set(mappo_res.keys()) & set(ppo_res.keys()))
    if common_lams:
        fig, ax = plt.subplots(figsize=(8, 5))
        # PPO
        xs_ppo = common_lams
        ys_ppo = [ppo_res[l]["w_crit"] for l in common_lams]
        lo_ppo = [ppo_res[l]["w_crit"] - ppo_res[l]["ci_lo"] for l in common_lams]
        hi_ppo = [ppo_res[l]["ci_hi"] - ppo_res[l]["w_crit"] for l in common_lams]
        ax.errorbar(xs_ppo, ys_ppo, yerr=[lo_ppo, hi_ppo], fmt="s-", color="tab:blue",
                    capsize=5, lw=2, markersize=8, label="PPO")
        # MAPPO
        ys_mappo = [mappo_res[l]["w_crit"] for l in common_lams]
        lo_mappo = [mappo_res[l]["w_crit"] - mappo_res[l]["ci_lo"] for l in common_lams]
        hi_mappo = [mappo_res[l]["ci_hi"] - mappo_res[l]["w_crit"] for l in common_lams]
        ax.errorbar(common_lams, ys_mappo, yerr=[lo_mappo, hi_mappo], fmt="o-", color="tab:red",
                    capsize=5, lw=2, markersize=8, label="MAPPO")
        ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
        ax.set_xticks(LAMBDAS)
        ax.set_title("w_crit vs lambda: PPO vs MAPPO (or-gym)")
        ax.legend()
        plt.tight_layout()
        f2 = os.path.join(OUTDIR, "ppo_vs_mappo_wcrit.png")
        plt.savefig(f2, dpi=150); print(f"Saved {f2}")

    # --- Results markdown -----------------------------------------------
    md = ["# MAPPO on or-gym - w_crit by lambda (alpha=beta=0.88)\n",
          f"{len(df)} runs total. Bootstrap n=2000.\n",
          "| lambda | w_crit | CI_lo | CI_hi | b1 | p | n |",
          "|---|---|---|---|---|---|---|"]
    for lam in LAMBDAS:
        if lam in mappo_res:
            r = mappo_res[lam]
            md.append(f"| {lam} | {r['w_crit']:.3f} | {r['ci_lo']:.3f} | {r['ci_hi']:.3f} | "
                      f"{r['slope']:.3f} | {r['pval']:.2e} | {r['n']} |")
    if common_lams:
        md += ["\n## PPO vs MAPPO comparison\n",
               "| lambda | PPO w_crit | MAPPO w_crit | Delta |",
               "|---|---|---|---|"]
        for lam in common_lams:
            delta = mappo_res[lam]["w_crit"] - ppo_res[lam]["w_crit"]
            md.append(f"| {lam} | {ppo_res[lam]['w_crit']:.3f} | "
                      f"{mappo_res[lam]['w_crit']:.3f} | {delta:+.3f} |")
    with open(os.path.join(OUTDIR, "mappo_orgym_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'mappo_orgym_results.md')}")


if __name__ == "__main__":
    main()
