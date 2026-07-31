"""
Analyze PPO extended lambda sweep: merge lambda  in  {2,4,5,6} with existing {1,3,7}
for a complete 7-point w_crit(lambda) curve.

Reads:
  - docs/expB_reduced_figures/expB_reduced_raw_data.csv  (lambda=1,3,7)
  - docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv  (lambda=2,4,5,6)

Outputs:
  - Full 7-point logistic fits
  - Figure: w_crit vs lambda (full curve)
  - Figure: 7 collapse curves overlaid
  - ppo_lambda_sweep_results.md
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

ORIG_CSV = "docs/expB_reduced_figures/expB_reduced_raw_data.csv"
NEW_CSV = "docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv"
OUTDIR = "docs/expPPO_lambda_sweep_figures"
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


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    dfs = []
    for p in (ORIG_CSV, NEW_CSV):
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
            print(f"Loaded {p} ({len(dfs[-1])} rows)")
        else:
            print(f"Missing: {p}")
    if not dfs:
        print("No data available.")
        sys.exit(0)

    df = pd.concat(dfs, ignore_index=True)
    df["lambda_loss"] = df["lambda_loss"].astype(float).round().astype(int)
    df = _coerce(df)
    print(f"\nCombined: {len(df)} rows, lambda values: {sorted(df['lambda_loss'].unique())}")

    res = {}
    for lam in LAMBDAS:
        r = analyze_lambda(df, lam)
        if r is not None:
            res[lam] = r

    # --- Print results --------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'lambda':>6} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | {'slope':>10} | {'p':>9}")
    print("-" * 72)
    for lam in LAMBDAS:
        if lam in res:
            r = res[lam]
            print(f"{lam:>6} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | {r['ci_hi']:>6.3f} | "
                  f"{r['slope']:>10.3f} | {r['pval']:>9.2e}")
    print("=" * 72)

    # Monotonicity check
    valid_lams = sorted(res.keys())
    wcrits = [res[l]["w_crit"] for l in valid_lams]
    monotone = all(wcrits[i] <= wcrits[i+1] for i in range(len(wcrits)-1))
    print(f"\nMonotonicity (w_crit increasing with lambda): {'[ok] YES' if monotone else '[x] NO'}")

    # --- Figure 1: 7 collapse curves ------------------------------------
    sns.set_style("whitegrid")
    wg = np.linspace(0.3, 1.0, 200)
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
    ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
    ax.set_title("PPO on or-gym: collapse transition by lambda (full sweep, alpha=beta=0.88)")
    ax.legend(title="loss aversion", ncol=2)
    plt.tight_layout()
    f1 = os.path.join(OUTDIR, "ppo_full_lambda_collapse_curves.png")
    plt.savefig(f1, dpi=150); print(f"\nSaved {f1}")

    # --- Figure 2: w_crit vs lambda (full 7-point) --------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = valid_lams
    ys = [res[l]["w_crit"] for l in valid_lams]
    lo = [res[l]["w_crit"] - res[l]["ci_lo"] for l in valid_lams]
    hi = [res[l]["ci_hi"] - res[l]["w_crit"] for l in valid_lams]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color="black", capsize=5, lw=2, markersize=8)
    for l in valid_lams:
        ax.scatter([l], [res[l]["w_crit"]], color=COLORS[l], s=80, zorder=5)
    ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
    ax.set_xticks(LAMBDAS)
    ax.set_title("w_crit vs lambda - Full PPO Sweep (or-gym, alpha=beta=0.88)")
    plt.tight_layout()
    f2 = os.path.join(OUTDIR, "ppo_wcrit_vs_lambda_full.png")
    plt.savefig(f2, dpi=150); print(f"Saved {f2}")

    # --- Results markdown -----------------------------------------------
    md = ["# PPO Full Lambda Sweep - w_crit by lambda (or-gym, alpha=beta=0.88)\n",
          f"{len(df)} runs total (combined). Bootstrap n=2000.\n",
          f"Monotonicity check: {'PASS' if monotone else 'FAIL'}\n",
          "| lambda | w_crit | CI_lo | CI_hi | slope | p | n |",
          "|---|---|---|---|---|---|---|"]
    for lam in LAMBDAS:
        if lam in res:
            r = res[lam]
            md.append(f"| {lam} | {r['w_crit']:.3f} | {r['ci_lo']:.3f} | {r['ci_hi']:.3f} | "
                      f"{r['slope']:.3f} | {r['pval']:.2e} | {r['n']} |")
    md += ["\nFigures: [ppo_full_lambda_collapse_curves.png](ppo_full_lambda_collapse_curves.png), "
           "[ppo_wcrit_vs_lambda_full.png](ppo_wcrit_vs_lambda_full.png)"]
    with open(os.path.join(OUTDIR, "ppo_lambda_sweep_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'ppo_lambda_sweep_results.md')}")


if __name__ == "__main__":
    main()
