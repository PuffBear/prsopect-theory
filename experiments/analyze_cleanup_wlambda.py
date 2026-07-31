"""
Analyze Experiment B -- custom-Cleanup w_crit(lambda). Same analysis as
analyze_meltingpot_cleanup.py (estimators reused verbatim), pointed at the
custom-Cleanup sweep CSV.

Reuses the paper's estimators (no new estimator):
  - analyze_expN.fit_logistic / bootstrap_wcrit / per_w_table / curve
  - analyze_firth_lambda_sweep.firth_logit (Firth for steep cells)

Per lambda: w_crit = -b0/b1 + bootstrap 95% CI; Firth w_crit + max |MLE-Firth|
disagreement; transition width Delta_w; direction verdict (does w_crit rise with
lambda?). Nothing tuned.

Run: python experiments/analyze_cleanup_wlambda.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import (fit_logistic, bootstrap_wcrit, _coerce, per_w_table, curve)
from analyze_firth_lambda_sweep import firth_logit

RAW_CSV = "docs/expCleanup_wlambda_figures/cleanup_wlambda_raw_data.csv"
OUTDIR  = "docs/expCleanup_wlambda_figures"
LAMBDAS = [1, 2, 3, 4, 5, 6, 7]


def transition_width(model, lo=0.25, hi=0.75, grid=None):
    if grid is None:
        grid = np.linspace(0.0, 1.0, 1001)
    p = curve(model, grid)
    w_at = lambda t: float(grid[int(np.argmin(np.abs(p - t)))])
    return abs(w_at(hi) - w_at(lo))


def analyze_lambda(df, lam):
    sub = df[df["lambda_loss"] == lam]
    if sub["collapsed"].nunique() < 2 or sub["w"].nunique() < 2:
        return {"lambda": lam, "n": len(sub), "cells": per_w_table(sub), "degenerate": True}
    m, slope, pval, r2, w_crit = fit_logistic(sub)
    _wc, wc_lo, wc_hi = bootstrap_wcrit(sub)
    firth = firth_logit(sub["w"].to_numpy(float), sub["collapsed"].to_numpy(int))
    return {"lambda": lam, "n": len(sub), "cells": per_w_table(sub), "model": m,
            "slope": slope, "pval": pval, "w_crit": w_crit, "ci_lo": wc_lo, "ci_hi": wc_hi,
            "w_crit_firth": firth["wcrit"], "delta_w": transition_width(m),
            "degenerate": False}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(RAW_CSV):
        print(f"No data at {RAW_CSV} yet. Run the sweep + --merge first.")
        sys.exit(0)
    df = _coerce(pd.read_csv(RAW_CSV))
    df["lambda_loss"] = df["lambda_loss"].astype(float).round().astype(int)
    print(f"Loaded {len(df)} rows, lambda={sorted(df['lambda_loss'].unique())}")

    res = {lam: analyze_lambda(df, lam) for lam in LAMBDAS
           if len(df[df["lambda_loss"] == lam]) > 0}

    print("\n" + "=" * 84)
    print(f"{'lam':>3} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | {'Firth':>7} | "
          f"{'|MLE-F|':>7} | {'Dw':>5} | {'p':>9} | n")
    print("-" * 84)
    max_disagree = 0.0
    for lam in LAMBDAS:
        r = res.get(lam)
        if r is None:
            continue
        if r.get("degenerate"):
            print(f"{lam:>3} | (degenerate: no boundary in grid) n={r['n']}"); continue
        d = abs(r["w_crit"] - r["w_crit_firth"]); max_disagree = max(max_disagree, d)
        print(f"{lam:>3} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | {r['ci_hi']:>6.3f} | "
              f"{r['w_crit_firth']:>7.3f} | {d:>7.3f} | {r['delta_w']:>5.2f} | "
              f"{r['pval']:>9.2e} | {r['n']}")
    print("=" * 84)
    print(f"Max |MLE - Firth| w_crit disagreement: {max_disagree:.3f}")

    fit_lams = [l for l in LAMBDAS if l in res and not res[l].get("degenerate")]
    wcrits = [res[l]["w_crit"] for l in fit_lams]
    monotone = all(wcrits[i] <= wcrits[i+1] for i in range(len(wcrits)-1)) if len(wcrits) > 1 else False
    endpoints_sep = (len(fit_lams) >= 2 and res[fit_lams[-1]]["ci_lo"] > res[fit_lams[0]]["ci_hi"])
    verdict = ("PASS (w_crit monotone non-decreasing in lambda)" if monotone else
               "PASS-direction (endpoints CI-separated upward)" if endpoints_sep else
               "UNRESOLVED at this seed count -- report as-is; propose more seeds / fallback")
    print(f"\nDIRECTION VERDICT: {verdict}")

    # Figures
    wg = np.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    for i, lam in enumerate(fit_lams):
        r = res[lam]; c = cmap(i / max(len(fit_lams) - 1, 1))
        ax.plot(wg, curve(r["model"], wg), color=c, lw=2, label=f"lambda={lam}")
        cells = r["cells"]
        ax.errorbar(cells["w"], cells["frac"],
                    yerr=[cells["frac"] - cells["lo"], cells["hi"] - cells["frac"]],
                    fmt="o", color=c, capsize=3, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
    ax.set_title("Custom Cleanup (MAPPO): collapse transition by lambda")
    ax.legend(title="loss aversion", ncol=2); plt.tight_layout()
    f1 = os.path.join(OUTDIR, "cleanup_wlambda_collapse_curves.png")
    plt.savefig(f1, dpi=150); print(f"Saved {f1}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(fit_lams, [res[l]["w_crit"] for l in fit_lams],
                yerr=[[res[l]["w_crit"] - res[l]["ci_lo"] for l in fit_lams],
                      [res[l]["ci_hi"] - res[l]["w_crit"] for l in fit_lams]],
                fmt="o-", color="black", capsize=5, lw=2, markersize=8)
    ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
    ax.set_xticks(LAMBDAS); ax.set_title("w_crit vs lambda -- custom Cleanup")
    plt.tight_layout()
    f2 = os.path.join(OUTDIR, "cleanup_wlambda_wcrit_vs_lambda.png")
    plt.savefig(f2, dpi=150); print(f"Saved {f2}")

    md = ["# Experiment B -- custom Cleanup: w_crit by lambda (MAPPO)\n",
          f"{len(df)} runs. Estimators: statsmodels logistic + Firth. Bootstrap n=2000.\n",
          "Substrate: in-repo env/cleanup_env.py (real DeepMind Melting Pot blocked by "
          "cluster glibc 2.17 < dmlab2d's 2.31 requirement; see RESULTS_B.md).\n",
          f"**Direction verdict: {verdict}**\n",
          f"Max |MLE - Firth| w_crit disagreement: {max_disagree:.3f}\n",
          "| lambda | w_crit | CI_lo | CI_hi | Firth | Delta_w | slope | p | n |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lam in LAMBDAS:
        r = res.get(lam)
        if r is None or r.get("degenerate"):
            continue
        md.append(f"| {lam} | {r['w_crit']:.3f} | {r['ci_lo']:.3f} | {r['ci_hi']:.3f} | "
                  f"{r['w_crit_firth']:.3f} | {r['delta_w']:.2f} | {r['slope']:.2f} | "
                  f"{r['pval']:.2e} | {r['n']} |")
    md += ["\nFigures: [cleanup_wlambda_collapse_curves.png](cleanup_wlambda_collapse_curves.png), "
           "[cleanup_wlambda_wcrit_vs_lambda.png](cleanup_wlambda_wcrit_vs_lambda.png)"]
    with open(os.path.join(OUTDIR, "cleanup_wlambda_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'cleanup_wlambda_results.md')}")


if __name__ == "__main__":
    main()
