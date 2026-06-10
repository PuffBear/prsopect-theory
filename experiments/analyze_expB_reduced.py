"""
Analyze Exp B-Reduced: w_crit(lambda) for lambda  in  {1,3,7} (alpha=beta=0.88), and whether loss
aversion shifts the collapse isocline.

Outputs:
  - per-lambda logistic (slope, p, McFadden R^2), w_crit with run-level bootstrap CI (n=2000)
  - summary table; CI-overlap behavioral test
  - Figure 2  : three logistic curves + w_crit lines/CI  (expB_wcrit_by_lambda.png)
  - Figure 2b : w_crit vs lambda with CI error bars            (expB_wcrit_vs_lambda.png)
  - expB_results.md

Sign note: statsmodels Logit `slope` is the w-coefficient (negative; collapse
decreases with w); the spec's "b1" in P=1/(1+exp(b0+b1*w)) equals -slope. w_crit =
-b0/slope either way. Reported below as `b1 = slope` (statsmodels convention),
explicitly labeled.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import wilson_ci, fit_logistic, bootstrap_wcrit, _coerce, per_w_table, curve

CSV = "docs/expB_reduced_figures/expB_reduced_raw_data.csv"
OUTDIR = "docs/expB_reduced_figures"
LAMBDAS = [1, 3, 7]
COLORS = {1: "tab:blue", 3: "tab:orange", 7: "tab:red"}


def analyze_lambda(df, lam):
    sub = df[df["lambda_loss"] == lam]
    cells = per_w_table(sub)
    m, slope, pval, r2, w_crit = fit_logistic(sub)
    wc_med, wc_lo, wc_hi = bootstrap_wcrit(sub)
    return {"lambda": lam, "n": len(sub), "cells": cells, "model": m, "slope": slope,
            "pval": pval, "r2": r2, "w_crit": w_crit, "ci_lo": wc_lo, "ci_hi": wc_hi}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(CSV):
        print(f"No data at {CSV}")
        sys.exit(0)
    df = pd.read_csv(CSV)
    df["lambda_loss"] = df["lambda_loss"].astype(float).round().astype(int)
    df = _coerce(df)  # coerces w, collapsed; keeps other columns

    res = {}
    for lam in LAMBDAS:
        res[lam] = analyze_lambda(df, lam)
        r = res[lam]
        print(f"\n=== lambda={lam} (n={r['n']}) per-w collapse fraction ===")
        for _, c in r["cells"].iterrows():
            print(f"  w={c['w']:.2f}: {int(c['k']):2d}/{int(c['n']):2d}={c['frac']:.2f} "
                  f"[{c['lo']:.2f},{c['hi']:.2f}]")

    print("\n" + "=" * 64)
    print(f"{'lambda':>6} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | {'b1(slope)':>10} | {'p':>9}")
    print("-" * 64)
    for lam in LAMBDAS:
        r = res[lam]
        print(f"{lam:>6} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | {r['ci_hi']:>6.3f} | "
              f"{r['slope']:>10.3f} | {r['pval']:>9.2e}")
    print("=" * 64)

    # Behavioral CI-overlap test
    lo7, hi1 = res[7]["ci_lo"], res[1]["ci_hi"]
    if lo7 > hi1:
        verdict = "BEHAVIORAL CLAIM CONFIRMED"
        detail = f"w_crit(lambda=7) CI_lo={lo7:.3f} > w_crit(lambda=1) CI_hi={hi1:.3f}"
    else:
        verdict = "CIs OVERLAP - behavioral claim not established at this resolution"
        detail = f"w_crit(lambda=7) CI_lo={lo7:.3f} <= w_crit(lambda=1) CI_hi={hi1:.3f}"
    print(f"\n{verdict}\n  ({detail})")

    # ---------- Figure 2: three logistic curves ----------
    sns.set_style("whitegrid")
    wg = np.linspace(0.4, 1.0, 200)
    fig, ax = plt.subplots(figsize=(8, 6))
    for lam in LAMBDAS:
        r = res[lam]; c = COLORS[lam]
        ax.plot(wg, curve(r["model"], wg), color=c, lw=2.2, label=f"lambda={lam}")
        ax.errorbar(r["cells"]["w"], r["cells"]["frac"],
                    yerr=[r["cells"]["frac"] - r["cells"]["lo"], r["cells"]["hi"] - r["cells"]["frac"]],
                    fmt="o", color=c, capsize=3, alpha=0.6)
        ax.axvline(r["w_crit"], color=c, ls="--", lw=1.4)
        ax.axvspan(r["ci_lo"], r["ci_hi"], color=c, alpha=0.10)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
    ax.set_title("Exp B-Reduced: collapse transition by loss aversion lambda (alpha=beta=0.88)")
    ax.legend(title="loss aversion")
    plt.tight_layout()
    f2 = os.path.join(OUTDIR, "expB_wcrit_by_lambda.png")
    plt.savefig(f2, dpi=150); print(f"\nSaved {f2}")

    # ---------- Figure 2b: w_crit vs lambda with CI error bars ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = LAMBDAS
    ys = [res[l]["w_crit"] for l in LAMBDAS]
    lo = [res[l]["w_crit"] - res[l]["ci_lo"] for l in LAMBDAS]
    hi = [res[l]["ci_hi"] - res[l]["w_crit"] for l in LAMBDAS]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color="black", capsize=5, lw=2, markersize=8)
    for l in LAMBDAS:
        ax.scatter([l], [res[l]["w_crit"]], color=COLORS[l], s=80, zorder=5)
    ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
    ax.set_xticks(LAMBDAS)
    ax.set_title("w_crit vs lambda (Exp B-Reduced)")
    plt.tight_layout()
    f2b = os.path.join(OUTDIR, "expB_wcrit_vs_lambda.png")
    plt.savefig(f2b, dpi=150); print(f"Saved {f2b}")

    # ---------- results md ----------
    md = ["# Exp B-Reduced - w_crit by loss aversion lambda (alpha=beta=0.88)\n",
          f"{len(df)} runs total. Bootstrap n=2000 (run-level).\n",
          "| lambda | w_crit | CI_lo | CI_hi | b1 (slope) | p | McFadden R^2 | n |",
          "|---|---|---|---|---|---|---|---|"]
    for lam in LAMBDAS:
        r = res[lam]
        md.append(f"| {lam} | {r['w_crit']:.3f} | {r['ci_lo']:.3f} | {r['ci_hi']:.3f} | "
                  f"{r['slope']:.3f} | {r['pval']:.2e} | {r['r2']:.3f} | {r['n']} |")
    md += [f"\n## Behavioral test\n", f"**{verdict}**", f"- {detail}",
           "\nFigures: [expB_wcrit_by_lambda.png](expB_wcrit_by_lambda.png), "
           "[expB_wcrit_vs_lambda.png](expB_wcrit_vs_lambda.png)",
           "\n(b1 is the statsmodels w-coefficient; the spec's P=1/(1+exp(b0+b1*w)) "
           "uses b1=-slope. w_crit=-b0/slope is identical either way.)"]
    with open(os.path.join(OUTDIR, "expB_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'expB_results.md')}")


if __name__ == "__main__":
    main()
