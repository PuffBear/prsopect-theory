"""
Behavioral-excess decomposition: isolate the collapse-boundary shift from Prospect
Theory by comparing Exp B-Reduced (λ∈{1,3,7}, α=β=0.88) against Exp N (α=β=1, λ=1 —
the true null with no PT distortion at all).

  behavioral_excess(λ) = w_crit(ExpB-Reduced, λ) − w_crit(ExpN, null)
  excess_ci(λ)         = [wc_lo(λ) − wc_hi(null),  wc_hi(λ) − wc_lo(null)]   (conservative)

Note on decomposition: excess(λ=1) captures CURVATURE alone (B-Reduced λ=1 differs from
the null only by α=β=0.88 vs 1); the increment with λ (excess(3)−excess(1), etc.)
isolates loss aversion on top of curvature.

Outputs: decomposition table, verdict, fig2_behavioral_decomposition.png,
fig2b_behavioral_excess_bar.png, behavioral_excess_results.md.
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
from analyze_expN import fit_logistic, bootstrap_wcrit, _coerce, curve

EXPN_CSV = "docs/expN_figures/expN_raw_data.csv"
EXPB_CSV = "docs/expB_reduced_figures/expB_reduced_raw_data.csv"
OUTDIR = "docs/expB_reduced_figures"
LAMBDAS = [1, 3, 7]
COLORS = {1: "tab:blue", 3: "tab:orange", 7: "tab:red"}


def wcrit_of(df):
    m, slope, pval, r2, w_crit = fit_logistic(df)
    _, lo, hi = bootstrap_wcrit(df)
    return {"model": m, "slope": slope, "pval": pval, "r2": r2,
            "w_crit": w_crit, "ci_lo": lo, "ci_hi": hi}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for p in (EXPN_CSV, EXPB_CSV):
        if not os.path.exists(p):
            print(f"Missing required data: {p}")
            sys.exit(0)

    null_df = _coerce(pd.read_csv(EXPN_CSV))
    nb = wcrit_of(null_df)

    bdf = pd.read_csv(EXPB_CSV)
    bdf["lambda_loss"] = bdf["lambda_loss"].astype(float).round().astype(int)
    bdf = _coerce(bdf)

    lam_res = {}
    for lam in LAMBDAS:
        r = wcrit_of(bdf[bdf["lambda_loss"] == lam])
        r["excess"] = r["w_crit"] - nb["w_crit"]
        r["excess_lo"] = r["ci_lo"] - nb["ci_hi"]   # conservative
        r["excess_hi"] = r["ci_hi"] - nb["ci_lo"]
        r["significant"] = r["excess_lo"] > 0
        lam_res[lam] = r

    # ---------- decomposition table ----------
    print("\n" + "=" * 86)
    print(f"{'λ':>5} | {'w_crit':>7} | {'w_crit CI':>16} | {'excess':>7} | {'excess CI':>18} | sig?")
    print("-" * 86)
    print(f"{'null':>5} | {nb['w_crit']:>7.3f} | [{nb['ci_lo']:.3f}, {nb['ci_hi']:.3f}]  | "
          f"{0.0:>7.3f} | {'—':>18} | —")
    for lam in LAMBDAS:
        r = lam_res[lam]
        print(f"{lam:>5} | {r['w_crit']:>7.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  | "
              f"{r['excess']:>7.3f} | [{r['excess_lo']:.3f}, {r['excess_hi']:.3f}] | "
              f"{'yes' if r['significant'] else 'no'}")
    print("=" * 86)

    # ---------- verdict ----------
    e = {lam: lam_res[lam]["excess"] for lam in LAMBDAS}
    if lam_res[7]["excess_lo"] > 0:
        verdict = "BEHAVIORAL CLAIM CONFIRMED: loss aversion shifts isocline"
    elif e[1] < e[3] < e[7]:
        verdict = "MONOTONE TREND, underpowered"
    else:
        verdict = "BEHAVIORAL NULL at λ≤7"
    print(f"\nVERDICT: {verdict}")

    # ---------- Figure 2: four logistic curves ----------
    sns.set_style("whitegrid")
    wg = np.linspace(0.4, 1.0, 200)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wg, curve(nb["model"], wg), color="gray", ls="--", lw=2.2, label="null (α=β=1)")
    ax.axvline(nb["w_crit"], color="gray", ls="--", lw=1.3)
    ax.axvspan(nb["ci_lo"], nb["ci_hi"], color="gray", alpha=0.10)
    for lam in LAMBDAS:
        r = lam_res[lam]; c = COLORS[lam]
        ax.plot(wg, curve(r["model"], wg), color=c, lw=2.2, label=f"λ={lam}")
        ax.axvline(r["w_crit"], color=c, ls="--", lw=1.3)
        ax.axvspan(r["ci_lo"], r["ci_hi"], color=c, alpha=0.10)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
    ax.set_title("Behavioral decomposition: null vs PT loss aversion")
    ax.legend()
    plt.tight_layout()
    f2 = os.path.join(OUTDIR, "fig2_behavioral_decomposition.png")
    plt.savefig(f2, dpi=150); print(f"\nSaved {f2}")

    # ---------- Figure 2b: behavioral_excess bar chart ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(LAMBDAS))
    vals = [lam_res[l]["excess"] for l in LAMBDAS]
    lo = [lam_res[l]["excess"] - lam_res[l]["excess_lo"] for l in LAMBDAS]
    hi = [lam_res[l]["excess_hi"] - lam_res[l]["excess"] for l in LAMBDAS]
    bars = ax.bar(xs, vals, color=[COLORS[l] for l in LAMBDAS], alpha=0.85)
    ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor="black", capsize=5, lw=1.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([f"λ={l}" for l in LAMBDAS])
    ax.set_ylabel("behavioral_excess = w_crit(λ) − w_crit(null)")
    ax.set_title("Behavioral excess by loss aversion λ")
    plt.tight_layout()
    f2b = os.path.join(OUTDIR, "fig2b_behavioral_excess_bar.png")
    plt.savefig(f2b, dpi=150); print(f"Saved {f2b}")

    # ---------- results md ----------
    md = ["# Behavioral-excess decomposition (Exp B-Reduced vs Exp N null)\n",
          f"Null (α=β=1): w_crit = {nb['w_crit']:.3f} [{nb['ci_lo']:.3f}, {nb['ci_hi']:.3f}].\n",
          "| λ | w_crit | w_crit CI | behavioral_excess | excess CI | significant |",
          "|---|---|---|---|---|---|",
          f"| null | {nb['w_crit']:.3f} | [{nb['ci_lo']:.3f}, {nb['ci_hi']:.3f}] | 0 (baseline) | — | — |"]
    for lam in LAMBDAS:
        r = lam_res[lam]
        md.append(f"| {lam} | {r['w_crit']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                  f"{r['excess']:.3f} | [{r['excess_lo']:.3f}, {r['excess_hi']:.3f}] | "
                  f"{'yes' if r['significant'] else 'no'} |")
    md += [f"\n## Verdict\n**{verdict}**\n",
           "significant := excess_ci_lo > 0 (conservative difference CI).",
           "\nDecomposition: excess(λ=1) ≈ curvature effect (α=β=0.88 vs 1); the "
           "increment with λ isolates loss aversion.",
           "\nFigures: [fig2_behavioral_decomposition.png](fig2_behavioral_decomposition.png), "
           "[fig2b_behavioral_excess_bar.png](fig2b_behavioral_excess_bar.png)"]
    with open(os.path.join(OUTDIR, "behavioral_excess_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'behavioral_excess_results.md')}")


if __name__ == "__main__":
    main()
