"""
Metastability line figure (matched alpha=beta=1 at every horizon).

Replaces the grid-misaligned heatmap with two clean line panels:
  (a) P(collapse | w) logistic curves for T in {150k, 300k, 600k}, with empirical
      points (Wilson CIs) and w_crit vertical lines + bootstrap-CI bands.
  (b) w_crit vs training horizon T, with bootstrap 95% CI error bars.

Sources (all alpha=beta=1, lambda=1):
  150k -> docs/expA_figures/expA_raw_data.csv
  300k -> docs/expA300k_figures/expA300k_raw_data.csv
  600k -> docs/exp0b_extended/exp0b_600k_raw_data.csv

Output: docs/exp0b_extended/metastability_lineplot.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from analyze_expN import fit_logistic, bootstrap_wcrit, _coerce, per_w_table, curve

HORIZONS = [
    ("150k", 150_000, "docs/expA_figures/expA_raw_data.csv",        "tab:blue"),
    ("300k", 300_000, "docs/expA300k_figures/expA300k_raw_data.csv", "tab:orange"),
    ("600k", 600_000, "docs/exp0b_extended/exp0b_600k_raw_data.csv", "tab:red"),
]
OUT = "docs/exp0b_extended/metastability_lineplot.png"


def load(csv):
    df = pd.read_csv(csv)
    if "lambda" in df.columns:        # Exp A / 300k carry a lambda col; keep the null
        df = df[df["lambda"] == 1.0]
    return _coerce(df)


def main():
    res = []
    for label, T, csv, color in HORIZONS:
        if not os.path.exists(csv):
            print(f"[skip] {label}: missing {csv}"); continue
        df = load(csv)
        m, slope, p, r2, wc = fit_logistic(df)
        _, lo, hi = bootstrap_wcrit(df)
        res.append(dict(label=label, T=T, color=color, df=df, model=m,
                        wc=wc, lo=lo, hi=hi, cells=per_w_table(df)))
        print(f"  {label}: w_crit={wc:.3f} [{lo:.3f}, {hi:.3f}]  slope={slope:.1f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1.7, 1]})

    # (a) logistic curves
    wg = np.linspace(0.0, 1.0, 300)
    for r in res:
        c = r["color"]
        axL.plot(wg, curve(r["model"], wg), color=c, lw=2.3, label=f"T={r['label']}")
        cells = r["cells"]
        axL.errorbar(cells["w"], cells["frac"],
                     yerr=[cells["frac"] - cells["lo"], cells["hi"] - cells["frac"]],
                     fmt="o", ms=4, color=c, capsize=2, alpha=0.55, lw=1)
        axL.axvline(r["wc"], color=c, ls="--", lw=1.3)
        axL.axvspan(r["lo"], r["hi"], color=c, alpha=0.08)
    axL.set_xlim(0.0, 1.0); axL.set_ylim(-0.04, 1.04)
    axL.set_xlabel("w (reward centralization)")
    axL.set_ylabel("P(collapse)")
    axL.set_title("Collapse boundary slides left with training horizon\n(matched $\\alpha=\\beta=1$)")
    axL.legend(title="horizon", loc="upper right")
    axL.grid(alpha=0.3)

    # (b) w_crit vs T
    Ts = [r["T"] / 1000 for r in res]
    wcs = [r["wc"] for r in res]
    lo = [r["wc"] - r["lo"] for r in res]
    hi = [r["hi"] - r["wc"] for r in res]
    axR.errorbar(Ts, wcs, yerr=[lo, hi], fmt="o-", color="black", capsize=5,
                 lw=2, markersize=8, zorder=3)
    for r in res:
        axR.scatter([r["T"] / 1000], [r["wc"]], color=r["color"], s=90, zorder=4)
        axR.annotate(f"{r['wc']:.3f}", (r["T"] / 1000, r["wc"]),
                     textcoords="offset points", xytext=(8, 8), fontsize=9)
    axR.set_xlabel("training horizon T (k steps)")
    axR.set_ylabel("$w_{\\mathrm{crit}}$ (P=0.5)")
    axR.set_title("$w_{\\mathrm{crit}}$ vs horizon")
    axR.set_xticks(Ts)
    axR.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"\nSaved {OUT}")
    print("slide (all alpha=beta=1): " + " -> ".join(f"{r['label']} {r['wc']:.3f}" for r in res))


if __name__ == "__main__":
    main()
