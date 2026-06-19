"""
Analyze the value-based (QMIX / IQL) supply-chain cells: w_crit(lambda) surface,
the null/curvature/loss-aversion decomposition, monotonicity verdict, and the
comparison series for the generalization figure.

Reads (whichever exist):
  docs/expVB_qmix_orgym_figures/qmix_orgym_raw_data.csv
  docs/expVB_iql_orgym_figures/iql_orgym_raw_data.csv
Outputs per algo:
  - per-lambda logistic fit + bootstrap(2000) w_crit + 95% CI
  - decomposition: curvature shift (lam1@0.88 - null) and
                   loss-aversion excess (lam@0.88 - lam1@0.88)
  - monotonicity verdict (w_crit(7) > w_crit(1), CIs non-overlapping)
  - collapse curves figure + w_crit-vs-lambda figure
  - {algo}_orgym_results.md

Uses the SAME estimator stack as analyze_mappo_orgym / analyze_expN
(fit_logistic with quasi-separation fallback, bootstrap_wcrit n=2000), so the
value-based w_crit lands on the identical statistical footing as PPO/MAPPO.

  python experiments/analyze_qmix_orgym.py            # both algos if present
  python experiments/analyze_qmix_orgym.py --algo qmix
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import (fit_logistic, bootstrap_wcrit, per_w_table, curve,
                          _coerce)

LAMBDAS = [1, 2, 3, 4, 5, 6, 7]
COLORS = {1: "tab:blue", 2: "tab:cyan", 3: "tab:orange", 4: "tab:olive",
          5: "tab:green", 6: "tab:purple", 7: "tab:red"}

CELLS = {
    "qmix": dict(csv="docs/expVB_qmix_orgym_figures/qmix_orgym_raw_data.csv",
                 outdir="docs/expVB_qmix_orgym_figures", label="QMIX"),
    "iql": dict(csv="docs/expVB_iql_orgym_figures/iql_orgym_raw_data.csv",
                outdir="docs/expVB_iql_orgym_figures", label="IQL"),
}


def _load(csv):
    # keep_default_na=False: the condition value "null" is a literal label, NOT a
    # missing value -- pandas' default na_values would silently turn the 100 null-
    # baseline rows into NaN and drop them from the decomposition.
    df = pd.read_csv(csv, keep_default_na=False)
    df["lambda_loss"] = df["lambda_loss"].astype(float).round().astype(int)
    df["mean_S"] = pd.to_numeric(df["mean_S"], errors="coerce")
    # condition column distinguishes the null baseline (alpha=beta=1) from the
    # curvature point (lambda=1, alpha=beta=0.88) -- both have lambda_loss==1, so
    # we must split on `condition`, not lambda, for the null row.
    return _coerce(df)


def _fit(sub):
    if len(sub) < 5 or sub["collapsed"].nunique() < 2:
        return None
    m, slope, pval, r2, wc = fit_logistic(sub)
    lo, hi = bootstrap_wcrit(sub)[1:]
    return dict(n=len(sub), cells=per_w_table(sub), model=m, slope=slope,
                pval=pval, r2=r2, w_crit=wc, ci_lo=lo, ci_hi=hi)


def analyze_algo(algo):
    spec = CELLS[algo]
    if not os.path.exists(spec["csv"]):
        print(f"[skip {spec['label']}] no data at {spec['csv']}")
        return None
    os.makedirs(spec["outdir"], exist_ok=True)
    df = _load(spec["csv"])
    null_df = df[df["condition"] == "null"]
    sweep_df = df[df["condition"] != "null"]   # curvature (lam=1) + loss_aversion

    # null baseline (alpha=beta=1, lambda=1)
    null = _fit(null_df)
    # per-lambda on the alpha=beta=0.88 sweep (lambda=1 is the curvature point)
    perlam = {lam: _fit(sweep_df[sweep_df["lambda_loss"] == lam]) for lam in LAMBDAS}
    perlam = {k: v for k, v in perlam.items() if v is not None}

    # --- print table -----------------------------------------------------
    print("\n" + "=" * 76)
    print(f"  {spec['label']} -- or-gym supply chain  (n={len(df)})")
    print("=" * 76)
    print(f"{'condition':>14} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | "
          f"{'slope':>8} | {'p':>9} | {'n':>4}")
    print("-" * 76)
    if null:
        print(f"{'null (l=1,a=1)':>14} | {null['w_crit']:>7.3f} | {null['ci_lo']:>6.3f} | "
              f"{null['ci_hi']:>6.3f} | {null['slope']:>8.2f} | {null['pval']:>9.2e} | {null['n']:>4}")
    for lam in LAMBDAS:
        if lam in perlam:
            r = perlam[lam]
            tag = f"curv (l=1)" if lam == 1 else f"loss l={lam}"
            print(f"{tag:>14} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | {r['ci_hi']:>6.3f} | "
                  f"{r['slope']:>8.2f} | {r['pval']:>9.2e} | {r['n']:>4}")
    print("=" * 76)

    # --- decomposition ---------------------------------------------------
    curv_shift = loss_excess = None
    if null and 1 in perlam:
        curv_shift = perlam[1]["w_crit"] - null["w_crit"]
        print(f"  curvature shift   (w_crit[l1@0.88] - w_crit[null]) = {curv_shift:+.3f}")
    if 1 in perlam and 7 in perlam:
        loss_excess = perlam[7]["w_crit"] - perlam[1]["w_crit"]
        print(f"  loss-aversion excess (w_crit[l7@0.88] - w_crit[l1@0.88]) = {loss_excess:+.3f}")

    # --- monotonicity verdict (the pre-registered success test) ----------
    verdict = "INSUFFICIENT DATA"
    if 1 in perlam and 7 in perlam:
        r1, r7 = perlam[1], perlam[7]
        rises = r7["w_crit"] > r1["w_crit"]
        # non-overlapping 95% CIs: lower bound of l7 above upper bound of l1
        non_overlap = r7["ci_lo"] > r1["ci_hi"]
        # monotone non-decreasing across the available sweep
        seq = [perlam[l]["w_crit"] for l in sorted(perlam)]
        monotone = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
        if rises and non_overlap:
            verdict = "PASS (w_crit rises, CIs non-overlapping)"
        elif rises:
            verdict = "DIRECTIONAL (rises, but CIs overlap -> scope as exploratory)"
        else:
            verdict = "FALSIFIED (flat/decreasing)"
        print(f"  monotone non-decreasing over swept lambda: {monotone}")
    print(f"  PRE-REGISTERED VERDICT: {verdict}")

    _figures(spec, null, perlam)
    _writeup(spec, df, null, perlam, curv_shift, loss_excess, verdict)
    return dict(label=spec["label"], perlam=perlam, null=null,
                curv_shift=curv_shift, loss_excess=loss_excess, verdict=verdict)


def _figures(spec, null, perlam):
    sns.set_style("whitegrid")
    wg = np.linspace(0.3, 1.0, 200)

    # collapse curves by lambda
    fig, ax = plt.subplots(figsize=(9, 6))
    for lam in LAMBDAS:
        if lam not in perlam:
            continue
        r = perlam[lam]; c = COLORS[lam]
        ax.plot(wg, curve(r["model"], wg), color=c, lw=2.2, label=f"lambda={lam}")
        ax.errorbar(r["cells"]["w"], r["cells"]["frac"],
                    yerr=[r["cells"]["frac"] - r["cells"]["lo"],
                          r["cells"]["hi"] - r["cells"]["frac"]],
                    fmt="o", color=c, capsize=3, alpha=0.6)
        ax.axvline(r["w_crit"], color=c, ls="--", lw=1.0)
        ax.axvspan(r["ci_lo"], r["ci_hi"], color=c, alpha=0.07)
    if null:
        ax.plot(wg, curve(null["model"], wg), color="black", lw=1.6, ls=":",
                label="null (l=1,a=1)")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
    ax.set_title(f"{spec['label']} on or-gym: collapse transition by lambda")
    ax.legend(title="loss aversion", ncol=2)
    plt.tight_layout()
    f1 = os.path.join(spec["outdir"], f"{spec['label'].lower()}_orgym_collapse_curves.png")
    plt.savefig(f1, dpi=150); print(f"  saved {f1}")
    plt.close(fig)

    # w_crit vs lambda
    lams = sorted(perlam)
    if lams:
        fig, ax = plt.subplots(figsize=(8, 5))
        ys = [perlam[l]["w_crit"] for l in lams]
        lo = [perlam[l]["w_crit"] - perlam[l]["ci_lo"] for l in lams]
        hi = [perlam[l]["ci_hi"] - perlam[l]["w_crit"] for l in lams]
        ax.errorbar(lams, ys, yerr=[lo, hi], fmt="o-", color="tab:red",
                    capsize=5, lw=2, markersize=8, label=spec["label"])
        if null:
            ax.axhline(null["w_crit"], color="black", ls=":", lw=1.5,
                       label="null baseline")
            ax.axhspan(null["ci_lo"], null["ci_hi"], color="gray", alpha=0.12)
        ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
        ax.set_xticks(LAMBDAS)
        ax.set_title(f"w_crit vs lambda: {spec['label']} (or-gym)")
        ax.legend()
        plt.tight_layout()
        f2 = os.path.join(spec["outdir"], f"{spec['label'].lower()}_orgym_wcrit_vs_lambda.png")
        plt.savefig(f2, dpi=150); print(f"  saved {f2}")
        plt.close(fig)


def _writeup(spec, df, null, perlam, curv_shift, loss_excess, verdict):
    md = [f"# {spec['label']} on or-gym -- value-based w_crit decomposition\n",
          f"n = {len(df)} runs. Bootstrap n=2000. Estimator identical to "
          f"PPO/MAPPO (analyze_expN.fit_logistic).\n",
          "| condition | w_crit | 95% CI | slope | p | n |",
          "|---|---|---|---|---|---|"]
    if null:
        md.append(f"| null (l=1,a=1) | {null['w_crit']:.3f} | "
                  f"[{null['ci_lo']:.3f}, {null['ci_hi']:.3f}] | {null['slope']:.2f} | "
                  f"{null['pval']:.2e} | {null['n']} |")
    for lam in LAMBDAS:
        if lam in perlam:
            r = perlam[lam]
            tag = "curvature (l=1, a=0.88)" if lam == 1 else f"loss aversion l={lam}"
            md.append(f"| {tag} | {r['w_crit']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                      f"{r['slope']:.2f} | {r['pval']:.2e} | {r['n']} |")
    md.append("")
    if curv_shift is not None:
        md.append(f"- **Curvature shift** (w_crit[l1@0.88] - w_crit[null]) = **{curv_shift:+.3f}**")
    if loss_excess is not None:
        md.append(f"- **Loss-aversion excess** (w_crit[l7@0.88] - w_crit[l1@0.88]) = **{loss_excess:+.3f}**")
    md.append(f"- **Pre-registered verdict:** {verdict}")
    path = os.path.join(spec["outdir"], f"{spec['label'].lower()}_orgym_results.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["qmix", "iql"], default=None,
                    help="analyze one cell; default analyzes whichever CSVs exist")
    args = ap.parse_args()
    algos = [args.algo] if args.algo else ["qmix", "iql"]
    for a in algos:
        analyze_algo(a)


if __name__ == "__main__":
    main()
