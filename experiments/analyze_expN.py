"""
Analyze Exp N (null model, v(x)=x) and compare its collapse transition against the
existing PT(λ=1) 300k data.

Outputs:
  - per-w collapse fraction with Wilson 95% CIs (both datasets)
  - logistic collapsed ~ w (statsmodels): slope, p-value, McFadden pseudo-R^2
  - w_crit = -b0/b1 with run-level bootstrap 95% CI (n=2000)
  - side-by-side w_crit(null) vs w_crit(PT λ=1 @300k) and behavioral_excess
  - figure docs/expN_figures/expN_vs_PT_comparison.png
  - summary docs/expN_figures/expN_results.md

Sign convention note: the spec writes P(collapse|w) = 1/(1+exp(b0+b1*w)). statsmodels
Logit fits P = 1/(1+exp(-(c0+c1*w))) so b0=-c0, b1=-c1; w_crit (P=0.5) = -b0/b1 =
-c0/c1 either way. Below, `slope` is the statsmodels w-coefficient (c1, negative:
collapse decreases with w); w_crit = -c0/c1.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

EXPN_CSV = "docs/expN_figures/expN_raw_data.csv"
PT_CSV = "docs/expA300k_figures/expA300k_raw_data.csv"
OUTDIR = "docs/expN_figures"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _coerce(df):
    df = df.copy()
    df["w"] = df["w"].astype(float)
    # collapsed may be bool, "True"/"False", or 0/1
    df["collapsed"] = df["collapsed"].map(
        lambda v: 1 if str(v).strip().lower() in ("1", "true", "1.0") else 0).astype(int)
    return df.dropna(subset=["w", "collapsed"])


def fit_logistic(df):
    """Return (model, slope, pval, mcfadden_r2, w_crit). Regularized fallback on
    quasi-separation; pseudo-R^2 is NaN when the regularized path is used."""
    X = sm.add_constant(df[["w"]].astype(float))
    y = df["collapsed"].astype(int)
    try:
        m = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=1000)
        if not np.all(np.isfinite(m.bse)):
            raise np.linalg.LinAlgError("non-finite SE")
        slope, pval = m.params["w"], m.pvalues["w"]
        try:
            r2 = 1.0 - (m.llf / m.llnull)
        except Exception:
            r2 = np.nan
    except Exception:
        m = sm.Logit(y, X).fit_regularized(disp=0, alpha=1e-3, maxiter=2000)
        slope, pval, r2 = m.params["w"], np.nan, np.nan
    b0 = m.params["const"]
    w_crit = -b0 / slope if slope != 0 else np.nan
    return m, slope, pval, r2, w_crit


def bootstrap_wcrit(df, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(df)
    crits = []
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, n, n)]
        if s["collapsed"].nunique() < 2 or s["w"].nunique() < 2:
            continue
        try:
            _, slope, _, _, wc = fit_logistic(s)
            if np.isfinite(wc):
                crits.append(wc)
        except Exception:
            continue
    if not crits:
        return (np.nan, np.nan, np.nan)
    a = np.array(crits)
    return (float(np.median(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def per_w_table(df):
    rows = []
    for w, g in df.groupby("w"):
        k, n = int(g["collapsed"].sum()), len(g)
        lo, hi = wilson_ci(k, n)
        rows.append({"w": w, "k": k, "n": n, "frac": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows).sort_values("w")


def analyze(df, label):
    cells = per_w_table(df)
    print(f"\n=== {label}: per-w collapse fraction (Wilson 95% CI), n={len(df)} ===")
    for _, r in cells.iterrows():
        print(f"  w={r['w']:.2f}: {int(r['k']):2d}/{int(r['n']):2d} = {r['frac']:.2f} "
              f"[{r['lo']:.2f}, {r['hi']:.2f}]")
    m, slope, pval, r2, w_crit = fit_logistic(df)
    wc_med, wc_lo, wc_hi = bootstrap_wcrit(df)
    print(f"  logistic: slope(w)={slope:.3f}  p={pval:.3e}  McFadden R^2={r2:.3f}")
    print(f"  w_crit = {w_crit:.3f}  bootstrap 95% CI [{wc_lo:.3f}, {wc_hi:.3f}]")
    return {"cells": cells, "model": m, "slope": slope, "pval": pval, "r2": r2,
            "w_crit": w_crit, "wc_ci": (wc_lo, wc_hi), "n": len(df)}


def curve(model, wg):
    return model.predict(sm.add_constant(pd.DataFrame({"w": wg})))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(EXPN_CSV):
        print(f"No Exp N data at {EXPN_CSV}")
        sys.exit(0)

    null_df = _coerce(pd.read_csv(EXPN_CSV))
    res_null = analyze(null_df, "NULL MODEL (v(x)=x)")

    pt_res = None
    if os.path.exists(PT_CSV):
        pt_df = _coerce(pd.read_csv(PT_CSV))
        pt_res = analyze(pt_df, "PT λ=1 @300k (expA300k)")

    print("\n" + "=" * 56)
    print("SIDE-BY-SIDE")
    print("=" * 56)
    nlo, nhi = res_null["wc_ci"]
    print(f"  w_crit(null model)    : {res_null['w_crit']:.3f} [{nlo:.3f}, {nhi:.3f}]")
    if pt_res:
        plo, phi = pt_res["wc_ci"]
        print(f"  w_crit(PT λ=1 @300k)  : {pt_res['w_crit']:.3f} [{plo:.3f}, {phi:.3f}]")
        excess = pt_res["w_crit"] - res_null["w_crit"]
        print(f"  behavioral_excess     : {excess:.3f}   (= w_crit(PT) - w_crit(null))")

    # ---------- figure ----------
    wg = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(8, 6))
    # null model: gray dashed
    ax.plot(wg, curve(res_null["model"], wg), color="gray", ls="--", lw=2.2, label="null model v(x)=x")
    ax.errorbar(res_null["cells"]["w"], res_null["cells"]["frac"],
                yerr=[res_null["cells"]["frac"] - res_null["cells"]["lo"],
                      res_null["cells"]["hi"] - res_null["cells"]["frac"]],
                fmt="o", color="gray", capsize=3, alpha=0.7)
    ax.axvline(res_null["w_crit"], color="gray", ls="--", lw=1.5)
    ax.axvspan(nlo, nhi, color="gray", alpha=0.12)

    if pt_res:
        ax.plot(wg, curve(pt_res["model"], wg), color="tab:blue", ls="-", lw=2.2, label="PT λ=1 @300k")
        ax.errorbar(pt_res["cells"]["w"], pt_res["cells"]["frac"],
                    yerr=[pt_res["cells"]["frac"] - pt_res["cells"]["lo"],
                          pt_res["cells"]["hi"] - pt_res["cells"]["frac"]],
                    fmt="s", color="tab:blue", capsize=3, alpha=0.7)
        ax.axvline(pt_res["w_crit"], color="tab:blue", ls="-", lw=1.5)
        ax.axvspan(plo, phi, color="tab:blue", alpha=0.12)

    ax.set_xlabel("w (reward centralization)")
    ax.set_ylabel("P(collapse)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Null model vs PT(λ=1): collapse transition")
    plt.tight_layout()
    fig_path = os.path.join(OUTDIR, "expN_vs_PT_comparison.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved {fig_path}")

    # ---------- results md ----------
    md = [f"# Exp N — null model (v(x)=x) vs PT(λ=1) @300k\n",
          f"Null model: {res_null['n']} runs. Logistic slope(w)={res_null['slope']:.3f}, "
          f"p={res_null['pval']:.3e}, McFadden R²={res_null['r2']:.3f}.",
          f"**w_crit(null) = {res_null['w_crit']:.3f}** [{nlo:.3f}, {nhi:.3f}] (bootstrap, n=2000).\n",
          "## Per-w collapse fraction (null, Wilson 95% CI)\n",
          "| w | collapse | CI |", "|---|---|---|"]
    for _, r in res_null["cells"].iterrows():
        md.append(f"| {r['w']:.2f} | {int(r['k'])}/{int(r['n'])} = {r['frac']:.2f} | [{r['lo']:.2f}, {r['hi']:.2f}] |")
    if pt_res:
        plo, phi = pt_res["wc_ci"]
        excess = pt_res["w_crit"] - res_null["w_crit"]
        md += ["\n## Comparison\n",
               f"- w_crit(null model) = {res_null['w_crit']:.3f} [{nlo:.3f}, {nhi:.3f}]",
               f"- w_crit(PT λ=1 @300k) = {pt_res['w_crit']:.3f} [{plo:.3f}, {phi:.3f}] "
               f"(slope={pt_res['slope']:.3f}, p={pt_res['pval']:.3e}, McFadden R²={pt_res['r2']:.3f}, n={pt_res['n']})",
               f"- **behavioral_excess = {excess:.3f}** (w_crit(PT) − w_crit(null))\n",
               "Figure: [expN_vs_PT_comparison.png](expN_vs_PT_comparison.png)",
               "\nNote: both conditions use λ=1, α=1, β=1 (v(x)=x). The PT(λ=1) dataset is "
               "the no-loss-aversion condition, so behavioral_excess near 0 is the expected "
               "null outcome and serves as a reproducibility/measurement check."]
    else:
        md.append("\n(PT comparison data docs/expA300k_figures/expA300k_raw_data.csv not found.)")
    with open(os.path.join(OUTDIR, "expN_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'expN_results.md')}")


if __name__ == "__main__":
    main()
