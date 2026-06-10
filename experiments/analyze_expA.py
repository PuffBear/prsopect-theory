"""
Experiment A analysis (experiment_spec_v1 sec.A primary + secondary analyses).

Primary:   logistic P(collapsed) ~ b0 + b1*w  ->  w_crit = -b0/b1 (bootstrap CI).
Secondary: (1) per-cell collapse fraction with Wilson CIs,
           (2) sharpness: add w^2 term + 10-90% transition width,
           (3) seed-as-random-effect mixed logistic (if statsmodels supports it).

Usage:
    python experiments/analyze_expA.py --in docs/expA_figures/expA_raw_data.csv
    # pipeline self-test against existing endpoint data (w=0 and w=1):
    python experiments/analyze_expA.py --selftest
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half, center + half)


def fit_logistic(df):
    """P(collapsed) ~ w via Logit, with an L2-regularized fallback.

    With only extreme w-levels the data are quasi-separated and plain MLE has a
    singular Hessian; a small ridge penalty regularizes it. Interior data (many
    w-levels with intermediate fractions) fit cleanly via MLE.
    """
    X = sm.add_constant(df[["w"]].astype(float))
    y = df["collapsed"].astype(int)
    try:
        model = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=500)
        if not np.all(np.isfinite(model.bse)):
            raise np.linalg.LinAlgError("non-finite standard errors")
    except (np.linalg.LinAlgError, Exception):
        model = sm.Logit(y, X).fit_regularized(disp=0, alpha=1e-3, maxiter=1000)
    b0, b1 = model.params["const"], model.params["w"]
    w_crit = -b0 / b1 if b1 != 0 else np.nan
    return model, w_crit


def bootstrap_wcrit(df, n_boot=2000, seed=0):
    """Bootstrap over seeds (resample whole runs) for a w_crit CI."""
    rng = np.random.default_rng(seed)
    crits = []
    n = len(df)
    for _ in range(n_boot):
        samp = df.iloc[rng.integers(0, n, n)]
        if samp["collapsed"].nunique() < 2 or samp["w"].nunique() < 2:
            continue
        try:
            X = sm.add_constant(samp[["w"]].astype(float))
            m = sm.Logit(samp["collapsed"].astype(int), X).fit(disp=0)
            b1 = m.params["w"]
            if b1 != 0:
                crits.append(-m.params["const"] / b1)
        except Exception:
            continue
    if not crits:
        return (np.nan, np.nan, np.nan)
    crits = np.array(crits)
    return (np.median(crits), np.percentile(crits, 2.5), np.percentile(crits, 97.5))


def transition_width(model):
    """Delta-w between P=0.1 and P=0.9 from the logistic fit."""
    b0, b1 = model.params["const"], model.params["w"]
    if b1 == 0:
        return np.nan
    def w_at(p):
        return (np.log(p / (1 - p)) - b0) / b1
    return abs(w_at(0.9) - w_at(0.1))


def sharpness_test(df):
    """Add w^2; report whether quadratic improves fit (LLR p-value).

    Returns (None, nan) when there are too few w-levels (<3) or the fit fails to
    converge (quasi-separation), since a quadratic is undefined in those cases.
    """
    if df["w"].nunique() < 3:
        return None, np.nan
    df = df.copy()
    df["w2"] = df["w"].astype(float) ** 2
    X1 = sm.add_constant(df[["w"]].astype(float))
    X2 = sm.add_constant(df[["w", "w2"]].astype(float))
    try:
        m1 = sm.Logit(df["collapsed"].astype(int), X1).fit(disp=0, method="bfgs", maxiter=500)
        m2 = sm.Logit(df["collapsed"].astype(int), X2).fit(disp=0, method="bfgs", maxiter=500)
        llr = 2 * (m2.llf - m1.llf)
        from scipy.stats import chi2
        return m2, chi2.sf(llr, df=1)
    except Exception:
        return None, np.nan


def run(df, out_prefix):
    df = df.dropna(subset=["w", "collapsed"]).copy()
    df["w"] = df["w"].astype(float)
    df["collapsed"] = df["collapsed"].astype(int)

    print(f"\n=== Experiment A analysis (n={len(df)} runs, "
          f"{df['w'].nunique()} w-levels) ===")

    # Per-cell collapse fraction + Wilson CIs
    cells = []
    for w, g in df.groupby("w"):
        k, n = int(g["collapsed"].sum()), len(g)
        lo, hi = wilson_ci(k, n)
        cells.append({"w": w, "k": k, "n": n, "frac": k / n, "lo": lo, "hi": hi})
    cells = pd.DataFrame(cells).sort_values("w")
    print("\nPer-cell collapse fraction (Wilson 95% CI):")
    for _, r in cells.iterrows():
        print(f"  w={r['w']:.2f}: {int(r['k']):2d}/{int(r['n']):2d} = {r['frac']:.2f} "
              f"[{r['lo']:.2f}, {r['hi']:.2f}]")

    result = {"n": len(df), "n_wlevels": int(df["w"].nunique())}

    if df["collapsed"].nunique() < 2 or df["w"].nunique() < 2:
        print("\n[Not enough variation in collapse/w yet for a logistic fit.]")
        cells.to_csv(f"{out_prefix}_cellfractions.csv", index=False)
        return result

    model, w_crit = fit_logistic(df)
    b1 = model.params["w"]
    ci = model.conf_int().loc["w"]
    width = transition_width(model)
    wc_med, wc_lo, wc_hi = bootstrap_wcrit(df)
    m2, sharp_p = sharpness_test(df)

    print(f"\nPrimary logistic  P(collapse) ~ logit(b0 + b1*w):")
    print(f"  b1 (w) = {b1:.3f}   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]   p={model.pvalues['w']:.3e}")
    print(f"  w_crit (P=0.5) = {w_crit:.3f}   bootstrap median {wc_med:.3f} "
          f"[{wc_lo:.3f}, {wc_hi:.3f}]")
    print(f"  10-90% transition width Deltaw = {width:.3f}")
    print(f"  sharpness: quadratic w^2 term LLR p = {sharp_p:.3f} "
          f"({'adds signal' if sharp_p < 0.05 else 'no added curvature -> monotone'})")

    # Pre-registered success criterion check
    sig = (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)
    narrow = (not np.isnan(width)) and width < 0.4
    print(f"\nPre-registered success: b1 CI excludes 0 = {sig}; Deltaw<0.4 = {narrow}")
    if cells["frac"].std() < 1e-6:
        print("  -> collapse fraction FLAT across w: w-transition FALSIFIED (per spec).")

    result.update({"b1": b1, "b1_ci": (ci[0], ci[1]), "w_crit": w_crit,
                   "w_crit_ci": (wc_lo, wc_hi), "width": width,
                   "sharpness_p": sharp_p, "b1_significant": bool(sig)})

    # Figure: per-cell fractions + logistic curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(cells["w"], cells["frac"],
                yerr=[cells["frac"] - cells["lo"], cells["hi"] - cells["frac"]],
                fmt="o", color="black", capsize=4, label="per-cell fraction (Wilson CI)")
    wg = np.linspace(0, 1, 200)
    pg = model.predict(sm.add_constant(pd.DataFrame({"w": wg})))
    ax.plot(wg, pg, "b-", lw=2, label="logistic fit")
    if not np.isnan(w_crit) and 0 <= w_crit <= 1:
        ax.axvline(w_crit, color="red", ls="--", label=f"w_crit={w_crit:.2f}")
    ax.set_xlabel("w (reward centralization)")
    ax.set_ylabel("P(collapse)")
    ax.set_title(f"Experiment A: collapse transition in w (n={len(df)})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_logistic.png", dpi=150)
    cells.to_csv(f"{out_prefix}_cellfractions.csv", index=False)
    print(f"\nSaved {out_prefix}_logistic.png and {out_prefix}_cellfractions.csv")
    return result


def load_selftest():
    """Existing endpoint data: w=1 (phase4) + w=0 (master decentralized rows)."""
    p4 = pd.read_csv("docs/phase4_figures/phase4_raw_data.csv").rename(
        columns={"Mean S": "mean_S", "Profit": "profit"})
    p4["w"] = 1.0
    mas = pd.read_csv("docs/final_archive/master_seed_dataset.csv").rename(
        columns={"Mean S": "mean_S", "Profit": "profit", "Alpha": "w"})
    dec = mas[mas["w"] == 0.0]
    both = pd.concat([p4[["w", "mean_S", "profit"]], dec[["w", "mean_S", "profit"]]],
                     ignore_index=True)
    both["collapsed"] = ((both["mean_S"] < 10.0) &
                         (both["profit"] <= -128.1 + 1.0)).astype(int)
    return both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="docs/expA_figures/expA_raw_data.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    os.makedirs("docs/expA_figures", exist_ok=True)

    if args.selftest:
        df = load_selftest()
        run(df, "docs/expA_figures/expA_SELFTEST")
    else:
        if not os.path.exists(args.inp):
            print(f"No data yet at {args.inp}")
            sys.exit(0)
        df = pd.read_csv(args.inp)
        run(df, "docs/expA_figures/expA")


if __name__ == "__main__":
    main()
