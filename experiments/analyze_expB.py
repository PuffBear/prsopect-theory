"""
Experiment B analysis — encodes docs/expB_preregistration.md (frozen 2026-06-03).

Primary model:  P(collapsed) ~ logit(b0 + b1*w + b2*log(λ) + b3*(w*log(λ)))
Primary test:   b3 (Wald CI + LRT vs no-interaction model).
Estimand:       w_crit(λ) = -(b0 + b2 logλ)/(b1 + b3 logλ), bootstrap CI band.
Decision (§5):  decide_outcome() returns "TILTED" or "VERTICAL" by the
                pre-committed rule (b3 significant AND |Δw_crit|>=0.1 -> TILTED;
                else VERTICAL, reported with equal prominence).
Headline (§4):  empirical P(collapse) heatmap over (w,λ) + w_crit(λ) overlay.

Usage:
    python experiments/analyze_expB.py --in docs/expB_figures/expB_raw_data.csv
    python experiments/analyze_expB.py --selftest   # 2-w endpoints from existing data
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
from scipy.stats import chi2

# Frozen thresholds (docs/expB_preregistration.md §5)
DELTA_WCRIT_THRESHOLD = 0.10
LAMBDA_MIN, LAMBDA_MAX = 1.0, 10.0


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half, center + half)


def _fit(y, X):
    """Logit MLE with L2-regularized fallback on quasi-separation."""
    try:
        m = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=800)
        if not np.all(np.isfinite(m.bse)):
            raise np.linalg.LinAlgError("non-finite SE")
        return m, True
    except Exception:
        return sm.Logit(y, X).fit_regularized(disp=0, alpha=1e-3, maxiter=2000), False


def fit_full(df):
    d = df.copy()
    d["logL"] = np.log(d["lambda"].astype(float))
    d["inter"] = d["w"].astype(float) * d["logL"]
    X = sm.add_constant(d[["w", "logL", "inter"]].astype(float))
    return _fit(d["collapsed"].astype(int), X), d


def w_crit_of_lambda(params, lam):
    b0, b1, b2, b3 = params["const"], params["w"], params["logL"], params["inter"]
    logL = np.log(lam)
    denom = b1 + b3 * logL
    return np.where(np.abs(denom) < 1e-9, np.nan, -(b0 + b2 * logL) / denom)


def lrt_interaction(df):
    """LRT: full vs reduced (no interaction)."""
    d = df.copy()
    d["logL"] = np.log(d["lambda"].astype(float))
    d["inter"] = d["w"].astype(float) * d["logL"]
    y = d["collapsed"].astype(int)
    try:
        full = sm.Logit(y, sm.add_constant(d[["w", "logL", "inter"]].astype(float))).fit(disp=0, method="bfgs", maxiter=800)
        red = sm.Logit(y, sm.add_constant(d[["w", "logL"]].astype(float))).fit(disp=0, method="bfgs", maxiter=800)
        return chi2.sf(2 * (full.llf - red.llf), df=1)
    except Exception:
        return np.nan


def bootstrap_boundary(df, n_boot=1500, seed=0):
    """Cluster-bootstrap over seeds; return Δw_crit dist and per-λ w_crit bands."""
    rng = np.random.default_rng(seed)
    seeds = df["seed"].unique()
    lam_grid = np.array(sorted(df["lambda"].unique()), dtype=float)
    wc_samples = []   # rows: w_crit at each lam_grid point
    dwc_samples = []
    for _ in range(n_boot):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        samp = pd.concat([df[df["seed"] == s] for s in chosen], ignore_index=True)
        if samp["collapsed"].nunique() < 2:
            continue
        try:
            (m, _), _ = fit_full(samp)
            wc = w_crit_of_lambda(m.params, lam_grid)
            wc_samples.append(wc)
            dwc_samples.append(w_crit_of_lambda(m.params, LAMBDA_MAX) - w_crit_of_lambda(m.params, LAMBDA_MIN))
        except Exception:
            continue
    wc_samples = np.array(wc_samples)
    dwc = np.array([d for d in dwc_samples if np.isfinite(d)])
    band = {}
    if len(wc_samples):
        for j, lam in enumerate(lam_grid):
            col = wc_samples[:, j]
            col = col[np.isfinite(col)]
            if len(col):
                band[lam] = (np.median(col), np.percentile(col, 2.5), np.percentile(col, 97.5))
    return band, dwc


def decide_outcome(b3_ci, lrt_p, dwc_point, dwc_ci):
    """Frozen §5 rule. Returns ('TILTED'|'VERTICAL', reason str)."""
    b3_sig = (b3_ci[0] > 0 and b3_ci[1] > 0) or (b3_ci[0] < 0 and b3_ci[1] < 0)
    lrt_sig = (not np.isnan(lrt_p)) and lrt_p < 0.05
    practical = (not np.isnan(dwc_point)) and abs(dwc_point) >= DELTA_WCRIT_THRESHOLD
    dwc_sig = dwc_ci is not None and ((dwc_ci[0] > 0 and dwc_ci[1] > 0) or (dwc_ci[0] < 0 and dwc_ci[1] < 0))
    if (b3_sig and lrt_sig) and (practical and dwc_sig):
        return "TILTED", ("behavioral phase transition: b3 significant (Wald+LRT) "
                          f"and |Δw_crit|={abs(dwc_point):.3f}>={DELTA_WCRIT_THRESHOLD} with CI excluding 0")
    return "VERTICAL", ("centralization dominates; loss aversion does not move the boundary "
                        f"(b3_sig={b3_sig}, lrt_sig={lrt_sig}, |Δw_crit|="
                        f"{abs(dwc_point) if np.isfinite(dwc_point) else float('nan'):.3f}, dwc_CI_excl0={dwc_sig})")


def run(df, out_prefix):
    df = df.dropna(subset=["w", "lambda", "collapsed"]).copy()
    df["w"] = df["w"].astype(float)
    df["lambda"] = df["lambda"].astype(float)
    df["collapsed"] = df["collapsed"].astype(int)
    ws = sorted(df["w"].unique())
    lams = sorted(df["lambda"].unique())
    print(f"\n=== Experiment B analysis (n={len(df)}, {len(ws)} w x {len(lams)} λ) ===")
    print(f"Global collapse rate: {df['collapsed'].mean():.1%}")

    # Per-cell Wilson table + empirical heatmap matrix
    heat = np.full((len(lams), len(ws)), np.nan)
    print("\nPer-(w,λ) collapse fraction (Wilson 95% CI):")
    for i, lam in enumerate(lams):
        for j, w in enumerate(ws):
            g = df[(df["w"] == w) & (df["lambda"] == lam)]
            if len(g):
                k, n = int(g["collapsed"].sum()), len(g)
                heat[i, j] = k / n
                lo, hi = wilson_ci(k, n)
                print(f"  w={w:.2f} λ={lam:>4.1f}: {k:2d}/{n:2d}={k/n:.2f} [{lo:.2f},{hi:.2f}]")

    if df["collapsed"].nunique() < 2 or len(ws) < 2 or len(lams) < 2:
        print("\n[Insufficient variation for the 2-D logistic surface yet.]")
        return {"status": "insufficient"}

    (model, clean), d = fit_full(df)
    p = model.params
    try:
        ci = model.conf_int()
        b3_ci = (ci.loc["inter", 0], ci.loc["inter", 1])
    except Exception:
        b3_ci = (np.nan, np.nan)
    lrt_p = lrt_interaction(df)
    band, dwc = bootstrap_boundary(df)
    dwc_point = w_crit_of_lambda(p, LAMBDA_MAX) - w_crit_of_lambda(p, LAMBDA_MIN)
    dwc_ci = (np.percentile(dwc, 2.5), np.percentile(dwc, 97.5)) if len(dwc) else None

    print(f"\nLogistic coefficients (MLE clean={clean}):")
    for name in ["const", "w", "logL", "inter"]:
        print(f"  {name:6s} = {p[name]:.4f}")
    print(f"\nPRIMARY b3 (w x logλ): {p['inter']:.4f}  95% CI [{b3_ci[0]:.4f}, {b3_ci[1]:.4f}]  LRT p={lrt_p:.4f}")
    print(f"Δw_crit (λ=10 minus λ=1): {dwc_point:.4f}" +
          (f"  bootstrap 95% CI [{dwc_ci[0]:.4f}, {dwc_ci[1]:.4f}]" if dwc_ci else ""))

    outcome, reason = decide_outcome(b3_ci, lrt_p, dwc_point, dwc_ci)
    print(f"\n>>> PRE-REGISTERED OUTCOME: {outcome}")
    print(f"    {reason}")
    if outcome == "VERTICAL":
        print('    Reported with equal prominence as: "Reward centralization governs the')
        print('    collapse transition; loss aversion does not move it." (prereg §5)')

    # Headline figure: empirical heatmap + w_crit(λ) overlay
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1, extent=[min(ws), max(ws), min(lams), max(lams)])
    fig.colorbar(im, ax=ax, label="empirical P(collapse)")
    lam_line = np.linspace(min(lams), max(lams), 100)
    wc_line = w_crit_of_lambda(p, lam_line)
    ax.plot(wc_line, lam_line, "k-", lw=2.5, label="w_crit(λ) fit")
    if band:
        bl = sorted(band.keys())
        lo = [band[l][1] for l in bl]; hi = [band[l][2] for l in bl]
        ax.fill_betweenx(bl, lo, hi, color="k", alpha=0.15, label="95% CI")
    ax.set_xlabel("w (reward centralization)")
    ax.set_ylabel("λ (loss aversion)")
    ax.set_xlim(min(ws), max(ws))
    ax.set_title(f"Exp B: collapse surface — OUTCOME: {outcome}\nΔw_crit={dwc_point:.3f}, b3 p(LRT)={lrt_p:.3f}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_surface.png", dpi=150)
    print(f"\nSaved {out_prefix}_surface.png")
    return {"outcome": outcome, "b3": p["inter"], "b3_ci": b3_ci, "lrt_p": lrt_p,
            "dwc": dwc_point, "dwc_ci": dwc_ci}


def load_selftest():
    """Existing data as a degenerate 2-w grid: w=1 (phase4, λ∈{1..10}) + w=0 (decentralized)."""
    p4 = pd.read_csv("docs/phase4_figures/phase4_raw_data.csv").rename(
        columns={"Mean S": "mean_S", "Profit": "profit", "Lambda": "lambda", "Seed": "seed"})
    p4["w"] = 1.0
    mas = pd.read_csv("docs/final_archive/master_seed_dataset.csv").rename(
        columns={"Mean S": "mean_S", "Profit": "profit", "Alpha": "w", "Lambda": "lambda", "Seed": "seed"})
    dec = mas[mas["w"] == 0.0]
    both = pd.concat([p4[["w", "lambda", "seed", "mean_S", "profit"]],
                      dec[["w", "lambda", "seed", "mean_S", "profit"]]], ignore_index=True)
    both["collapsed"] = ((both["mean_S"] < 10.0) & (both["profit"] <= -128.1 + 1.0)).astype(int)
    return both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="docs/expB_figures/expB_raw_data.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    os.makedirs("docs/expB_figures", exist_ok=True)
    if args.selftest:
        run(load_selftest(), "docs/expB_figures/expB_SELFTEST")
    elif not os.path.exists(args.inp):
        print(f"No Exp B data yet at {args.inp}")
    else:
        run(pd.read_csv(args.inp), "docs/expB_figures/expB")


if __name__ == "__main__":
    main()
