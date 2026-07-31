"""
Part-2 reanalysis (NO new training) -- robustness of the lambda=1->7 rightward shift in
w_crit under nonparametric estimation, seed-paired uncertainty, and raw (un-fitted) data.

Answers each Part-2 ask on the PRIMARY supply-chain sweep (full lambda in {1..7}, the same
20 seeds shared across every lambda), and mirrors the core ones on MPE (occupancy) and LBF
(food collected):

  (1) Isotonic (nonparametric monotone) regression alongside logistic; show the
      substantive lambda=1->7 shift is similar under both estimators.
  (2) Seed-BLOCKED / paired bootstrap for w_crit and for the lambda=1->7 DIFFERENCE
      (resampling unit = seed, the same seed drawn across all lambda in a replicate).
  (3) Raw collapse COUNTS at every w (k/n), not only fitted curves.
  (4) Continuous metric distributions around the boundary (supply: base-stock vs profit;
      MPE: occupancy; LBF: food/mean_reward), so the reader can judge whether the object
      is bimodal or a smooth threshold-defined behavioral CROSSING.
  (5) Explicit METHOD reporting: #bootstrap replicates, resampling unit, separation
      handling, and how difference CIs were computed.

Estimators reused verbatim from the paper's pipeline:
  analyze_expN.fit_logistic / wilson_ci / per_w_table / curve  (logistic + Wilson)
  analyze_firth_lambda_sweep.firth_logit                       (Firth for separated cells)
Isotonic via sklearn.isotonic.IsotonicRegression (monotone DECREASING in w:
P(collapse) falls as centralization rises).

w_crit definitions:
  logistic:  -b0/b1
  Firth:     -b0/b1 (penalized)
  isotonic:  smallest w at which the fitted monotone P(collapse) drops below 0.5
             (linear interpolation between the two bracketing grid points; NaN if the
             curve never crosses 0.5 within the grid).

Run:
  python experiments/reanalysis_controls.py            # all substrates
  python experiments/reanalysis_controls.py --substrate supply
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import wilson_ci, fit_logistic, curve, _coerce
from analyze_firth_lambda_sweep import firth_logit

# --- Method constants (reported in the output, single source of truth) -------
N_BOOT = 2000
BOOT_UNIT = "seed (seed-blocked / paired: one seed index resampled across all lambda per replicate)"
SEP_HANDLING = ("Firth penalized logistic (Jeffreys-prior) for separated/steep cells; "
                "logistic MLE falls back to L2-regularized fit on non-finite SE; isotonic "
                "is separation-free by construction. Bootstrap replicates whose resample "
                "is degenerate (collapsed constant or <2 distinct w) are skipped and the "
                "count of usable replicates is reported.")
DIFF_CI = ("Difference w_crit(lambda=1) - w_crit(lambda=7) recomputed within each paired "
           "bootstrap replicate (same resampled seeds used for both lambda), 2.5/97.5 "
           "percentiles of the replicate differences.")

OUTDIR = "docs/reanalysis_controls"


# --- Substrate registry ------------------------------------------------------
# Each entry: how to load the full-lambda raw data + which continuous metrics to plot.

def _load_supply():
    parts = [pd.read_csv("docs/expB_reduced_figures/expB_reduced_raw_data.csv"),
             pd.read_csv("docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv")]
    d = pd.concat(parts, ignore_index=True)
    return d, ["mean_S", "profit"]


def _relabel_mpe(d):
    """Recompute collapse from raw occupancy with the CALIBRATED threshold, exactly as
    analyze_mpe does (the stored `collapsed` column is under an un-recalibrated theta)."""
    try:
        from env.mpe_coop_nav import load_occupancy_threshold
        theta = float(load_occupancy_threshold())
    except Exception:
        theta = 0.164   # MAPPO calibration midpoint (w=0 0.053 / w=1 0.276)
    d = d.copy()
    d["mean_occupancy"] = pd.to_numeric(d["mean_occupancy"], errors="coerce")
    d["collapsed"] = (d["mean_occupancy"] < theta).astype(int)
    return d, theta


def _load_mpe():
    d = pd.read_csv("docs/expMAPPO_mpe_figures/mappo_mpe_lambda_sweep_raw_data.csv")
    return d, ["mean_occupancy", "mean_reward"]


def _load_lbf():
    # LBF's boundary sits at LOW w; the main CSV floors at w=0.3, so the low-w and
    # w=0.90 extension files must be merged (same as analyze_mappo_lbf._load), or the
    # boundary falls off the bottom of the grid and there is no collapse variation to fit.
    base = "docs/expMAPPO_lbf_figures/"
    files = ["mappo_lbf_raw_data.csv", "mappo_lbf_loww_raw_data.csv",
             "mappo_lbf_highw_raw_data.csv"]
    frames = [pd.read_csv(base + f, keep_default_na=False)
              for f in files if os.path.exists(base + f)]
    d = pd.concat(frames, ignore_index=True)
    for c in ("mean_reward", "w", "lambda_loss"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["mean_reward", "w", "lambda_loss"]).copy()
    return d, ["mean_reward"]


def _relabel_lbf(d):
    """Recompute collapse with the recalibrated theta = midpoint of the collapsed regime
    (highest-lambda, lowest-w mean_reward) and the coordinated regime (null / highest-w),
    exactly as analyze_mappo_lbf.recalibrate_theta (the stored `collapsed` is near-all-0
    under the un-recalibrated threshold)."""
    d = d.copy()
    d["lambda_loss"] = pd.to_numeric(d["lambda_loss"], errors="coerce").round()
    w_min, w_max = d["w"].min(), d["w"].max()
    hi_lam = d["lambda_loss"].max()
    collapsed_ref = d[(d["lambda_loss"] == hi_lam) & (d["w"] == w_min)]["mean_reward"].mean()
    null = d[d["condition"] == "null"] if "condition" in d.columns else d.iloc[0:0]
    coord_ref = (null[null["w"] == w_max]["mean_reward"].mean() if not null.empty
                 else d[d["w"] == w_max]["mean_reward"].mean())
    theta = (float((collapsed_ref + coord_ref) / 2.0)
             if np.isfinite(collapsed_ref) and np.isfinite(coord_ref) else 0.62)
    d["collapsed"] = (d["mean_reward"] < theta).astype(int)
    return d, theta


SUBSTRATES = {
    "supply": dict(loader=_load_supply, title="Supply chain (PPO)", relabel=None),
    "mpe":    dict(loader=_load_mpe,    title="MPE (MAPPO)", relabel=_relabel_mpe),
    "lbf":    dict(loader=_load_lbf,    title="LBF (MAPPO)", relabel=_relabel_lbf),
}


# --- Prep --------------------------------------------------------------------

def prep(d):
    d = d.copy()
    d["lambda_loss"] = d["lambda_loss"].astype(float).round().astype(int)
    d = _coerce(d)                      # coerces w (float) + collapsed (0/1), drops NaN
    if "seed" in d.columns:
        d["seed"] = d["seed"].astype(int)
    return d


# --- Isotonic ----------------------------------------------------------------

def isotonic_curve_and_wcrit(sub, grid):
    """Monotone-decreasing isotonic fit of collapsed ~ w; return (p_on_grid, w_crit)."""
    x = sub["w"].to_numpy(float)
    y = sub["collapsed"].to_numpy(float)
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    ir.fit(x, y)
    p = ir.predict(grid)
    # w_crit = first grid crossing of 0.5 (monotone decreasing -> unique region)
    wcrit = np.nan
    for i in range(len(grid) - 1):
        if p[i] >= 0.5 >= p[i + 1] and p[i] != p[i + 1]:
            frac = (p[i] - 0.5) / (p[i] - p[i + 1])
            wcrit = grid[i] + frac * (grid[i + 1] - grid[i])
            break
    return p, wcrit


def logistic_wcrit(sub):
    _m, slope, _p, _r2, wc = fit_logistic(sub)
    return wc


def firth_wcrit(sub):
    r = firth_logit(sub["w"].to_numpy(float), sub["collapsed"].to_numpy(int))
    return r["wcrit"]


# --- Seed-paired bootstrap ---------------------------------------------------

def paired_bootstrap(d, lam_lo, lam_hi, n_boot=N_BOOT, seed=0):
    """Seed-blocked paired bootstrap. Resample seeds ONCE per replicate and apply the
    SAME seed multiset to both lambda, so w_crit(lam_lo), w_crit(lam_hi) and their
    difference share resampling noise (paired). Returns per-lambda percentile CIs +
    the difference CI, using Firth w_crit (stable under separation)."""
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(d["seed"].unique()))
    lo = {s: d[(d.lambda_loss == lam_lo) & (d.seed == s)] for s in seeds}
    hi = {s: d[(d.lambda_loss == lam_hi) & (d.seed == s)] for s in seeds}

    wc_lo, wc_hi, wc_diff = [], [], []
    usable = 0
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        blo = pd.concat([lo[s] for s in pick], ignore_index=True)
        bhi = pd.concat([hi[s] for s in pick], ignore_index=True)
        if (blo["collapsed"].nunique() < 2 or blo["w"].nunique() < 2 or
                bhi["collapsed"].nunique() < 2 or bhi["w"].nunique() < 2):
            continue
        a = firth_wcrit(blo)
        b = firth_wcrit(bhi)
        if np.isfinite(a) and np.isfinite(b):
            wc_lo.append(a); wc_hi.append(b); wc_diff.append(a - b)
            usable += 1

    def pct(v):
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return (np.nan, np.nan, np.nan)
        return (float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(lo=pct(wc_lo), hi=pct(wc_hi), diff=pct(wc_diff), usable=usable)


def unpaired_bootstrap(d, lam_lo, lam_hi, n_boot=N_BOOT, seed=0):
    """Unpaired comparison: resample lambda_lo and lambda_hi seeds INDEPENDENTLY
    (no shared seed multiset), so the difference CI does not benefit from paired-resampling
    noise cancellation. Same Firth w_crit estimator as paired_bootstrap; only the resampling
    scheme differs."""
    rng = np.random.default_rng(seed)
    seeds_lo = np.array(sorted(d[d.lambda_loss == lam_lo]["seed"].unique()))
    seeds_hi = np.array(sorted(d[d.lambda_loss == lam_hi]["seed"].unique()))
    lo = {s: d[(d.lambda_loss == lam_lo) & (d.seed == s)] for s in seeds_lo}
    hi = {s: d[(d.lambda_loss == lam_hi) & (d.seed == s)] for s in seeds_hi}

    wc_lo, wc_hi, wc_diff = [], [], []
    usable = 0
    for _ in range(n_boot):
        pick_lo = rng.choice(seeds_lo, size=len(seeds_lo), replace=True)
        pick_hi = rng.choice(seeds_hi, size=len(seeds_hi), replace=True)
        blo = pd.concat([lo[s] for s in pick_lo], ignore_index=True)
        bhi = pd.concat([hi[s] for s in pick_hi], ignore_index=True)
        if (blo["collapsed"].nunique() < 2 or blo["w"].nunique() < 2 or
                bhi["collapsed"].nunique() < 2 or bhi["w"].nunique() < 2):
            continue
        a = firth_wcrit(blo)
        b = firth_wcrit(bhi)
        if np.isfinite(a) and np.isfinite(b):
            wc_lo.append(a); wc_hi.append(b); wc_diff.append(a - b)
            usable += 1

    def pct(v):
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return (np.nan, np.nan, np.nan)
        return (float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(lo=pct(wc_lo), hi=pct(wc_hi), diff=pct(wc_diff), usable=usable)


# --- Per-substrate reanalysis ------------------------------------------------

def analyze_substrate(key):
    cfg = SUBSTRATES[key]
    d, metrics = cfg["loader"]()
    theta = None
    if cfg.get("relabel") is not None:
        d, theta = cfg["relabel"](d)     # recompute collapse with recalibrated theta
    d = prep(d)
    lams = sorted(d["lambda_loss"].unique())
    lam_lo, lam_hi = lams[0], lams[-1]
    grid = np.linspace(float(d["w"].min()), float(d["w"].max()), 401)

    # (1) three estimators of w_crit per lambda
    rows = []
    iso_curves = {}
    for lam in lams:
        sub = d[d.lambda_loss == lam]
        if sub["collapsed"].nunique() < 2 or sub["w"].nunique() < 2:
            rows.append(dict(lam=lam, n=len(sub), logistic=np.nan, firth=np.nan,
                             isotonic=np.nan, degenerate=True, in_grid=False))
            iso_curves[lam] = isotonic_curve_and_wcrit(sub, grid)[0]
            continue
        p_iso, wc_iso = isotonic_curve_and_wcrit(sub, grid)
        iso_curves[lam] = p_iso
        wc_firth = firth_wcrit(sub)
        w_max = float(sub["w"].max())
        rows.append(dict(lam=lam, n=len(sub), logistic=logistic_wcrit(sub),
                         firth=wc_firth, isotonic=wc_iso, degenerate=False,
                         # in_grid: is the (Firth) boundary inside the tested w-range?
                         # w_crit beyond [min,max] is EXTRAPOLATED, flagged in the report.
                         in_grid=bool(np.isfinite(wc_firth) and
                                      grid[0] - 0.05 <= wc_firth <= w_max + 0.05)))
    est = pd.DataFrame(rows)

    # (2) seed-paired bootstrap for the endpoints + the lambda_lo->lambda_hi difference,
    # plus an UNPAIRED comparison to test whether pairing understates uncertainty.
    boot = paired_bootstrap(d, lam_lo, lam_hi) if "seed" in d.columns else None
    boot_unpaired = unpaired_bootstrap(d, lam_lo, lam_hi) if "seed" in d.columns else None

    # (3) raw counts per (lambda, w)
    counts = []
    for lam in lams:
        for w, g in d[d.lambda_loss == lam].groupby("w"):
            k, n = int(g["collapsed"].sum()), len(g)
            wl, wh = wilson_ci(k, n)
            counts.append(dict(lam=lam, w=w, k=k, n=n, frac=k / n, lo=wl, hi=wh))
    counts = pd.DataFrame(counts)

    return dict(key=key, title=cfg["title"], d=d, metrics=metrics, lams=lams,
                lam_lo=lam_lo, lam_hi=lam_hi, grid=grid, est=est, boot=boot,
                boot_unpaired=boot_unpaired, counts=counts, iso_curves=iso_curves, theta=theta)


# --- Plots -------------------------------------------------------------------

def plot_estimator_overlay(R, outdir):
    """Isotonic vs logistic vs Firth: w_crit(lambda) three ways, plus raw fractions."""
    d, lams, grid = R["d"], R["lams"], R["grid"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.2))

    cmap = plt.get_cmap("viridis")
    for i, lam in enumerate(lams):
        c = cmap(i / max(len(lams) - 1, 1))
        sub = d[d.lambda_loss == lam]
        cw = R["counts"][R["counts"].lam == lam].sort_values("w")
        ax0.plot(cw["w"], cw["frac"], "o", color=c, alpha=0.5, ms=5)
        ax0.plot(grid, R["iso_curves"][lam], color=c, lw=1.6, alpha=0.9)
        if sub["collapsed"].nunique() >= 2 and sub["w"].nunique() >= 2:
            m, *_ = fit_logistic(sub)
            ax0.plot(grid, curve(m, grid), color=c, lw=1.2, ls="--", alpha=0.7)
    ax0.set_title(f"{R['title']}: isotonic (solid) vs logistic (dashed), raw pts")
    ax0.set_xlabel("w"); ax0.set_ylabel("P(collapse)"); ax0.set_ylim(-0.05, 1.05)

    e = R["est"]
    ax1.plot(e["lam"], e["logistic"], "s--", label="logistic w_crit", color="tab:blue")
    ax1.plot(e["lam"], e["firth"], "o-", label="Firth w_crit", color="black")
    ax1.plot(e["lam"], e["isotonic"], "^:", label="isotonic w_crit", color="tab:red")
    ax1.set_title(f"{R['title']}: w_crit(lambda), three estimators")
    ax1.set_xlabel("lambda"); ax1.set_ylabel("w_crit (P=0.5)"); ax1.legend()
    plt.tight_layout()
    p = os.path.join(outdir, f"{R['key']}_estimator_overlay.png")
    plt.savefig(p, dpi=150); plt.close(); return p


def plot_boundary_metrics(R, outdir):
    """Continuous metrics vs w (the crossing object): per-w distribution, lambda_lo & lambda_hi."""
    d, metrics = R["d"], R["metrics"]
    lam_lo, lam_hi = R["lam_lo"], R["lam_hi"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 5), squeeze=False)
    for j, met in enumerate(metrics):
        ax = axes[0][j]
        for lam, col in ((lam_lo, "tab:blue"), (lam_hi, "tab:red")):
            s = d[d.lambda_loss == lam]
            gp = s.groupby("w")[met]
            ws = sorted(s["w"].unique())
            med = [gp.get_group(w).median() for w in ws]
            q1 = [gp.get_group(w).quantile(0.25) for w in ws]
            q3 = [gp.get_group(w).quantile(0.75) for w in ws]
            ax.plot(ws, med, "o-", color=col, label=f"lambda={lam}")
            ax.fill_between(ws, q1, q3, color=col, alpha=0.15)
            # raw scatter to show whether the distribution is bimodal or smooth
            ax.scatter(s["w"] + (0.004 if lam == lam_hi else -0.004), s[met],
                       s=8, color=col, alpha=0.25)
        ax.set_xlabel("w"); ax.set_ylabel(met)
        ax.set_title(f"{R['title']}: {met} vs w (median, IQR, raw)")
        ax.legend()
    plt.tight_layout()
    p = os.path.join(outdir, f"{R['key']}_boundary_metrics.png")
    plt.savefig(p, dpi=150); plt.close(); return p


def plot_supply_scatter(R, outdir):
    """Supply-chain-only: base-stock vs profit scatter, colored by collapse, to show the
    collapse label's two-variable structure around the boundary."""
    if R["key"] != "supply":
        return None
    d = R["d"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for cval, col, lab in ((0, "tab:green", "not collapsed"), (1, "tab:red", "collapsed")):
        s = d[d.collapsed == cval]
        ax.scatter(s["mean_S"], s["profit"], s=10, color=col, alpha=0.4, label=lab)
    ax.axhline(-127.1, color="gray", ls="--", lw=1, label="profit floor (-127.1)")
    ax.axvline(10.0, color="gray", ls=":", lw=1, label="base-stock floor (10)")
    ax.set_xlabel("mean base-stock S"); ax.set_ylabel("profit")
    ax.set_title(f"{R['title']}: base-stock vs profit (collapse label structure)")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(outdir, f"{R['key']}_basestock_vs_profit.png")
    plt.savefig(p, dpi=150); plt.close(); return p


# --- Report ------------------------------------------------------------------

def write_report(results, outdir):
    lines = ["# Part-2 reanalysis -- robustness of the lambda=1->7 rightward w_crit shift\n",
             "No new training. Reanalysis of existing raw-data CSVs.\n",
             "## Method\n",
             f"- Bootstrap replicates: **{N_BOOT}**.",
             f"- Resampling unit: **{BOOT_UNIT}**.",
             f"- Separation handling: {SEP_HANDLING}",
             f"- Difference CI: {DIFF_CI}",
             "- Estimators: logistic (statsmodels, L2 fallback), Firth penalized logistic, "
             "and isotonic regression (monotone-decreasing in w).",
             "- w_crit: logistic/Firth = -b0/b1; isotonic = interpolated grid crossing of "
             "P(collapse)=0.5 (NaN if the monotone curve never crosses within the w-grid).\n"]

    for R in results:
        e = R["est"]
        lines.append(f"## {R['title']}  (lambda in {[int(l) for l in R['lams']]})\n")
        if R.get("theta") is not None:
            lines.append(f"Collapse label RECOMPUTED with recalibrated theta = "
                         f"**{R['theta']:.3f}** (the stored CSV `collapsed` column is under "
                         "an un-recalibrated threshold; recalibration mirrors the paper's "
                         f"analyze_{R['key']} exactly).\n")
        lines.append("### w_crit(lambda), three estimators")
        lines.append("| lambda | logistic | Firth | isotonic | in-grid? | n |")
        lines.append("|---|---|---|---|---|---|")
        any_offgrid = False
        for _, r in e.iterrows():
            fmt = lambda v: "n/a" if not np.isfinite(v) else f"{v:.3f}"
            ig = "yes" if r.get("in_grid") else "**extrapolated**"
            if not r.get("in_grid") and not r.get("degenerate"):
                any_offgrid = True
            lines.append(f"| {int(r['lam'])} | {fmt(r['logistic'])} | {fmt(r['firth'])} | "
                         f"{fmt(r['isotonic'])} | {ig} | {int(r['n'])} |")
        if any_offgrid:
            lines.append("\n> **Note:** rows marked *extrapolated* have w_crit beyond the "
                         "tested w-grid, so the numeric value is an out-of-range logistic "
                         "extrapolation; isotonic is n/a there (its monotone curve never "
                         "crosses 0.5 within the grid). The *direction* of the shift is "
                         "still valid; the magnitude at those lambda is a lower bound.")
        # endpoint shift under each estimator
        lo, hi = R["lam_lo"], R["lam_hi"]
        rlo = e[e.lam == lo].iloc[0]; rhi = e[e.lam == hi].iloc[0]
        lines.append("")
        for name in ("logistic", "firth", "isotonic"):
            if np.isfinite(rlo[name]) and np.isfinite(rhi[name]):
                lines.append(f"- **{name} shift lambda={lo}->{hi}: "
                             f"{rhi[name]-rlo[name]:+.3f}** ({rlo[name]:.3f} -> {rhi[name]:.3f})")
        if R["boot"] is not None:
            b = R["boot"]
            lines.append("")
            lines.append(f"### Seed-paired bootstrap (Firth w_crit, {b['usable']}/{N_BOOT} usable replicates)")
            lines.append(f"- w_crit(lambda={lo}) = {b['lo'][0]:.3f} [{b['lo'][1]:.3f}, {b['lo'][2]:.3f}]")
            lines.append(f"- w_crit(lambda={hi}) = {b['hi'][0]:.3f} [{b['hi'][1]:.3f}, {b['hi'][2]:.3f}]")
            lines.append(f"- **difference (lambda={hi} - lambda={lo}) = {-b['diff'][0]:.3f} "
                         f"[{-b['diff'][2]:.3f}, {-b['diff'][1]:.3f}]** "
                         "(paired; CI excludes 0 => shift is seed-robust)"
                         if (b['diff'][1] > 0 or b['diff'][2] < 0) else
                         f"- difference (lambda={hi} - lambda={lo}) = {-b['diff'][0]:.3f} "
                         f"[{-b['diff'][2]:.3f}, {-b['diff'][1]:.3f}] (paired; CI includes 0)")

        # raw counts
        lines.append("\n### Raw collapse counts k/n at every w")
        piv = R["counts"].pivot(index="lam", columns="w", values="frac").round(2)
        kn = R["counts"].copy()
        kn["cell"] = kn.apply(lambda r: f"{r['k']}/{r['n']}", axis=1)
        knp = kn.pivot(index="lam", columns="w", values="cell")
        lines.append("Collapse fraction:")
        lines.append("```"); lines.append(piv.to_string()); lines.append("```")
        lines.append("Raw counts (k collapsed / n):")
        lines.append("```"); lines.append(knp.to_string()); lines.append("```")

        figs = R.get("figs", {})
        for label, path in figs.items():
            if path:
                lines.append(f"\n![{label}]({os.path.basename(path)})")
        lines.append("\n### Nature of the object")
        lines.append("See the continuous-metric plot: if the per-w distributions are smooth "
                     "rather than bimodal, describe the boundary as a **threshold-defined "
                     "behavioral crossing**, not a physical phase transition.\n")

    with open(os.path.join(outdir, "REANALYSIS_controls.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {os.path.join(outdir, 'REANALYSIS_controls.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=list(SUBSTRATES) + ["all"], default="all")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    keys = list(SUBSTRATES) if args.substrate == "all" else [args.substrate]
    results = []
    for key in keys:
        try:
            R = analyze_substrate(key)
        except FileNotFoundError as e:
            print(f"[skip {key}] {e}")
            continue
        R["figs"] = {
            "estimator overlay": plot_estimator_overlay(R, OUTDIR),
            "boundary metrics": plot_boundary_metrics(R, OUTDIR),
            "base-stock vs profit": plot_supply_scatter(R, OUTDIR),
        }
        results.append(R)
        # console summary
        print(f"\n=== {R['title']} ===")
        print(R["est"].to_string(index=False))
        if R["boot"]:
            b = R["boot"]; lo, hi = R["lam_lo"], R["lam_hi"]
            print(f"paired bootstrap diff w_crit(l{hi}-l{lo}) = {-b['diff'][0]:.3f} "
                  f"[{-b['diff'][2]:.3f}, {-b['diff'][1]:.3f}]  ({b['usable']}/{N_BOOT} usable)")
        if R["boot_unpaired"]:
            u = R["boot_unpaired"]
            print(f"unpaired bootstrap diff w_crit(l{hi}-l{lo}) = {-u['diff'][0]:.3f} "
                  f"[{-u['diff'][2]:.3f}, {-u['diff'][1]:.3f}]  ({u['usable']}/{N_BOOT} usable)")

    if results:
        write_report(results, OUTDIR)


if __name__ == "__main__":
    main()
