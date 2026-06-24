"""
Analyze the MAPPO-MAgent breadth sweep (4th environment): does w_crit rise with
loss aversion (lambda) as on supply chain / MPE / LBF?

Mirrors analyze_mappo_lbf.py: data-driven collapse theta, per-lambda Firth/
logistic w_crit with bootstrap CIs, null-anchored curvature/loss-aversion
decomposition, and the pre-registered monotonicity verdict. Reuses the SAME
fit/bootstrap helpers (analyze_expN) so the result is methodologically identical
to the other environments.

theta calibration: midpoint of the COLLAPSED regime (w=0 selfish, null) and the
COORDINATED regime (w=1 team, null) from the calibration run if present, else
from the sweep's own null endpoints.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import fit_logistic, bootstrap_wcrit, per_w_table

CSV   = "docs/exp_magent/mappo_magent_raw_data.csv"
CALIB = "docs/exp_magent/magent_calibration_raw_data.csv"
THRESHOLD_JSON = "docs/exp_magent/magent_collapse_threshold.json"
OUTDIR = "docs/exp_magent"
LAMBDAS = [1, 2, 3, 4, 5, 6, 7]


def _load(path):
    df = pd.read_csv(path, keep_default_na=False)
    # the calibration CSV has no lambda_loss/condition columns -- coerce only the
    # numeric cols that are present so this loader works for both files.
    for c in ("mean_reward", "w", "lambda_loss"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    present = [c for c in ("mean_reward", "w", "lambda_loss") if c in df.columns]
    df = df.dropna(subset=present).copy()
    if "lambda_loss" in df.columns:
        df["lambda_loss"] = df["lambda_loss"].round().astype(int)
    return df


def calibrate_theta(df):
    """theta = midpoint(collapsed regime, coordinated regime). Prefer the
    dedicated calibration endpoints (null, w=0 vs w=1); fall back to the sweep's
    own null endpoints. Persist to JSON so the env wrapper's is_collapsed_magent
    reads the same value."""
    src = _load(CALIB) if os.path.exists(CALIB) else df[df["condition"] == "null"]
    if src.empty:
        return -10.0
    w_min, w_max = src["w"].min(), src["w"].max()
    collapsed_ref = src[src["w"] == w_min]["mean_reward"].mean()
    coord_ref     = src[src["w"] == w_max]["mean_reward"].mean()
    if not (np.isfinite(collapsed_ref) and np.isfinite(coord_ref)):
        return -10.0
    theta = float((collapsed_ref + coord_ref) / 2.0)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(THRESHOLD_JSON, "w") as f:
        json.dump({"threshold": theta,
                   "collapsed_ref": float(collapsed_ref),
                   "coord_ref": float(coord_ref)}, f, indent=2)
    return theta


def _label(df, theta):
    df = df.copy()
    df["collapsed"] = (df["mean_reward"] < theta).astype(int)
    return df


def _valid_boundary(r, w_max):
    return (r is not None and np.isfinite(r["w_crit"])
            and -0.05 <= r["w_crit"] <= w_max + 0.05
            and (r["ci_hi"] - r["ci_lo"]) < 0.6)


def _fit(sub):
    if len(sub) < 5 or sub["collapsed"].nunique() < 2:
        return None
    m, slope, pval, r2, wc = fit_logistic(sub)
    lo, hi = bootstrap_wcrit(sub)[1:]
    return dict(n=len(sub), model=m, slope=slope, pval=pval, r2=r2,
                w_crit=wc, ci_lo=lo, ci_hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # Calibration-only mode: the sweep CSV does not exist yet, but the calibration
    # endpoints do. Compute + persist theta (so the sweep's is_collapsed_magent
    # reads it) and stop -- there is no boundary to fit yet.
    if not os.path.exists(CSV):
        if os.path.exists(CALIB):
            theta = calibrate_theta(_load(CALIB))
            print(f"  [calibration-only] no sweep data yet at {CSV}")
            print(f"  collapse theta (team reward) = {theta:.3f}")
            print(f"  wrote {THRESHOLD_JSON}")
            print("  -> push this JSON to the cluster, then qsub run_mappo_magent.pbs")
        else:
            print(f"No data at {CSV} and no calibration data at {CALIB}")
        sys.exit(0)

    df = _load(CSV)
    theta = args.theta if args.theta is not None else calibrate_theta(df)
    df = _label(df, theta)
    null_df  = df[df["condition"] == "null"]
    sweep_df = df[df["condition"] != "null"]
    w_max = df["w"].max()

    print(f"\n  collapse theta (team reward) = {theta:.3f} "
          f"({'override' if args.theta is not None else 'calibrated'})")

    null = _fit(null_df)
    perlam = {lam: _fit(sweep_df[sweep_df["lambda_loss"] == lam]) for lam in LAMBDAS}
    perlam = {k: v for k, v in perlam.items() if v is not None}
    null_valid = _valid_boundary(null, w_max)
    saturated = sorted(l for l, r in perlam.items()
                       if r["ci_hi"] > w_max or r["w_crit"] >= w_max - 0.02)

    print("=" * 76)
    print(f"  MAPPO -- MAgent (4th environment)  n={len(df)}  theta={theta:.3f}")
    print("=" * 76)
    print(f"{'condition':>14} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | "
          f"{'slope':>8} | {'p':>9} | {'n':>4}")
    print("-" * 76)
    if null_valid:
        print(f"{'null (l=1,a=1)':>14} | {null['w_crit']:>7.3f} | {null['ci_lo']:>6.3f} | "
              f"{null['ci_hi']:>6.3f} | {null['slope']:>8.2f} | {null['pval']:>9.2e} | {null['n']:>4}")
    else:
        print(f"{'null (l=1,a=1)':>14} |   n/a (no in-grid boundary)")
    for lam in LAMBDAS:
        if lam in perlam:
            r = perlam[lam]
            tag = ("curv (l=1)" if lam == 1 else f"loss l={lam}") + ("*" if lam in saturated else "")
            print(f"{tag:>14} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | {r['ci_hi']:>6.3f} | "
                  f"{r['slope']:>8.2f} | {r['pval']:>9.2e} | {r['n']:>4}")
    if saturated:
        print("  (* boundary at/over the w-grid ceiling -- saturated, exploratory)")
    print("=" * 76)

    curv_shift = loss_excess = None
    if null_valid and 1 in perlam:
        curv_shift = perlam[1]["w_crit"] - null["w_crit"]
        print(f"  curvature shift      (w_crit[l1] - w_crit[null]) = {curv_shift:+.3f}")
    if 1 in perlam and 7 in perlam:
        loss_excess = perlam[7]["w_crit"] - perlam[1]["w_crit"]
        print(f"  loss-aversion excess (w_crit[l7] - w_crit[l1])   = {loss_excess:+.3f}")

    verdict = "INSUFFICIENT DATA"
    if 1 in perlam and 7 in perlam:
        lams = sorted(perlam)
        wcs = [perlam[l]["w_crit"] for l in lams]
        rises = wcs[-1] > wcs[0]
        monotone = all(b >= a - 0.02 for a, b in zip(wcs, wcs[1:]))
        nonoverlap = perlam[lams[-1]]["ci_lo"] > perlam[lams[0]]["ci_hi"]
        if rises and nonoverlap:
            verdict = "PASS (w_crit rises, CIs non-overlapping)"
        elif rises and monotone:
            verdict = "DIRECTIONAL (rises, CIs overlap -> exploratory)"
        else:
            verdict = "FAIL (w_crit does not rise with lambda)"
        print(f"  monotone non-decreasing over swept lambda: {monotone}")
    print(f"  PRE-REGISTERED VERDICT: {verdict}")

    # figure: w_crit vs lambda
    lams = sorted(perlam)
    if lams:
        wcs = [perlam[l]["w_crit"] for l in lams]
        los = [perlam[l]["w_crit"] - perlam[l]["ci_lo"] for l in lams]
        his = [perlam[l]["ci_hi"] - perlam[l]["w_crit"] for l in lams]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(lams, wcs, yerr=[los, his], fmt="o-", color="#c00000",
                    capsize=4, label="$w_{crit}$ (logistic +/- 95% CI)")
        if null_valid:
            ax.axhline(null["w_crit"], color="gray", ls="--",
                       label=f"null = {null['w_crit']:.3f}")
        ax.set_xlabel("loss aversion $\\lambda$"); ax.set_ylabel("$w_{crit}$")
        ax.set_title("MAgent (4th env): coordination boundary vs loss aversion")
        ax.legend(fontsize=9); fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "mappo_magent_wcrit_vs_lambda.png"),
                    dpi=150, bbox_inches="tight")
        print(f"  saved {OUTDIR}/mappo_magent_wcrit_vs_lambda.png")

    with open(os.path.join(OUTDIR, "mappo_magent_results.md"), "w") as f:
        f.write("# MAPPO-MAgent (4th environment) results\n\n")
        f.write(f"- collapse theta (team reward): **{theta:.3f}**\n")
        f.write(f"- curvature shift: **{curv_shift if curv_shift is None else round(curv_shift,3)}**\n")
        f.write(f"- loss-aversion excess: **{loss_excess if loss_excess is None else round(loss_excess,3)}**\n")
        f.write(f"- PRE-REGISTERED VERDICT: **{verdict}**\n\n")
        f.write("| condition | w_crit | CI_lo | CI_hi | n |\n|---|---|---|---|---|\n")
        for lam in LAMBDAS:
            if lam in perlam:
                r = perlam[lam]
                f.write(f"| lambda{lam} | {r['w_crit']:.3f} | {r['ci_lo']:.3f} | "
                        f"{r['ci_hi']:.3f} | {r['n']} |\n")
    print(f"  saved {OUTDIR}/mappo_magent_results.md")


if __name__ == "__main__":
    main()
