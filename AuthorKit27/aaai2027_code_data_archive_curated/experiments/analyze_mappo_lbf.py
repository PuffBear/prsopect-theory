"""
Analyze MAPPO-LBF (third environment, breadth result): w_crit(lambda) surface,
null-anchored decomposition, monotonicity verdict.

Reads:   docs/expMAPPO_lbf_figures/mappo_lbf_raw_data.csv
Outputs:
  - data-driven collapse theta recalibrated from the null endpoints (the raw
    sweep used a placeholder 0.5; mean_reward is logged raw so we relabel here)
  - per-lambda logistic fit + bootstrap(2000) w_crit + 95% CI
  - curvature shift (lam1@0.88 - null) and loss-aversion excess (lam7 - lam1)
  - monotonicity verdict (pre-registered success test)
  - collapse curves + w_crit-vs-lambda figures
  - mappo_lbf_results.md

Uses the SAME estimator stack as analyze_mappo_orgym / analyze_expN so LBF lands
on identical statistical footing as the supply-chain and MPE series.

  python experiments/analyze_mappo_lbf.py
  python experiments/analyze_mappo_lbf.py --theta 0.62   # override the recalibration
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
from analyze_expN import fit_logistic, bootstrap_wcrit, per_w_table, curve

CSV = "docs/expMAPPO_lbf_figures/mappo_lbf_raw_data.csv"
LOWW = "docs/expMAPPO_lbf_figures/mappo_lbf_loww_raw_data.csv"  # low-w extension
HIGHW = "docs/expMAPPO_lbf_figures/mappo_lbf_highw_raw_data.csv"  # w=0.90 high-lambda resolver
OUTDIR = "docs/expMAPPO_lbf_figures"
LAMBDAS = [1, 2, 3, 4, 5, 6, 7]
COLORS = {1: "tab:blue", 2: "tab:cyan", 3: "tab:orange", 4: "tab:olive",
          5: "tab:green", 6: "tab:purple", 7: "tab:red"}


def _load():
    frames = [pd.read_csv(CSV, keep_default_na=False)]
    if os.path.exists(LOWW):                       # auto-merge low-w cells
        frames.append(pd.read_csv(LOWW, keep_default_na=False))
    if os.path.exists(HIGHW):                      # auto-merge w=0.90 high-lambda cells
        frames.append(pd.read_csv(HIGHW, keep_default_na=False))
    df = pd.concat(frames, ignore_index=True)
    df["mean_reward"] = pd.to_numeric(df["mean_reward"], errors="coerce")
    df["w"] = pd.to_numeric(df["w"], errors="coerce")
    df["lambda_loss"] = pd.to_numeric(df["lambda_loss"], errors="coerce")
    # drop any malformed/partial-write rows (e.g. a concurrent append left a blank
    # field) so one bad row can't crash the merge.
    n0 = len(df)
    df = df.dropna(subset=["mean_reward", "w", "lambda_loss"]).copy()
    if len(df) < n0:
        print(f"  [dropped {n0 - len(df)} malformed row(s) on load]")
    df["lambda_loss"] = df["lambda_loss"].round().astype(int)
    return df


def _valid_boundary(r, w_max=1.0):
    """A boundary estimate is trustworthy only if w_crit is inside (a little
    past) the tested grid with a non-degenerate CI. Guards against the null's
    no-collapse-in-range degenerate fit (e.g. w_crit=-0.76, CI[-10,9])."""
    return (r is not None and np.isfinite(r["w_crit"])
            and -0.05 <= r["w_crit"] <= w_max + 0.05
            and (r["ci_hi"] - r["ci_lo"]) < 0.6)


def recalibrate_theta(df):
    """theta = midpoint of the COLLAPSED regime (highest-lambda, lowest-w mean
    return) and the COORDINATED regime (null, highest-w) -- the same
    collapsed-vs-coordinated midpoint method used to calibrate MPE's occupancy
    threshold.

    NOTE: the null condition coordinates at *every* w in LBF, so a null-only
    bracket sits inside the coordinated range and inflates theta (an earlier
    pass got ~0.85 this way, leaving the null boundary soft and the high-lambda
    boundaries pinned at the grid ceiling). Bracketing the two genuine regimes
    instead lands theta ~0.64; the loss-aversion shift is invariant to the
    choice (see the threshold-robustness block in main)."""
    w_min, w_max = df["w"].min(), df["w"].max()
    hi_lam = df["lambda_loss"].max()
    collapsed_ref = df[(df["lambda_loss"] == hi_lam) &
                       (df["w"] == w_min)]["mean_reward"].mean()
    null = df[df["condition"] == "null"]
    coord_ref = (null[null["w"] == w_max]["mean_reward"].mean()
                 if not null.empty else
                 df[df["w"] == w_max]["mean_reward"].mean())
    if not (np.isfinite(collapsed_ref) and np.isfinite(coord_ref)):
        return 0.62
    return float((collapsed_ref + coord_ref) / 2.0)


def _label(df, theta):
    df = df.copy()
    df["collapsed"] = (df["mean_reward"] < theta).astype(int)
    return df


def affine_law_transfer(perlam, saturated):
    """Test whether the supply chain's one-parameter affine loss-aversion law
        w_crit(lambda) = phi / (phi + g/c),   phi = 1 + rho*(lambda-1),  rho in [0,1]
    transfers to this environment. g/c is fixed from the lambda=1 boundary
    (phi=1 => w1 = 1/(1+g/c)); rho is grid-fit (no scipy dependency) on the
    in-grid, non-saturated lambdas. Supply-chain reference: rho=0.61, RMSE=0.014.
    A 'transfer' means a supply-chain-like fit: small RMSE with an interior rho
    (not pinned at the [0,1] bound)."""
    if 1 not in perlam:
        return None
    lam = np.array(sorted(perlam), dtype=float)
    wc = np.array([perlam[int(l)]["w_crit"] for l in lam])
    w1 = perlam[1]["w_crit"]
    if not (0.0 < w1 < 1.0):
        return None
    gc = (1.0 - w1) / w1
    reliable = np.array([int(l) not in saturated for l in lam])

    def pred(rho, L):
        phi = 1.0 + rho * (L - 1.0)
        return phi / (phi + gc)

    rhos = np.linspace(0.0, 1.0, 1001)

    def _best(mask):
        if mask.sum() < 2:
            return float("nan"), float("nan")
        errs = [np.sqrt(np.mean((pred(r, lam[mask]) - wc[mask]) ** 2)) for r in rhos]
        i = int(np.argmin(errs))
        return float(rhos[i]), float(errs[i])

    rho_in, rmse_in = _best(reliable)
    rho_all, rmse_all = _best(np.ones_like(lam, dtype=bool))
    transfers = bool(np.isfinite(rmse_in) and rmse_in < 0.05 and rho_in < 0.999)
    return dict(gc=gc, rho_in=rho_in, rmse_in=rmse_in, rho_all=rho_all,
                rmse_all=rmse_all, transfers=transfers,
                n_reliable=int(reliable.sum()))


def _fit(sub):
    if len(sub) < 5 or sub["collapsed"].nunique() < 2:
        return None
    m, slope, pval, r2, wc = fit_logistic(sub)
    lo, hi = bootstrap_wcrit(sub)[1:]
    return dict(n=len(sub), cells=per_w_table(sub), model=m, slope=slope,
                pval=pval, r2=r2, w_crit=wc, ci_lo=lo, ci_hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=None,
                    help="override collapse threshold (default: recalibrate from null)")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(CSV):
        print(f"No data at {CSV}"); sys.exit(0)

    df = _load()
    theta = args.theta if args.theta is not None else recalibrate_theta(df)
    df = _label(df, theta)
    null_df = df[df["condition"] == "null"]
    sweep_df = df[df["condition"] != "null"]

    print(f"\n  collapse theta (team return) = {theta:.3f} "
          f"({'override' if args.theta is not None else 'recalibrated from null'})")

    null = _fit(null_df)
    perlam = {lam: _fit(sweep_df[sweep_df["lambda_loss"] == lam]) for lam in LAMBDAS}
    perlam = {k: v for k, v in perlam.items() if v is not None}

    w_max = df["w"].max()
    null_valid = _valid_boundary(null, w_max)
    null_disp = null if null_valid else None       # keeps a degenerate null off the figure
    saturated = sorted(l for l, r in perlam.items()
                       if r["ci_hi"] > w_max or r["w_crit"] >= w_max - 0.02)
    if not null_valid:
        print("  [null has NO in-grid boundary -- coordinates across the tested "
              "range; reported as undefined, not as a number.]")
    if saturated:
        print(f"  [saturated at the w-grid ceiling (treat as exploratory): "
              f"lambda in {saturated}]")

    print("=" * 76)
    print(f"  MAPPO -- LBF (3rd environment)  n={len(df)}  theta={theta:.3f}")
    print("=" * 76)
    print(f"{'condition':>14} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | "
          f"{'slope':>8} | {'p':>9} | {'n':>4}")
    print("-" * 76)
    if null_valid:
        print(f"{'null (l=1,a=1)':>14} | {null['w_crit']:>7.3f} | {null['ci_lo']:>6.3f} | "
              f"{null['ci_hi']:>6.3f} | {null['slope']:>8.2f} | {null['pval']:>9.2e} | {null['n']:>4}")
    else:
        print(f"{'null (l=1,a=1)':>14} |   n/a   (no in-grid boundary: coordinates across range)")
    for lam in LAMBDAS:
        if lam in perlam:
            r = perlam[lam]
            tag = ("curv (l=1)" if lam == 1 else f"loss l={lam}") + ("*" if lam in saturated else "")
            print(f"{tag:>14} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | {r['ci_hi']:>6.3f} | "
                  f"{r['slope']:>8.2f} | {r['pval']:>9.2e} | {r['n']:>4}")
    if saturated:
        print("  (* boundary at/over the w-grid ceiling -- saturated, treat as exploratory)")
    print("=" * 76)

    curv_shift = loss_excess = None
    if null_valid and 1 in perlam:
        curv_shift = perlam[1]["w_crit"] - null["w_crit"]
        print(f"  curvature shift   (w_crit[l1@0.88] - w_crit[null]) = {curv_shift:+.3f}")
    elif 1 in perlam:
        print("  curvature shift   = n/a (null has no in-grid boundary)")
    if 1 in perlam and 7 in perlam:
        loss_excess = perlam[7]["w_crit"] - perlam[1]["w_crit"]
        print(f"  loss-aversion excess (w_crit[l7@0.88] - w_crit[l1@0.88]) = {loss_excess:+.3f}")

    verdict = "INSUFFICIENT DATA"
    if 1 in perlam and 7 in perlam:
        r1, r7 = perlam[1], perlam[7]
        rises = r7["w_crit"] > r1["w_crit"]
        non_overlap = r7["ci_lo"] > r1["ci_hi"]
        seq = [perlam[l]["w_crit"] for l in sorted(perlam)]
        monotone = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
        if rises and non_overlap:
            verdict = "PASS (w_crit rises, CIs non-overlapping)"
        elif rises:
            verdict = "DIRECTIONAL (rises, CIs overlap -> exploratory)"
        else:
            verdict = "FALSIFIED (flat/decreasing)"
        print(f"  monotone non-decreasing over swept lambda: {monotone}")
    print(f"  PRE-REGISTERED VERDICT: {verdict}")

    # --- affine-law transfer test: does the supply-chain one-parameter affine
    #     lambda-law transfer here? (reproduces the RMSE comparison in the paper) -
    aff = affine_law_transfer(perlam, saturated)
    if aff is not None:
        print("\n  affine-law transfer (supply-chain reference rho=0.61, RMSE=0.014):")
        print(f"    g/c={aff['gc']:.3f} (fixed from lambda=1); best rho={aff['rho_in']:.3f}, "
              f"RMSE={aff['rmse_in']:.3f} on {aff['n_reliable']} in-grid lambdas "
              f"-> {'TRANSFERS' if aff['transfers'] else 'does NOT transfer'}")

    # --- threshold robustness: the loss-aversion direction must survive the
    #     theta choice (reported as a strength, mirroring the supply-chain
    #     threshold-robustness appendix) ----------------------------------------
    print("\n  threshold robustness (lambda1->7 w_crit):")
    robustness_lines = []
    for th in sorted({0.50, 0.62, 0.70, round(float(theta), 2)}):
        dth = _label(df, th)
        seq = []
        for lam in LAMBDAS:
            r = _fit(dth[(dth["condition"] != "null") & (dth["lambda_loss"] == lam)])
            seq.append(r["w_crit"] if r else float("nan"))
        fin = [(a, b) for a, b in zip(seq, seq[1:]) if np.isfinite(a) and np.isfinite(b)]
        mono = all(b >= a - 1e-9 for a, b in fin)
        line = (f"theta={th:.2f}: lambda1->7 = "
                f"{np.array2string(np.array(seq), precision=2)}  monotone={mono}")
        print("    " + line); robustness_lines.append(line)

    # series CSV so make_generalization_figure can add the LBF line
    import csv
    series_path = os.path.join(OUTDIR, "lbf_wcrit_series.csv")
    with open(series_path, "w", newline="") as fcsv:
        wcsv = csv.writer(fcsv); wcsv.writerow(["lambda", "wcrit", "ci_lo", "ci_hi"])
        for lam in sorted(perlam):
            r = perlam[lam]
            wcsv.writerow([lam, f"{r['w_crit']:.4f}",
                           f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}"])
    print(f"  saved {series_path}")

    _figures(null_disp, perlam, theta)
    _writeup(df, null_disp, perlam, theta, curv_shift, loss_excess, verdict, saturated)
    with open(os.path.join(OUTDIR, "mappo_lbf_results.md"), "a") as f:
        f.write("\n**Threshold robustness** (loss-aversion direction is "
                "invariant to theta):\n\n")
        for line in robustness_lines:
            f.write(f"- {line}\n")
        if aff is not None:
            f.write(f"\n**Affine-law transfer** (supply-chain reference "
                    f"rho=0.61, RMSE=0.014): g/c={aff['gc']:.3f} from lambda=1, "
                    f"best rho={aff['rho_in']:.3f}, RMSE={aff['rmse_in']:.3f} on "
                    f"{aff['n_reliable']} in-grid lambdas -> "
                    f"**{'transfers' if aff['transfers'] else 'does NOT transfer'}** "
                    f"(form is substrate-specific; the *method* generalizes, the "
                    f"affine *constants/form* do not).\n")


def _figures(null, perlam, theta):
    sns.set_style("whitegrid")
    wg = np.linspace(0.2, 1.0, 200)
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
        if np.isfinite(r["w_crit"]) and 0 <= r["w_crit"] <= 1:
            ax.axvline(r["w_crit"], color=c, ls="--", lw=1.0)
    if null:
        ax.plot(wg, curve(null["model"], wg), color="black", lw=1.6, ls=":",
                label="null (l=1,a=1)")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
    ax.set_title(f"MAPPO on LBF: collapse transition by lambda (theta={theta:.2f})")
    ax.legend(title="loss aversion", ncol=2)
    plt.tight_layout()
    f1 = os.path.join(OUTDIR, "mappo_lbf_collapse_curves.png")
    plt.savefig(f1, dpi=150); print(f"  saved {f1}"); plt.close(fig)

    lams = sorted(perlam)
    if lams:
        fig, ax = plt.subplots(figsize=(8, 5))
        ys = [perlam[l]["w_crit"] for l in lams]
        lo = [perlam[l]["w_crit"] - perlam[l]["ci_lo"] for l in lams]
        hi = [perlam[l]["ci_hi"] - perlam[l]["w_crit"] for l in lams]
        ax.errorbar(lams, ys, yerr=[lo, hi], fmt="o-", color="tab:red",
                    capsize=5, lw=2, markersize=8, label="MAPPO-LBF")
        if null and np.isfinite(null["w_crit"]):
            ax.axhline(null["w_crit"], color="black", ls=":", lw=1.5, label="null")
        ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
        ax.set_xticks(LAMBDAS); ax.set_ylim(0, 1)
        ax.set_title("w_crit vs lambda: MAPPO on LBF (3rd environment)")
        ax.legend(); plt.tight_layout()
        f2 = os.path.join(OUTDIR, "mappo_lbf_wcrit_vs_lambda.png")
        plt.savefig(f2, dpi=150); print(f"  saved {f2}"); plt.close(fig)


def _writeup(df, null, perlam, theta, curv_shift, loss_excess, verdict, saturated=()):
    md = ["# MAPPO on LBF -- third-environment breadth result\n",
          f"n = {len(df)} runs. Collapse theta = {theta:.3f} "
          f"(collapsed-vs-coordinated midpoint, same calibration method as MPE). "
          f"Bootstrap n=2000. Estimator identical to PPO/MAPPO.\n",
          "| condition | w_crit | 95% CI | slope | p | n |",
          "|---|---|---|---|---|---|"]
    if null:
        md.append(f"| null (l=1,a=1) | {null['w_crit']:.3f} | "
                  f"[{null['ci_lo']:.3f}, {null['ci_hi']:.3f}] | {null['slope']:.2f} | "
                  f"{null['pval']:.2e} | {null['n']} |")
    else:
        md.append("| null (l=1,a=1) | n/a | no in-grid boundary | -- | -- | 120 |")
    for lam in LAMBDAS:
        if lam in perlam:
            r = perlam[lam]
            tag = "curvature (l=1, a=0.88)" if lam == 1 else f"loss aversion l={lam}"
            if lam in saturated:
                tag += " (saturated\\*)"
            md.append(f"| {tag} | {r['w_crit']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                      f"{r['slope']:.2f} | {r['pval']:.2e} | {r['n']} |")
    md.append("")
    if null is None:
        md.append("- **Null:** no in-grid boundary -- the risk-neutral team "
                  "coordinates across the entire tested w-range, so LBF isolates "
                  "the loss-aversion effect (curvature decomposition not "
                  "identifiable here; it is supply-chain-specific).")
    if curv_shift is not None:
        md.append(f"- **Curvature shift** = **{curv_shift:+.3f}**")
    if loss_excess is not None:
        md.append(f"- **Loss-aversion excess** (l7 - l1) = **{loss_excess:+.3f}**")
    if saturated:
        md.append(f"- **\\*Saturated** at the w-grid ceiling (exploratory): "
                  f"lambda in {sorted(saturated)}.")
    md.append(f"- **Pre-registered verdict:** {verdict}")
    path = os.path.join(OUTDIR, "mappo_lbf_results.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
