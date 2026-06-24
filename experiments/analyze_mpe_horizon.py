"""
Analyze the MAPPO-MPE horizon sweep: does w_crit slide LEFT as the training
horizon T grows (metastability), as on the supply chain?

Reads:   docs/expMAPPO_mpe_horizon_figures/mappo_mpe_horizon_raw_data.csv
Outputs: per-horizon Firth w_crit + bootstrap(2000) 95% CI, the slide, verdict,
         and a figure.

Collapse = occupancy < theta. theta is recalibrated from the data's own w=0 vs
w=1 occupancy endpoints (midpoint), pooled across horizons -- the same
collapsed-vs-coordinated method as analyze_mpe_decomposition (and the same reason:
the shipped theta=0.164 was stale; recalibrate per run). mean_occupancy is logged
raw. Read with keep_default_na=False is unnecessary here (no 'null' string col).

  python experiments/analyze_mpe_horizon.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import fit_logistic, bootstrap_wcrit, per_w_table, curve

CSV = "docs/expMAPPO_mpe_horizon_figures/mappo_mpe_horizon_raw_data.csv"
OUTDIR = "docs/expMAPPO_mpe_horizon_figures"
AXIS = "horizon"
# layout from exp_mappo_mpe_horizon.map_run_idx: w(outer) x horizon(middle) x seed(inner)
HORIZONS = [2_000_000, 8_000_000]
N_SEED = 20


def _axis_from_idx(run_idx):
    block = len(HORIZONS) * N_SEED
    return HORIZONS[(int(run_idx) % block) // N_SEED]


def recalibrate_theta(df):
    """Midpoint of pooled w=0 (collapsed) and w=1 (coordinated) mean occupancy."""
    w0 = df[df["w"] == df["w"].min()]["mean_occupancy"].mean()
    w1 = df[df["w"] == df["w"].max()]["mean_occupancy"].mean()
    if not (np.isfinite(w0) and np.isfinite(w1)):
        return 0.144
    return float((w0 + w1) / 2.0)


def _load():
    df = pd.read_csv(CSV)
    for c in ("mean_occupancy", "w", "run_idx"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mean_occupancy", "w", "run_idx"]).copy()
    # recover the axis from run_idx if the (buggy) driver dropped the column
    if AXIS not in df.columns or pd.to_numeric(df[AXIS], errors="coerce").isna().all():
        df[AXIS] = df["run_idx"].map(_axis_from_idx)
    else:
        df[AXIS] = pd.to_numeric(df[AXIS], errors="coerce")
    return df


def _fit(sub, theta):
    sub = sub.copy()
    sub["collapsed"] = (sub["mean_occupancy"] < theta).astype(int)
    if len(sub) < 5 or sub["collapsed"].nunique() < 2:
        return None
    m, slope, pval, r2, wc = fit_logistic(sub)
    lo, hi = bootstrap_wcrit(sub)[1:]
    return dict(n=len(sub), cells=per_w_table(sub), model=m, slope=slope,
                pval=pval, w_crit=wc, ci_lo=lo, ci_hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(CSV):
        print(f"No data at {CSV}"); return
    df = _load()
    theta = args.theta if args.theta is not None else recalibrate_theta(df)
    levels = sorted(df[AXIS].unique())

    print(f"\n  MPE horizon (metastability)  theta={theta:.3f} "
          f"({'override' if args.theta else 'recalibrated from w0/w1'})  n={len(df)}")
    print("=" * 64)
    print(f"{'horizon':>12} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | {'n':>4}")
    print("-" * 64)
    res = {}
    for h in levels:
        r = _fit(df[df[AXIS] == h], theta)
        res[h] = r
        if r:
            print(f"{int(h):>12} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | "
                  f"{r['ci_hi']:>6.3f} | {r['n']:>4}")
        else:
            print(f"{int(h):>12} |   n/a")
    print("=" * 64)

    if len(levels) >= 2 and res[levels[0]] and res[levels[-1]]:
        sh, lo = res[levels[0]], res[levels[-1]]
        slide = sh["w_crit"] - lo["w_crit"]
        nonover = lo["ci_hi"] < sh["ci_lo"]
        print(f"  slide (w_crit[T={int(levels[0])}] - w_crit[T={int(levels[-1])}]) = {slide:+.3f}")
        verdict = ("METASTABLE (slides left with T, CIs non-overlapping)" if slide > 0 and nonover
                   else "DIRECTIONAL (slides left, CIs overlap)" if slide > 0
                   else "FLAT/NO SLIDE (w_crit ~ horizon-invariant here)")
        print(f"  VERDICT: {verdict}")

    wg = np.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {levels[0]: "tab:orange", levels[-1]: "tab:blue"}
    for h in levels:
        r = res[h]
        if not r:
            continue
        c = colors.get(h, "gray")
        ax.plot(wg, curve(r["model"], wg), color=c, lw=2.2,
                label=f"T={int(h):,} ($w_{{crit}}$={r['w_crit']:.3f})")
        ax.errorbar(r["cells"]["w"], r["cells"]["frac"],
                    yerr=[r["cells"]["frac"] - r["cells"]["lo"],
                          r["cells"]["hi"] - r["cells"]["frac"]],
                    fmt="o", color=c, capsize=3, alpha=0.6)
        if np.isfinite(r["w_crit"]):
            ax.axvline(r["w_crit"], color=c, ls="--", lw=1.0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
    ax.set_title(f"MPE metastability: collapse boundary vs horizon ($\\theta$={theta:.3f})")
    ax.legend()
    fig.tight_layout()
    f = os.path.join(OUTDIR, "mpe_horizon_metastability.png")
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"  saved {f}")


if __name__ == "__main__":
    main()
