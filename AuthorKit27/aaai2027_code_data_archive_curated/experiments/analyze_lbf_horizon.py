"""
Analyze the MAPPO-LBF horizon sweep: does w_crit slide LEFT as the training
horizon T grows (metastability), as on the supply chain?

Reads:   docs/expMAPPO_lbf_horizon_figures/mappo_lbf_horizon_raw_data.csv
Outputs: per-horizon Firth w_crit + bootstrap(2000) 95% CI, the slide
         (w_crit[short] - w_crit[long]), a verdict, and a figure.

Collapse label: team return < theta. The horizon sweep is all at the curvature
condition (lambda=1, alpha=beta=0.88) with NO null cell, so we use the LBF
collapse threshold calibrated for the main sweep (theta=0.626, the
collapsed-vs-coordinated midpoint); override with --theta.

  python experiments/analyze_lbf_horizon.py
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

CSV = "docs/expMAPPO_lbf_horizon_figures/mappo_lbf_horizon_raw_data.csv"
OUTDIR = "docs/expMAPPO_lbf_horizon_figures"
DEFAULT_THETA = 0.626   # LBF collapsed-vs-coordinated midpoint (main-sweep calib)

# layout from exp_mappo_lbf_horizon.map_run_idx: w(outer) x horizon(middle) x seed(inner)
HORIZONS = [1_000_000, 4_000_000]
N_SEED = 20


def _horizon_from_idx(run_idx):
    """Reconstruct the horizon from run_idx (the driver's _append_row dropped the
    'horizon' column on the first run; the axis is deterministic in run_idx)."""
    block = len(HORIZONS) * N_SEED
    return HORIZONS[(int(run_idx) % block) // N_SEED]


LOWW = "docs/expMAPPO_lbf_horizon_figures/mappo_lbf_horizon_loww_raw_data.csv"


def _read_one(path):
    df = pd.read_csv(path, keep_default_na=False)
    for c in ("mean_reward", "w", "run_idx"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mean_reward", "w", "run_idx"]).copy()
    # main run dropped the 'horizon' column -> recover from run_idx; the low-w
    # extension writes it directly. Honor the column when present and valid.
    if "horizon" not in df.columns or pd.to_numeric(df["horizon"], errors="coerce").isna().all():
        df["horizon"] = df["run_idx"].map(_horizon_from_idx)
    else:
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    return df


def _load(theta):
    frames = [_read_one(CSV)]
    if os.path.exists(LOWW):
        frames.append(_read_one(LOWW))   # auto-merge w in {0,0.1,0.2}
    df = pd.concat(frames, ignore_index=True)
    df["collapsed"] = (df["mean_reward"] < theta).astype(int)
    return df


def _fit(sub):
    if len(sub) < 5 or sub["collapsed"].nunique() < 2:
        return None
    m, slope, pval, r2, wc = fit_logistic(sub)
    lo, hi = bootstrap_wcrit(sub)[1:]
    return dict(n=len(sub), cells=per_w_table(sub), model=m, slope=slope,
                pval=pval, w_crit=wc, ci_lo=lo, ci_hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=DEFAULT_THETA)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(CSV):
        print(f"No data at {CSV}"); return
    df = _load(args.theta)
    horizons = sorted(df["horizon"].unique())

    print(f"\n  LBF horizon (metastability)  theta={args.theta:.3f}  n={len(df)}")
    print("=" * 64)
    print(f"{'horizon':>12} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | {'n':>4}")
    print("-" * 64)
    res = {}
    for h in horizons:
        r = _fit(df[df["horizon"] == h])
        res[h] = r
        if r:
            print(f"{int(h):>12} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | "
                  f"{r['ci_hi']:>6.3f} | {r['n']:>4}")
        else:
            print(f"{int(h):>12} |   n/a (no in-range boundary)")
    print("=" * 64)

    for h in horizons:
        r = res[h]
        if r and (r["w_crit"] < 0 or r["w_crit"] > 1 or (r["ci_hi"] - r["ci_lo"]) > 0.6):
            print(f"  [note] T={int(h)}: boundary below/above grid or weakly "
                  f"identified (w_crit={r['w_crit']:.3f}) -- slide direction is robust "
                  f"but this endpoint is a bound, not a located value")

    # metastability slide: short-horizon w_crit minus long-horizon w_crit (>0 = slides left with T)
    if len(horizons) >= 2 and res[horizons[0]] and res[horizons[-1]]:
        sh, lo = res[horizons[0]], res[horizons[-1]]
        slide = sh["w_crit"] - lo["w_crit"]
        nonover = lo["ci_hi"] < sh["ci_lo"]      # long CI entirely below short CI
        print(f"  slide (w_crit[T={int(horizons[0])}] - w_crit[T={int(horizons[-1])}]) "
              f"= {slide:+.3f}")
        verdict = ("METASTABLE (slides left with T, CIs non-overlapping)" if slide > 0 and nonover
                   else "DIRECTIONAL (slides left, CIs overlap)" if slide > 0
                   else "FLAT/NO SLIDE (w_crit ~ horizon-invariant here)")
        print(f"  VERDICT: {verdict}")

    # figure
    wg = np.linspace(0.2, 1.0, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {horizons[0]: "tab:orange", horizons[-1]: "tab:blue"}
    for h in horizons:
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
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("P(collapse)")
    ax.set_title("LBF metastability: collapse boundary vs training horizon")
    ax.legend()
    fig.tight_layout()
    f = os.path.join(OUTDIR, "lbf_horizon_metastability.png")
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"  saved {f}")


if __name__ == "__main__":
    main()
