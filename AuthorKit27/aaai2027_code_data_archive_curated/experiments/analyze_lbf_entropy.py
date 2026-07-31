"""
Analyze the MAPPO-LBF entropy ablation: does w_crit RISE with exploration noise
(PPO entropy coefficient), the mean-field signal-to-noise prediction, as on the
supply chain?

Reads:   docs/expMAPPO_lbf_entropy_figures/mappo_lbf_entropy_raw_data.csv
Outputs: per-ent_coef Firth w_crit + bootstrap(2000) 95% CI, the trend
         (w_crit[high noise] - w_crit[low noise]), a verdict, and a figure.

Collapse label: team return < theta (LBF main-sweep calibration 0.626; --theta to
override). All cells are the curvature condition (lambda=1, alpha=beta=0.88).

  python experiments/analyze_lbf_entropy.py
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

CSV = "docs/expMAPPO_lbf_entropy_figures/mappo_lbf_entropy_raw_data.csv"
OUTDIR = "docs/expMAPPO_lbf_entropy_figures"
DEFAULT_THETA = 0.626

# layout from exp_mappo_lbf_entropy.map_run_idx: w(outer) x ent_coef(middle) x seed(inner)
ENT_COEFS = [0.0, 0.001, 0.01, 0.05]
N_SEED = 20


def _ent_from_idx(run_idx):
    """Reconstruct ent_coef from run_idx (the driver's _append_row dropped the
    'ent_coef' column on the first run; the axis is deterministic in run_idx)."""
    block = len(ENT_COEFS) * N_SEED
    return ENT_COEFS[(int(run_idx) % block) // N_SEED]


LOWW = "docs/expMAPPO_lbf_entropy_figures/mappo_lbf_entropy_loww_raw_data.csv"


def _read_one(path):
    df = pd.read_csv(path, keep_default_na=False)
    for c in ("mean_reward", "w", "run_idx"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mean_reward", "w", "run_idx"]).copy()
    if "ent_coef" not in df.columns or pd.to_numeric(df["ent_coef"], errors="coerce").isna().all():
        df["ent_coef"] = df["run_idx"].map(_ent_from_idx)
    else:
        df["ent_coef"] = pd.to_numeric(df["ent_coef"], errors="coerce")
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
    ents = sorted(df["ent_coef"].unique())

    print(f"\n  LBF entropy ablation  theta={args.theta:.3f}  n={len(df)}")
    print("=" * 64)
    print(f"{'ent_coef':>10} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | {'n':>4}")
    print("-" * 64)
    res = {}
    for e in ents:
        r = _fit(df[df["ent_coef"] == e])
        res[e] = r
        if r:
            print(f"{e:>10.4g} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | "
                  f"{r['ci_hi']:>6.3f} | {r['n']:>4}")
        else:
            print(f"{e:>10.4g} |   n/a (no in-range boundary)")
    print("=" * 64)

    # SNR prediction: w_crit RISES with exploration noise.
    # flag degenerate (sub-grid / unbounded-CI) fits so the trend isn't read off a
    # misleading extrapolated number (e.g. ent=0 -> w_crit=-0.08, CI spans -0.6).
    for e in ents:
        r = res[e]
        if r and (r["w_crit"] < 0 or r["w_crit"] > 1 or (r["ci_hi"] - r["ci_lo"]) > 0.6):
            print(f"  [note] ent={e:g}: boundary below/above grid or weakly identified "
                  f"(w_crit={r['w_crit']:.3f}, CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]) "
                  f"-- read as a bound, not a located value")

    fin = [e for e in ents if res[e]]
    if len(fin) >= 2:
        lo_e, hi_e = res[fin[0]], res[fin[-1]]
        trend = hi_e["w_crit"] - lo_e["w_crit"]
        nonover = hi_e["ci_lo"] > lo_e["ci_hi"]
        seq = [res[e]["w_crit"] for e in fin]
        monotone = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
        print(f"  trend (w_crit[ent={fin[-1]:.4g}] - w_crit[ent={fin[0]:.4g}]) = {trend:+.3f}")
        print(f"  monotone non-decreasing in ent_coef: {monotone}")
        verdict = ("PASS (w_crit rises with noise, CIs non-overlapping)" if trend > 0 and nonover
                   else "DIRECTIONAL (rises, CIs overlap)" if trend > 0
                   else "FALSIFIED (flat/decreasing)")
        print(f"  SNR-prediction VERDICT: {verdict}")

    # figure: w_crit vs ent_coef
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [e for e in ents if res[e]]
    ys = [res[e]["w_crit"] for e in xs]
    lo = [res[e]["w_crit"] - res[e]["ci_lo"] for e in xs]
    hi = [res[e]["ci_hi"] - res[e]["w_crit"] for e in xs]
    ax.errorbar(range(len(xs)), ys, yerr=[lo, hi], fmt="o-", color="tab:purple",
                capsize=5, lw=2, markersize=8)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{e:g}" for e in xs])
    ax.set_xlabel("PPO entropy coefficient (exploration noise)")
    ax.set_ylabel("$w_{crit}$ (collapse boundary)")
    ax.set_title("LBF: collapse boundary rises with exploration noise (SNR prediction)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f = os.path.join(OUTDIR, "lbf_entropy_wcrit_vs_noise.png")
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"  saved {f}")


if __name__ == "__main__":
    main()
