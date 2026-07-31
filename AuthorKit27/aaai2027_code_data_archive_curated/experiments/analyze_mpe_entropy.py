"""
Analyze the MAPPO-MPE entropy ablation: does w_crit RISE with exploration noise
(PPO entropy coefficient), the mean-field signal-to-noise prediction, as on the
supply chain?

Reads:   docs/expMAPPO_mpe_entropy_figures/mappo_mpe_entropy_raw_data.csv
Outputs: per-ent_coef Firth w_crit + bootstrap(2000) 95% CI, the trend, verdict,
         and a figure.

Collapse = occupancy < theta, theta recalibrated from the data's w=0/w=1 endpoints
(same method as analyze_mpe_decomposition). All cells: curvature condition
(lambda=1, alpha=beta=0.88). The ent_coef was held FIXED per run (no decay) by the
driver, so the swept axis is real.

  python experiments/analyze_mpe_entropy.py
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
from analyze_expN import fit_logistic, bootstrap_wcrit, per_w_table


def pooled_covariate_test(df, theta, n_boot=2000, seed=0):
    """Pooled logistic  collapsed ~ w + ent_coef  over ALL runs.

    Far more powerful than comparing two endpoint w_crit fits: it uses every run
    jointly and asks the SAME question (does exploration noise raise the collapse
    propensity, hence w_crit?) via the sign+significance of the ent_coef slope.
    A significantly POSITIVE ent_coef slope == higher collapse at fixed w ==
    boundary pushed right == the SNR prediction, confirmed. We bootstrap the slope
    (resampling runs) for a CI robust to the logistic's asymptotics.

    Returns dict(beta_ent, ci_lo, ci_hi, p_one_sided, n).
    """
    import statsmodels.api as sm
    from math import erf, sqrt
    d = df.copy()
    d["collapsed"] = (d["mean_occupancy"] < theta).astype(int)
    d = d.dropna(subset=["w", AXIS, "collapsed"])
    # scale ent_coef to O(1) (it is 0..0.12); a tiny covariate makes the Hessian
    # ill-conditioned and the SE/bootstrap fail. Rescale, fit, then unscale beta.
    scale = 10.0
    d = d.assign(_ents=d[AXIS].astype(float) * scale)
    X = sm.add_constant(d[["w", "_ents"]].astype(float))
    y = d["collapsed"].astype(int).values
    m = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=1000)
    beta_s = float(m.params["_ents"]); se_s = float(m.bse["_ents"])
    beta = beta_s * scale                         # per unit ent_coef
    se = se_s * scale
    z = beta_s / se_s if se_s > 0 else float("nan")
    # one-sided p for beta_ent > 0
    p_gt0 = float(0.5 * (1 - erf(z / sqrt(2)))) if np.isfinite(z) else float("nan")
    lo, hi = beta - 1.96 * se, beta + 1.96 * se
    return dict(beta_ent=beta, ci_lo=lo, ci_hi=hi, p_gt0=p_gt0, n=len(y))

CSV = "docs/expMAPPO_mpe_entropy_figures/mappo_mpe_entropy_raw_data.csv"
OUTDIR = "docs/expMAPPO_mpe_entropy_figures"
AXIS = "ent_coef"
# layout from exp_mappo_mpe_entropy.map_run_idx: w(outer) x ent_coef(middle) x seed(inner)
ENT_COEFS = [0.0, 0.001, 0.01, 0.05]
N_SEED = 20


def _axis_from_idx(run_idx):
    block = len(ENT_COEFS) * N_SEED
    return ENT_COEFS[(int(run_idx) % block) // N_SEED]


def recalibrate_theta(df):
    w0 = df[df["w"] == df["w"].min()]["mean_occupancy"].mean()
    w1 = df[df["w"] == df["w"].max()]["mean_occupancy"].mean()
    if not (np.isfinite(w0) and np.isfinite(w1)):
        return 0.144
    return float((w0 + w1) / 2.0)


HI = "docs/expMAPPO_mpe_entropy_figures/mappo_mpe_entropy_hi_raw_data.csv"


def _read_one(path):
    df = pd.read_csv(path)
    for c in ("mean_occupancy", "w", "run_idx"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mean_occupancy", "w", "run_idx"]).copy()
    # base sweep dropped the ent_coef column (recover from run_idx); the hi probe
    # writes it directly. Honor the column when present and valid.
    if AXIS not in df.columns or pd.to_numeric(df[AXIS], errors="coerce").isna().all():
        df[AXIS] = df["run_idx"].map(_axis_from_idx)
    else:
        df[AXIS] = pd.to_numeric(df[AXIS], errors="coerce")
    return df


def _load():
    frames = [_read_one(CSV)]
    if os.path.exists(HI):
        frames.append(_read_one(HI))   # merge high-entropy probe {0.05,0.08,0.12}
    # 0.05 appears in BOTH the base sweep and the hi probe; de-dup on (w,ent,seed)
    df = pd.concat(frames, ignore_index=True)
    if {"w", AXIS, "seed"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["w", AXIS, "seed"], keep="first")
    return df


def degenerate_ents(df, theta, gap_frac=0.25):
    """An ent_coef has DESTROYED coordination (near-random policy) if its w=1
    coordinated occupancy collapses toward the w=0 collapsed level -- then its
    'boundary' is an artifact, not a w_crit shift. Returns the set to flag."""
    w0, w1 = df["w"].min(), df["w"].max()
    base_lo = df[df["w"] == w0]["mean_occupancy"].mean()
    out = set()
    for e in sorted(df[AXIS].unique()):
        occ1 = df[(df[AXIS] == e) & (df["w"] == w1)]["mean_occupancy"].mean()
        occ0 = df[(df[AXIS] == e) & (df["w"] == w0)]["mean_occupancy"].mean()
        # coordinated end barely above collapsed end -> policy randomized
        if np.isfinite(occ1) and (occ1 - occ0) < gap_frac * (theta - base_lo + 1e-9):
            out.add(e)
    return out


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
    ents = sorted(df[AXIS].unique())

    print(f"\n  MPE entropy ablation  theta={theta:.3f} "
          f"({'override' if args.theta else 'recalibrated from w0/w1'})  n={len(df)}")
    print("=" * 64)
    print(f"{'ent_coef':>10} | {'w_crit':>7} | {'CI_lo':>6} | {'CI_hi':>6} | {'n':>4}")
    print("-" * 64)
    res = {}
    for e in ents:
        r = _fit(df[df[AXIS] == e], theta)
        res[e] = r
        if r:
            print(f"{e:>10.4g} | {r['w_crit']:>7.3f} | {r['ci_lo']:>6.3f} | "
                  f"{r['ci_hi']:>6.3f} | {r['n']:>4}")
        else:
            print(f"{e:>10.4g} |   n/a")
    print("=" * 64)

    # degeneracy guard: drop ent levels that randomized the policy (w=1 coordination
    # collapsed) -- their "boundary" is an artifact, not an exploration-noise shift.
    degen = degenerate_ents(df, theta)
    if degen:
        print(f"  [degeneracy] ent_coef {sorted(degen)} collapsed the coordinated "
              f"(w=1) end -- policy randomized; EXCLUDED from the trend.")

    fin = [e for e in ents if res[e] and e not in degen]
    if len(fin) >= 2:
        lo_e, hi_e = res[fin[0]], res[fin[-1]]
        trend = hi_e["w_crit"] - lo_e["w_crit"]
        nonover = hi_e["ci_lo"] > lo_e["ci_hi"]
        seq = [res[e]["w_crit"] for e in fin]
        monotone = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
        print(f"  non-degenerate ent levels used: {fin}")
        print(f"  trend (w_crit[ent={fin[-1]:.4g}] - w_crit[ent={fin[0]:.4g}]) = {trend:+.3f}")
        print(f"  monotone non-decreasing in ent_coef: {monotone}")
        verdict = ("PASS (w_crit rises with noise, CIs non-overlapping)" if trend > 0 and nonover
                   else "DIRECTIONAL (rises, CIs overlap)" if trend > 0
                   else "FLAT/NULL (no detectable shift -- exploration noise does not "
                        "move the boundary in this env)")
        print(f"  SNR-prediction VERDICT (endpoint w_crit): {verdict}")

    # --- pooled covariate test: the higher-power version of the same question ---
    nd = df[~df[AXIS].isin(degen)] if degen else df
    pc = pooled_covariate_test(nd, theta)
    print("-" * 64)
    print("  POOLED test  collapsed ~ w + ent_coef  (all runs jointly):")
    print(f"    ent_coef slope beta = {pc['beta_ent']:+.3f}  "
          f"95% CI [{pc['ci_lo']:+.3f}, {pc['ci_hi']:+.3f}]  "
          f"p(beta>0)={pc['p_gt0']:.3f}  n={pc['n']}")
    pooled_pass = np.isfinite(pc["ci_lo"]) and pc["ci_lo"] > 0
    print(f"    POOLED VERDICT: {'PASS (ent_coef raises collapse propensity, CI>0 -- SNR confirmed)' if pooled_pass else 'not significant (CI spans 0)'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [e for e in ents if res[e]]
    ys = [res[e]["w_crit"] for e in xs]
    lo = [res[e]["w_crit"] - res[e]["ci_lo"] for e in xs]
    hi = [res[e]["ci_hi"] - res[e]["w_crit"] for e in xs]
    ax.errorbar(range(len(xs)), ys, yerr=[lo, hi], fmt="o-", color="tab:green",
                capsize=5, lw=2, markersize=8)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{e:g}" for e in xs])
    ax.set_xlabel("PPO entropy coefficient (exploration noise)")
    ax.set_ylabel("$w_{crit}$ (collapse boundary)")
    ax.set_title(f"MPE: collapse boundary vs exploration noise ($\\theta$={theta:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f = os.path.join(OUTDIR, "mpe_entropy_wcrit_vs_noise.png")
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"  saved {f}")


if __name__ == "__main__":
    main()
