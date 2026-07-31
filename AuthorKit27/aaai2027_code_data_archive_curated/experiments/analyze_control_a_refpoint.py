"""
Analyze Control A -- does the lambda=1->7 rightward w_crit shift survive alternative
reference points z0?

For each z0 arm {zero, mean, median, neg_half_sigma}: fit w_crit(lambda) for
lambda in {1,3,7} (Firth + logistic, reusing the paper's estimators), and report the
lambda=1->7 shift. Seed-paired bootstrap (shared 20 seeds) gives the shift CI per arm.

DECISIVE READ:
  - If the lambda=1->7 shift is present (CI excludes 0) in ALL arms and of similar
    magnitude -> the loss-asymmetry interpretation is robust to the reference point.
  - If the shift changes substantially across arms -> the paper becomes about how
    reference points move coordination-collapse boundaries (still publishable).
Also reports the free diagnostics (frac_loss, mean_cpt_reward, reward_std, return_std)
per arm to interpret WHY a shift does or doesn't move.

Run: python experiments/analyze_control_a_refpoint.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import fit_logistic, _coerce
from analyze_firth_lambda_sweep import firth_logit

RAW_CSV = "docs/expControlA_refpoint_figures/control_a_refpoint_raw_data.csv"
OUTDIR  = "docs/expControlA_refpoint_figures"
ARMS    = ["zero", "mean", "median", "neg_half_sigma"]
ARM_LABEL = {"zero": "z0=0 (paper)", "mean": "z0=mean", "median": "z0=median",
             "neg_half_sigma": "z0=-0.5sigma"}
LAMBDAS = [1, 3, 7]
N_BOOT = 2000


def firth_wcrit(sub):
    if sub["collapsed"].nunique() < 2 or sub["w"].nunique() < 2:
        return np.nan
    return firth_logit(sub["w"].to_numpy(float), sub["collapsed"].to_numpy(int))["wcrit"]


def paired_shift_ci(d_arm, lam_lo=1, lam_hi=7, n_boot=N_BOOT, seed=0):
    """Seed-paired bootstrap of w_crit(lam_hi) - w_crit(lam_lo) within one arm."""
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(d_arm["seed"].unique()))
    lo = {s: d_arm[(d_arm.lambda_loss == lam_lo) & (d_arm.seed == s)] for s in seeds}
    hi = {s: d_arm[(d_arm.lambda_loss == lam_hi) & (d_arm.seed == s)] for s in seeds}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        blo = pd.concat([lo[s] for s in pick], ignore_index=True)
        bhi = pd.concat([hi[s] for s in pick], ignore_index=True)
        a, b = firth_wcrit(blo), firth_wcrit(bhi)
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(b - a)
    if not diffs:
        return (np.nan, np.nan, np.nan, 0)
    a = np.array(diffs)
    return (float(np.median(a)), float(np.percentile(a, 2.5)),
            float(np.percentile(a, 97.5)), len(diffs))


def main():
    if not os.path.exists(RAW_CSV):
        print(f"No data at {RAW_CSV} yet. Run the sweep + --merge first."); sys.exit(0)
    os.makedirs(OUTDIR, exist_ok=True)
    d = _coerce(pd.read_csv(RAW_CSV))
    d["lambda_loss"] = d["lambda_loss"].astype(float).round().astype(int)
    d["seed"] = d["seed"].astype(int)
    print(f"Loaded {len(d)} rows. arms={sorted(d.z0_arm.unique())}")

    results = {}
    for arm in ARMS:
        da = d[d.z0_arm == arm]
        if da.empty:
            continue
        wc = {lam: firth_wcrit(da[da.lambda_loss == lam]) for lam in LAMBDAS}
        shift = paired_shift_ci(da)
        z0val = float(da["z0_value"].iloc[0])
        diag = da.groupby("lambda_loss")[["frac_loss", "mean_cpt_reward",
                                          "reward_std", "return_std"]].mean()
        results[arm] = dict(wc=wc, shift=shift, z0=z0val, diag=diag, n=len(da))

    # --- table ---------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"{'arm':>16} | {'z0':>7} | {'wc(1)':>6} | {'wc(3)':>6} | {'wc(7)':>6} | "
          f"{'shift 1->7':>10} | {'95% CI':>16}")
    print("-" * 88)
    for arm in ARMS:
        if arm not in results:
            continue
        r = results[arm]; wc = r["wc"]; s = r["shift"]
        f = lambda v: "  n/a" if not np.isfinite(v) else f"{v:.3f}"
        print(f"{ARM_LABEL[arm]:>16} | {r['z0']:>7.3f} | {f(wc[1])} | {f(wc[3])} | "
              f"{f(wc[7])} | {s[0]:>+10.3f} | [{s[1]:+.3f}, {s[2]:+.3f}]")
    print("=" * 88)

    # --- verdict -------------------------------------------------------------
    survives = []
    for arm in ARMS:
        if arm not in results:
            continue
        s = results[arm]["shift"]
        if np.isfinite(s[1]) and s[1] > 0:   # CI lower bound > 0
            survives.append(arm)
    base = results.get("zero", {}).get("shift", (np.nan,))[0]
    print(f"\nSHIFT PRESENT (CI excludes 0) in arms: {[ARM_LABEL[a] for a in survives]}")
    if len(survives) == len([a for a in ARMS if a in results]):
        verdict = ("ROBUST: the lambda=1->7 shift survives in EVERY reference-point arm "
                   "-> the loss-asymmetry interpretation holds independent of z0.")
    elif "zero" in survives:
        verdict = ("PARTIAL: the shift survives at z0=0 and some but not all alternative "
                   "reference points -> reframe around how reference points move the "
                   "boundary (still publishable).")
    else:
        verdict = ("REVERSED/NULL at z0=0-adjacent arms -> report honestly; the result "
                   "is reference-point dependent.")
    print(f"VERDICT: {verdict}")

    # --- figure: w_crit(lambda) per arm --------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, arm in enumerate(ARMS):
        if arm not in results:
            continue
        wc = results[arm]["wc"]
        ys = [wc[l] for l in LAMBDAS]
        ax.plot(LAMBDAS, ys, "o-", color=cmap(i), lw=2, label=ARM_LABEL[arm])
    ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
    ax.set_xticks(LAMBDAS)
    ax.set_title("Control A: w_crit(lambda) under alternative reference points z0")
    ax.legend(); plt.tight_layout()
    f1 = os.path.join(OUTDIR, "control_a_wcrit_by_refpoint.png")
    plt.savefig(f1, dpi=150); print(f"\nSaved {f1}")

    # --- report --------------------------------------------------------------
    md = ["# Control A -- reference-point robustness of the lambda=1->7 w_crit shift\n",
          f"{len(d)} runs. Estimator: Firth penalized logistic. Seed-paired bootstrap "
          f"n={N_BOOT} (shared 20 seeds).\n",
          f"**VERDICT: {verdict}**\n",
          "| reference point | z0 | w_crit(1) | w_crit(3) | w_crit(7) | shift 1->7 | 95% CI | n |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        if arm not in results:
            continue
        r = results[arm]; wc = r["wc"]; s = r["shift"]
        f = lambda v: "n/a" if not np.isfinite(v) else f"{v:.3f}"
        md.append(f"| {ARM_LABEL[arm]} | {r['z0']:.3f} | {f(wc[1])} | {f(wc[3])} | "
                  f"{f(wc[7])} | {s[0]:+.3f} | [{s[1]:+.3f}, {s[2]:+.3f}] | {r['n']} |")
    md.append("\n## Free diagnostics per arm (mean over cells)\n")
    for arm in ARMS:
        if arm not in results:
            continue
        md.append(f"### {ARM_LABEL[arm]} (z0={results[arm]['z0']:.3f})")
        md.append("```"); md.append(results[arm]["diag"].round(3).to_string()); md.append("```")
    md.append(f"\n![w_crit by reference point]({os.path.basename(f1)})")
    with open(os.path.join(OUTDIR, "control_a_results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'control_a_results.md')}")


if __name__ == "__main__":
    main()
