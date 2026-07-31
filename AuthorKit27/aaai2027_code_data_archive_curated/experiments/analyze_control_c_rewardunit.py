"""
Analyze Control C -- reward-unit sensitivity of w_crit.

Two questions:
  (i)  Does the NUMERICAL w_crit move when local/global reward is rescaled (2x) before
       the blend? (Expected YES -> w is a nominal mixture coefficient, not scale-free.)
  (ii) Does the qualitative loss-aversion effect (w_crit rises with lambda) PERSIST
       within each scale arm? (Expected YES.)

For each scale arm {baseline, local2x, global2x}: fit w_crit(lambda) for lambda{1,3,7}
(Firth), report the lambda=1->7 shift with a seed-paired bootstrap CI, and report the
w_crit MOVEMENT vs baseline at matched lambda (the reward-unit sensitivity itself).

Also reports a NORMALIZED local-to-global incentive ratio per arm at w_crit:
    ratio(w) = (1-w) * c_local / (w * c_global)
evaluated at each arm's own w_crit(lambda=1) -- the quantity the paper should report
alongside the nominal w to make the boundary comparable across reparameterizations.

Run: python experiments/analyze_control_c_rewardunit.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import _coerce
from analyze_firth_lambda_sweep import firth_logit

RAW_CSV = "docs/expControlC_rewardunit_figures/control_c_rewardunit_raw_data.csv"
OUTDIR  = "docs/expControlC_rewardunit_figures"
ARMS    = ["baseline", "local2x", "global2x"]
ARM_SCALE = {"baseline": (1.0, 1.0), "local2x": (2.0, 1.0), "global2x": (1.0, 2.0)}
LAMBDAS = [1, 3, 7]
N_BOOT  = 2000


def firth_wcrit(sub):
    if sub["collapsed"].nunique() < 2 or sub["w"].nunique() < 2:
        return np.nan
    return firth_logit(sub["w"].to_numpy(float), sub["collapsed"].to_numpy(int))["wcrit"]


def paired_shift_ci(d_arm, lam_lo=1, lam_hi=7, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(d_arm["seed"].unique()))
    lo = {s: d_arm[(d_arm.lambda_loss == lam_lo) & (d_arm.seed == s)] for s in seeds}
    hi = {s: d_arm[(d_arm.lambda_loss == lam_hi) & (d_arm.seed == s)] for s in seeds}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        a = firth_wcrit(pd.concat([lo[s] for s in pick], ignore_index=True))
        b = firth_wcrit(pd.concat([hi[s] for s in pick], ignore_index=True))
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(b - a)
    if not diffs:
        return (np.nan, np.nan, np.nan)
    a = np.array(diffs)
    return (float(np.median(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def incentive_ratio(w, c_local, c_global):
    """Normalized local-to-global incentive ratio at mixing coefficient w."""
    denom = w * c_global
    return (1.0 - w) * c_local / denom if denom > 0 else np.nan


def main():
    if not os.path.exists(RAW_CSV):
        print(f"No data at {RAW_CSV} yet. Run the sweep + --merge first."); sys.exit(0)
    os.makedirs(OUTDIR, exist_ok=True)
    d = _coerce(pd.read_csv(RAW_CSV))
    d["lambda_loss"] = d["lambda_loss"].astype(float).round().astype(int)
    d["seed"] = d["seed"].astype(int)
    print(f"Loaded {len(d)} rows. arms={sorted(d.scale_arm.unique())}")

    res = {}
    for arm in ARMS:
        da = d[d.scale_arm == arm]
        if da.empty:
            continue
        wc = {lam: firth_wcrit(da[da.lambda_loss == lam]) for lam in LAMBDAS}
        res[arm] = dict(wc=wc, shift=paired_shift_ci(da), n=len(da))

    base_wc = res.get("baseline", {}).get("wc", {})

    # --- table ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"{'arm':>9} | {'wc(1)':>6} | {'wc(3)':>6} | {'wc(7)':>6} | {'shift 1->7':>10} | "
          f"{'CI':>16} | {'ratio@wc(1)':>11}")
    print("-" * 92)
    for arm in ARMS:
        if arm not in res:
            continue
        wc = res[arm]["wc"]; s = res[arm]["shift"]
        cl, cg = ARM_SCALE[arm]
        ratio = incentive_ratio(wc[1], cl, cg) if np.isfinite(wc[1]) else np.nan
        f = lambda v: "  n/a" if not np.isfinite(v) else f"{v:.3f}"
        print(f"{arm:>9} | {f(wc[1])} | {f(wc[3])} | {f(wc[7])} | {s[0]:>+10.3f} | "
              f"[{s[1]:+.3f}, {s[2]:+.3f}] | {ratio:>11.3f}")
    print("=" * 92)

    # --- movement vs baseline + verdict --------------------------------------
    print("\nw_crit MOVEMENT vs baseline (reward-unit sensitivity):")
    moved = False
    for arm in ARMS:
        if arm == "baseline" or arm not in res:
            continue
        for lam in LAMBDAS:
            b, a = base_wc.get(lam, np.nan), res[arm]["wc"][lam]
            if np.isfinite(b) and np.isfinite(a):
                dlt = a - b
                if abs(dlt) > 0.02:
                    moved = True
                print(f"  {arm} lambda={lam}: w_crit {b:.3f} -> {a:.3f}  (delta {dlt:+.3f})")

    shifts_persist = all(np.isfinite(res[a]["shift"][1]) and res[a]["shift"][1] > 0
                         for a in res)
    if moved and shifts_persist:
        verdict = ("EXPECTED: numerical w_crit MOVES under reward reparameterization "
                   "(so w is a nominal mixture coefficient, not scale-free), BUT the "
                   "lambda=1->7 loss-aversion shift PERSISTS within every arm. Report the "
                   "normalized incentive ratio alongside w.")
    elif not moved:
        verdict = ("w_crit barely moves under 2x local/global -- the boundary is more "
                   "scale-robust than expected; report as-is.")
    else:
        verdict = ("Mixed: w_crit moves AND the loss-aversion shift weakens in some arm -- "
                   "report honestly; the effect is partly reward-scale dependent.")
    print(f"\nVERDICT: {verdict}")

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = {"baseline": "black", "local2x": "tab:blue", "global2x": "tab:red"}
    for arm in ARMS:
        if arm not in res:
            continue
        wc = res[arm]["wc"]
        ax.plot(LAMBDAS, [wc[l] for l in LAMBDAS], "o-", color=cmap[arm], lw=2, label=arm)
    ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
    ax.set_xticks(LAMBDAS)
    ax.set_title("Control C: w_crit(lambda) under 2x local / 2x global reward")
    ax.legend(); plt.tight_layout()
    f1 = os.path.join(OUTDIR, "control_c_wcrit_by_scale.png")
    plt.savefig(f1, dpi=150); print(f"\nSaved {f1}")

    # --- report --------------------------------------------------------------
    md = ["# Control C -- reward-unit sensitivity of w_crit\n",
          f"{len(d)} runs. Firth penalized logistic; seed-paired bootstrap n={N_BOOT}.\n",
          f"**VERDICT: {verdict}**\n",
          "| arm | (c_local,c_global) | w_crit(1) | w_crit(3) | w_crit(7) | shift 1->7 | CI | ratio@wc(1) | n |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        if arm not in res:
            continue
        wc = res[arm]["wc"]; s = res[arm]["shift"]; cl, cg = ARM_SCALE[arm]
        ratio = incentive_ratio(wc[1], cl, cg) if np.isfinite(wc[1]) else np.nan
        f = lambda v: "n/a" if not np.isfinite(v) else f"{v:.3f}"
        md.append(f"| {arm} | ({cl:g},{cg:g}) | {f(wc[1])} | {f(wc[3])} | {f(wc[7])} | "
                  f"{s[0]:+.3f} | [{s[1]:+.3f}, {s[2]:+.3f}] | {ratio:.3f} | {res[arm]['n']} |")
    md.append("\nThe normalized local-to-global incentive ratio  (1-w)*c_local / (w*c_global)  "
              "is the scale-invariant quantity to report alongside the nominal w.")
    md.append(f"\n![w_crit by scale]({os.path.basename(f1)})")
    with open(os.path.join(OUTDIR, "control_c_results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'control_c_results.md')}")


if __name__ == "__main__":
    main()
