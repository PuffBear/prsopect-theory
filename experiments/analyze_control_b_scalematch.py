"""
Analyze Control B -- scale-matched control. Does the lambda=1->7 w_crit shift SURVIVE
after the transformed-reward RMS is matched across lambda?

  raw arm        : w_crit(lambda) with no magnitude match (baseline).
  stdmatched arm : w_crit(lambda) with CPT reward RMS-matched to lambda=1.

If the seed-paired lambda=1->7 shift CI EXCLUDES 0 in the stdmatched arm, lambda is NOT
merely reward amplification -- the loss ASYMMETRY drives the boundary. If the shift
collapses toward 0 under matching, the effect is (partly) a magnitude artifact.

Also reports reward-side confounds (reward variance, fraction pos/neg, return variance)
per lambda, pulling from Control A/C raw CSVs when present (the "report these first" ask)
in addition to B's own logged columns.

Run: python experiments/analyze_control_b_scalematch.py
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

RAW_CSV = "docs/expControlB_scalematch_figures/control_b_scalematch_raw_data.csv"
OUTDIR  = "docs/expControlB_scalematch_figures"
A_CSV   = "docs/expControlA_refpoint_figures/control_a_refpoint_raw_data.csv"
C_CSV   = "docs/expControlC_rewardunit_figures/control_c_rewardunit_raw_data.csv"
ARMS    = ["raw", "stdmatched"]
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


def reward_side_table(df, label):
    """Reward variance, frac pos/neg, return variance per lambda (the 'report first' ask)."""
    if df is None or df.empty:
        return None
    g = df.groupby("lambda_loss")
    tbl = pd.DataFrame({
        "reward_std": g["reward_std"].mean(),
        "reward_var": g["reward_std"].mean() ** 2,
        "frac_gain": g["frac_gain"].mean(),
        "frac_loss": g["frac_loss"].mean(),
        "return_std": g["return_std"].mean(),
        "mean_cpt_reward": g["mean_cpt_reward"].mean(),
    }).round(3)
    tbl.index.name = f"lambda ({label})"
    return tbl


def main():
    if not os.path.exists(RAW_CSV):
        print(f"No data at {RAW_CSV} yet. Run the sweep + --merge first."); sys.exit(0)
    os.makedirs(OUTDIR, exist_ok=True)
    d = _coerce(pd.read_csv(RAW_CSV))
    d["lambda_loss"] = d["lambda_loss"].astype(float).round().astype(int)
    d["seed"] = d["seed"].astype(int)
    print(f"Loaded {len(d)} rows. arms={sorted(d.match_arm.unique())}")

    res = {}
    for arm in ARMS:
        da = d[d.match_arm == arm]
        if da.empty:
            continue
        wc = {lam: firth_wcrit(da[da.lambda_loss == lam]) for lam in LAMBDAS}
        res[arm] = dict(wc=wc, shift=paired_shift_ci(da), n=len(da))

    print("\n" + "=" * 78)
    print(f"{'arm':>11} | {'wc(1)':>6} | {'wc(3)':>6} | {'wc(7)':>6} | {'shift 1->7':>10} | {'CI':>16}")
    print("-" * 78)
    for arm in ARMS:
        if arm not in res:
            continue
        wc = res[arm]["wc"]; s = res[arm]["shift"]
        f = lambda v: "  n/a" if not np.isfinite(v) else f"{v:.3f}"
        print(f"{arm:>11} | {f(wc[1])} | {f(wc[3])} | {f(wc[7])} | {s[0]:>+10.3f} | "
              f"[{s[1]:+.3f}, {s[2]:+.3f}]")
    print("=" * 78)

    raw_shift = res.get("raw", {}).get("shift", (np.nan, np.nan, np.nan))
    m_shift = res.get("stdmatched", {}).get("shift", (np.nan, np.nan, np.nan))
    survives = np.isfinite(m_shift[1]) and m_shift[1] > 0
    if survives:
        verdict = (f"NOT amplification: the lambda=1->7 shift SURVIVES RMS-matching "
                   f"(stdmatched shift {m_shift[0]:+.3f} [{m_shift[1]:+.3f}, {m_shift[2]:+.3f}], "
                   f"CI excludes 0). The loss asymmetry, not reward magnitude, drives w_crit.")
    elif np.isfinite(m_shift[0]):
        verdict = (f"CONFOUNDED: the shift COLLAPSES under RMS-matching "
                   f"(stdmatched shift {m_shift[0]:+.3f} [{m_shift[1]:+.3f}, {m_shift[2]:+.3f}], "
                   "CI includes 0). Report honestly -- lambda acts partly as amplification.")
    else:
        verdict = "stdmatched arm not resolvable (degenerate); report raw + diagnostics."
    print(f"\nVERDICT: {verdict}")
    print(f"(raw-arm shift for reference: {raw_shift[0]:+.3f} [{raw_shift[1]:+.3f}, {raw_shift[2]:+.3f}])")

    # reward-side confound tables (B, plus A/C if present)
    tables = {"Control B (raw arm)": reward_side_table(d[d.match_arm == "raw"], "B-raw")}
    for path, lbl in ((A_CSV, "Control A"), (C_CSV, "Control C")):
        if os.path.exists(path):
            try:
                dd = _coerce(pd.read_csv(path))
                dd["lambda_loss"] = dd["lambda_loss"].astype(float).round().astype(int)
                tables[lbl] = reward_side_table(dd, lbl)
            except Exception:
                pass

    # figure
    fig, ax = plt.subplots(figsize=(8, 5.2))
    cmap = {"raw": "black", "stdmatched": "tab:green"}
    for arm in ARMS:
        if arm not in res:
            continue
        wc = res[arm]["wc"]
        ax.plot(LAMBDAS, [wc[l] for l in LAMBDAS], "o-", color=cmap[arm], lw=2, label=arm)
    ax.set_xlabel("loss aversion lambda"); ax.set_ylabel("w_crit (P=0.5)")
    ax.set_xticks(LAMBDAS)
    ax.set_title("Control B: w_crit(lambda), raw vs RMS-matched transformed reward")
    ax.legend(); plt.tight_layout()
    f1 = os.path.join(OUTDIR, "control_b_wcrit_raw_vs_matched.png")
    plt.savefig(f1, dpi=150); print(f"\nSaved {f1}")

    # report
    md = ["# Control B -- scale-matched control (is lambda just amplification?)\n",
          f"{len(d)} runs. Firth penalized logistic; seed-paired bootstrap n={N_BOOT}.\n",
          f"**VERDICT: {verdict}**\n",
          "| arm | w_crit(1) | w_crit(3) | w_crit(7) | shift 1->7 | 95% CI | n |",
          "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        if arm not in res:
            continue
        wc = res[arm]["wc"]; s = res[arm]["shift"]
        f = lambda v: "n/a" if not np.isfinite(v) else f"{v:.3f}"
        md.append(f"| {arm} | {f(wc[1])} | {f(wc[3])} | {f(wc[7])} | {s[0]:+.3f} | "
                  f"[{s[1]:+.3f}, {s[2]:+.3f}] | {res[arm]['n']} |")
    md.append("\n## Reward-side confounds per lambda (variance, sign fractions, return variance)\n")
    for lbl, tbl in tables.items():
        if tbl is not None:
            md.append(f"### {lbl}")
            md.append("```"); md.append(tbl.to_string()); md.append("```")
    md.append(f"\n![raw vs matched]({os.path.basename(f1)})")
    with open(os.path.join(OUTDIR, "control_b_results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'control_b_results.md')}")


if __name__ == "__main__":
    main()
