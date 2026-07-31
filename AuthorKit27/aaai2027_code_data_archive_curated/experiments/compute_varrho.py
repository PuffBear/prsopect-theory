"""
Compute the scale-invariant incentive
ratio varrho = (1-w_crit)*c_local/(w_crit*c_global) from EXISTING data (Control C's
reward-unit sweep, which already has c_local/c_global recorded per cell), no new training.

Checks the coordinate-adjusted comparison directly: do the three Control C arms (baseline cl=1,cg=1;
local2x cl=2,cg=1; global2x cl=1,cg=2), which have DIFFERENT absolute w_crit, collapse to
a CONSISTENT varrho once the known c_local/c_global are used? If yes, varrho is validated
as the portable quantity. If not, the invariance claim needs to be withdrawn or qualified.

Also reports varrho for the main structural sweep (baseline arm only, c_local=c_global=1)
across lambda, for a Table-1-style varrho column.

Reads existing raw CSVs; NO new training.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_firth_lambda_sweep import firth_logit

N_BOOT = 2000
RNG_SEED = 0

CONTROL_C_CSV = "docs/expControlC_rewardunit_figures/control_c_rewardunit_raw_data.csv"
STRUCTURAL_CSVS = [
    "docs/expB_reduced_figures/expB_reduced_raw_data.csv",
    "docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv",
]


def firth_wcrit(w, y):
    r = firth_logit(np.asarray(w, float), np.asarray(y, int))
    return r["wcrit"]


def varrho(wcrit, c_local, c_global):
    return (1.0 - wcrit) * c_local / (wcrit * c_global)


def bootstrap_varrho_ci(sub, c_local, c_global, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    seeds = sub["seed"].unique()
    w = sub["w"].to_numpy(float)
    y = sub["collapsed"].to_numpy(int)
    seed_col = sub["seed"].to_numpy()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        mask_rows = np.concatenate([np.where(seed_col == s)[0] for s in pick])
        try:
            wc = firth_wcrit(w[mask_rows], y[mask_rows])
            if np.isfinite(wc) and 0 < wc < 1:
                boots.append(varrho(wc, c_local, c_global))
        except Exception:
            continue
    boots = np.array(boots)
    if len(boots) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(boots)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def report_control_c():
    d = pd.read_csv(CONTROL_C_CSV)
    d["lambda_loss"] = d["lambda_loss"].round().astype(int)
    print("=== Control C: varrho by arm x lambda (reward-unit sweep) ===")
    print(f"{'arm':<10} {'lambda':<7} {'c_local':<8} {'c_global':<9} {'wcrit':<8} "
          f"{'varrho':<8} {'CI_lo':<8} {'CI_hi':<8}")
    rows = []
    for (arm, lam), sub in d.groupby(["scale_arm", "lambda_loss"]):
        c_local = sub["local_scale"].iloc[0]
        c_global = sub["global_scale"].iloc[0]
        wc = firth_wcrit(sub["w"], sub["collapsed"])
        rho = varrho(wc, c_local, c_global)
        med, lo, hi = bootstrap_varrho_ci(sub, c_local, c_global)
        rows.append(dict(arm=arm, lam=lam, c_local=c_local, c_global=c_global,
                          wcrit=wc, varrho=rho, ci_lo=lo, ci_hi=hi))
        print(f"{arm:<10} {lam:<7} {c_local:<8.1f} {c_global:<9.1f} {wc:<8.3f} "
              f"{rho:<8.3f} {lo:<8.3f} {hi:<8.3f}")
    out = pd.DataFrame(rows)
    os.makedirs("docs/reanalysis_controls", exist_ok=True)
    out.to_csv("docs/reanalysis_controls/varrho_control_c.csv", index=False)
    print(f"\nSaved docs/reanalysis_controls/varrho_control_c.csv")

    print("\n=== Cross-arm consistency check (same lambda, different c_local/c_global) ===")
    for lam in sorted(d["lambda_loss"].unique()):
        sub = out[out.lam == lam]
        vals = sub.set_index("arm")["varrho"]
        spread = vals.max() - vals.min()
        print(f"lambda={lam}: varrho by arm = {vals.to_dict()}  (spread={spread:.3f})")
    return out


def report_structural_sweep():
    d = pd.concat([pd.read_csv(p) for p in STRUCTURAL_CSVS if os.path.exists(p)],
                   ignore_index=True)
    d["lambda_loss"] = d["lambda_loss"].round().astype(int)
    print("\n=== Structural sweep: varrho by lambda (baseline c_local=c_global=1) ===")
    rows = []
    for lam, sub in d.groupby("lambda_loss"):
        if sub["collapsed"].nunique() < 2 or sub["w"].nunique() < 2:
            continue
        wc = firth_wcrit(sub["w"], sub["collapsed"])
        rho = varrho(wc, 1.0, 1.0)
        med, lo, hi = bootstrap_varrho_ci(sub, 1.0, 1.0)
        rows.append(dict(lam=lam, wcrit=wc, varrho=rho, ci_lo=lo, ci_hi=hi))
        print(f"lambda={lam}: wcrit={wc:.3f}  varrho={rho:.3f}  CI=[{lo:.3f},{hi:.3f}]")
    out = pd.DataFrame(rows)
    out.to_csv("docs/reanalysis_controls/varrho_structural_sweep.csv", index=False)
    print(f"\nSaved docs/reanalysis_controls/varrho_structural_sweep.csv")
    return out


if __name__ == "__main__":
    report_control_c()
    report_structural_sweep()
