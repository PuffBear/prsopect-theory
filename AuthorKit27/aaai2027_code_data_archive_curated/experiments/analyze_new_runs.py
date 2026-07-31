"""
Formal analysis of the additional runbook experiments.

Reuses the SAME estimator as every other w_crit in the paper
(analyze_expN.fit_logistic + bootstrap_wcrit, n=2000), so all numbers land on
identical statistical footing.

  python experiments/analyze_new_runs.py            # all three
  python experiments/analyze_new_runs.py --only horizon|highlambda|cascade
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import fit_logistic, bootstrap_wcrit, _coerce

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HORIZON_CSV = os.path.join(ROOT, "docs/expHorizon_extra_figures/horizon_extra_raw_data.csv")
HIGHLAM_CSV = os.path.join(ROOT, "docs/expHighLambda_finegrid_figures/highlambda_finegrid_raw_data.csv")
PPO_SWEEP_CSV = os.path.join(ROOT, "docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv")
EXPB_CSV = os.path.join(ROOT, "docs/expB_reduced_figures/expB_reduced_raw_data.csv")
CASCADE_CSV = os.path.join(ROOT, "docs/expCascade_continuous_figures/cascade_continuous_raw_data.csv")

# existing structural-null boundaries at the two shorter horizons (Exp A, a=b=1),
# as reported in the paper; used as anchor points for the w_crit vs ln T fit.
ANCHORS = {150_000: 0.822, 300_000: 0.666}


def _wcrit(df):
    """w_crit + bootstrap 95% CI on a (w, collapsed) frame."""
    d = _coerce(df.rename(columns={"collapsed": "collapsed"})[["w", "collapsed"]])
    if d["collapsed"].nunique() < 2 or len(d) < 5:
        return None
    _, slope, pval, _, wc = fit_logistic(d)
    _, lo, hi = bootstrap_wcrit(d)
    return dict(w_crit=wc, lo=lo, hi=hi, slope=slope, p=pval, n=len(d))


# ----------------------------------------------------------------------------
def analyze_horizon():
    print("\n" + "=" * 70)
    print("  POINT 2 -- METASTABILITY: w_crit vs training horizon T")
    print("=" * 70)
    df = pd.read_csv(HORIZON_CSV)
    pts = dict(ANCHORS)  # T -> w_crit (anchors first)
    print(f"{'horizon':>10} | {'w_crit':>7} | {'95% CI':>16} | {'n':>4} | source")
    print("-" * 70)
    for T in (150_000, 300_000):
        print(f"{T:>10} | {ANCHORS[T]:>7.3f} | {'(paper anchor)':>16} |    - | Exp A")
    for T in sorted(df["horizon"].unique()):
        r = _wcrit(df[df["horizon"] == T])
        if r:
            pts[int(T)] = r["w_crit"]
            print(f"{int(T):>10} | {r['w_crit']:>7.3f} | "
                  f"[{r['lo']:.3f},{r['hi']:.3f}] | {r['n']:>4} | NEW")
    # w_crit vs ln T  (Kramers / metastable escape: linear in ln T)
    Ts = np.array(sorted(pts)); ys = np.array([pts[t] for t in Ts])
    x = np.log(Ts)
    A = np.column_stack([np.ones_like(x), x])
    (b0, b1), *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ np.array([b0, b1])
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    print("-" * 70)
    print(f"  w_crit(T) = {b0:+.3f} {b1:+.4f} * ln T   (4-point fit, RMSE={rmse:.4f})")
    print(f"  slope in ln T = {b1:+.4f}  (negative => boundary slides LEFT as T grows)")
    # crude asymptote read at a long horizon for intuition (NOT the formal T->inf)
    for Tx in (2_000_000, 10_000_000):
        print(f"    extrapolated w_crit at T={Tx:>9} : {b0 + b1*np.log(Tx):.3f}")


# ----------------------------------------------------------------------------
def analyze_highlambda():
    print("\n" + "=" * 70)
    print("  POINT 4 -- HIGH-LAMBDA fine grid: tightened w_crit at lambda>=5")
    print("=" * 70)
    new = pd.read_csv(HIGHLAM_CSV)
    # merge the new w=0.8/0.9 cells with the full existing lambda sweep so the
    # per-lambda fit uses every w-point (identical schema -> direct concat)
    frames = [new]
    for p in (PPO_SWEEP_CSV, EXPB_CSV):
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df["lambda_loss"] = df["lambda_loss"].astype(float).round().astype(int)
    print(f"{'lambda':>6} | {'w_crit':>7} | {'95% CI':>16} | {'CI width':>8} | {'n':>4}")
    print("-" * 70)
    for lam in (5, 6, 7):
        sub = df[df["lambda_loss"] == lam]
        r = _wcrit(sub)
        if r:
            print(f"{lam:>6} | {r['w_crit']:>7.3f} | [{r['lo']:.3f},{r['hi']:.3f}] | "
                  f"{r['hi']-r['lo']:>8.3f} | {r['n']:>4}")
    print("  (w=0.8/0.9 interior points now bracket the steep high-lambda transition;")
    print("   compare CI width to the pre-fine-grid sweep.)")


# ----------------------------------------------------------------------------
def analyze_cascade():
    print("\n" + "=" * 70)
    print("  POINT 3 -- CASCADE: position curve (Arm A) + dose-response (Arm B)")
    print("=" * 70)
    df = pd.read_csv(CASCADE_CSV)

    print("\n  ARM A -- single lambda=7 node at each chain position (downstream->upstream):")
    print(f"  {'node':>8} | {'w_crit':>7} | {'95% CI':>16} | {'n':>4}")
    print("  " + "-" * 50)
    a = df[df["cell_tag"].str.startswith("posA")]
    order = ["node_2", "node_3", "node_4", "node_5", "node_6"]
    for nd in order:
        r = _wcrit(a[a["la_node"] == nd])
        if r:
            print(f"  {nd:>8} | {r['w_crit']:>7.3f} | [{r['lo']:.3f},{r['hi']:.3f}] | {r['n']:>4}")
    print("  (expect downstream node_2 highest -> upstream falloff: 'cascades upstream')")

    print("\n  ARM B -- downstream node_2, swept lambda (dose-response):")
    print(f"  {'lambda':>6} | {'w_crit':>7} | {'95% CI':>16} | {'n':>4}")
    print("  " + "-" * 50)
    b = df[df["cell_tag"].str.startswith("lamB")]
    for lam in sorted(b["la_lambda"].unique()):
        r = _wcrit(b[b["la_lambda"] == lam])
        if r:
            print(f"  {lam:>6.0f} | {r['w_crit']:>7.3f} | [{r['lo']:.3f},{r['hi']:.3f}] | {r['n']:>4}")
    print("  (expect monotone rise toward homogeneous-LA ~0.851 as the downstream")
    print("   node's bias grows.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["horizon", "highlambda", "cascade"], default=None)
    args = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    if args.only in (None, "horizon"):
        analyze_horizon()
    if args.only in (None, "highlambda"):
        analyze_highlambda()
    if args.only in (None, "cascade"):
        analyze_cascade()


if __name__ == "__main__":
    main()
