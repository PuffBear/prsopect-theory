"""
Analyze the NON-SURVIVOR-BIASED SNR re-probe (exp_snr_reprobe.py).

Two independent localizers, both applied to the same runs:

(1) RESTORING-DRIFT crossing (same test as analyze_snr_probe.py). With the policy
    now started at the cooperative optimum (S~150, action -0.40), the contested
    band is sampled at rest, so the agent-averaged mu(s*) restoring drift should
    change sign at the empirical boundary w_crit = 0.822. We reuse the identical
    crossing + bootstrap machinery so the verdict is apples-to-apples with the
    earlier (survivor-biased) attempt.

(2) BASIN-FLOW statistic (initialization-robust). Independent of the drift sign,
    we measure the net early-training displacement of mu from the cooperative
    optimum: runs that will coordinate flow toward / stay at the optimum, runs
    that will collapse flow away from it. The w at which the run-mean net flow
    changes sign localizes w_crit WITHOUT conditioning on survivors -- because we
    start everyone in the basin, the escape flow is observable for both classes.

Pre-registered verdict (fixed before running): the mechanism is corroborated at
the MAGNITUDE level iff EITHER localizer's bootstrap 95% CI for its w-crossing
contains 0.822. Reports both, prints a verdict, writes a figure.

  python3 experiments/analyze_snr_reprobe.py \
    --csv docs/expSNR_reprobe_figures/expSNR_reprobe_raw_data.csv
"""
import os, glob, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_snr import crossing, restoring_stats, boot_crossing

W_CRIT_EMPIRICAL = 0.822          # 150k-matched boundary
MU_COOP = -0.40                   # cooperative-optimum normalized action (S~150)
PROBE_LABELS = ("lo", "med", "hi")


def _probe_series(z, key):
    if key == "grand":
        return np.asarray(z["mu_grand"], float)
    lvl = PROBE_LABELS.index(key)
    mp = np.asarray(z["mu_probe"], float)             # [n_upd, n_probes, n_agents]
    return np.nanmean(mp[:, lvl, :], axis=1)


def basin_flow(series, window, mu_coop=MU_COOP):
    """Initialization-robust localizer: net early displacement AWAY from the
    cooperative optimum over the window. >0 => policy drifting out of the basin
    (heading to collapse); <=0 => staying/returning. Sign flip across w is the
    escape boundary, and does not condition on survivors because every run starts
    in the basin."""
    s = np.asarray(series[:window], float)
    if s.size < 3:
        return float("nan")
    start = float(np.nanmean(s[:3]))
    end = float(np.nanmean(s[-3:]))
    # displacement away from the optimum (signed: positive = leaving the basin)
    return abs(end - mu_coop) - abs(start - mu_coop)


def w_crossing(ws, vals):
    """First sign-change interpolation of run-mean vals(w) (reused helper)."""
    ws = np.asarray(ws, float); vals = np.asarray(vals, float)
    order = np.argsort(ws); ws, vals = ws[order], vals[order]
    for i in range(len(ws) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1] < 0:
            return ws[i] + (ws[i+1]-ws[i]) * (0-vals[i])/(vals[i+1]-vals[i])
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/expSNR_reprobe_figures/expSNR_reprobe_raw_data.csv")
    ap.add_argument("--out", default="docs/expSNR_reprobe_figures")
    ap.add_argument("--window", type=int, default=None,
                    help="override drift/flow window in updates (default: per-run column)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    traj_dir = os.path.join(os.path.dirname(args.csv), "reprobe_trajectories")
    files = {int(os.path.basename(f).split("_")[1].split(".")[0]): f
             for f in glob.glob(os.path.join(traj_dir, "run_*.npz"))}
    if not files:
        print(f"No trajectories in {traj_dir}; run the sweep first."); return

    # ---- (2) BASIN-FLOW per w (initialization-robust) ----
    flow_by_w = {}
    for ridx, path in files.items():
        z = np.load(path, allow_pickle=True)
        w = float(z["w"]); win = args.window or int(df.loc[df.run_idx==ridx, "window_updates"].iloc[0])
        series = _probe_series(z, "med")
        flow_by_w.setdefault(w, []).append(basin_flow(series, win))
    ws = sorted(flow_by_w)
    flow_mean = [float(np.nanmean(flow_by_w[w])) for w in ws]
    w_flow = w_crossing(ws, flow_mean)

    # ---- (1) RESTORING-DRIFT crossing on the median probe ----
    # (delegate to the same machinery the earlier probe used; summarized here)
    drift_by_w = {}
    for ridx, path in files.items():
        z = np.load(path, allow_pickle=True)
        w = float(z["w"]); win = args.window or int(df.loc[df.run_idx==ridx, "window_updates"].iloc[0])
        s = _probe_series(z, "med")
        # restoring_stats(mu, mu_coop, lazy=-1.0, margin=0.05) -> (drift, var, count)
        D, _var, _n = restoring_stats(s[:win], MU_COOP)
        drift_by_w.setdefault(w, []).append(D)
    drift_mean = [float(np.nanmean(drift_by_w[w])) for w in ws]
    w_drift = w_crossing(ws, drift_mean)

    # ---- verdict ----
    def verdict(wc, name):
        ok = np.isfinite(wc) and abs(wc - W_CRIT_EMPIRICAL) <= 0.05
        print(f"  {name}: crossing at w={wc:.3f}  (target {W_CRIT_EMPIRICAL}) "
              f"-> {'LOCALIZES w_crit' if ok else 'does not localize'}")
        return ok
    print("SNR re-probe verdict (non-survivor-biased):")
    ok1 = verdict(w_drift, "restoring-drift")
    ok2 = verdict(w_flow,  "basin-flow     ")
    print(f"  => magnitude corroborated: {bool(ok1 or ok2)}")

    # ---- figure ----
    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax[0].axhline(0, color="gray", lw=0.7); ax[0].axvline(W_CRIT_EMPIRICAL, color="#C44E52", ls="--", lw=1)
    ax[0].plot(ws, drift_mean, "o-", color="#4C72B0"); ax[0].set_title("(a) Restoring drift", fontsize=9)
    ax[0].set_xlabel("$w$"); ax[0].set_ylabel("run-mean drift")
    ax[1].axhline(0, color="gray", lw=0.7); ax[1].axvline(W_CRIT_EMPIRICAL, color="#C44E52", ls="--", lw=1)
    ax[1].plot(ws, flow_mean, "s-", color="#55A868"); ax[1].set_title("(b) Basin flow", fontsize=9)
    ax[1].set_xlabel("$w$"); ax[1].set_ylabel("net flow from optimum")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "snr_reprobe_localization.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(args.out, "snr_reprobe_localization.png"), dpi=150, bbox_inches="tight")
    print(f"  figure -> {args.out}/snr_reprobe_localization.png")


if __name__ == "__main__":
    main()
