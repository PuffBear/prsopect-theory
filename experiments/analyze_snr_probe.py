"""
Analyze the PROBE-STATE SNR sweep (exp_snr_probe.py): does the restoring drift of
mu(s*) -- the policy's action at a FIXED inventory-position probe -- cross zero at
the empirical boundary w_crit = 0.822, where the grand-mean mu observable did NOT?

For each IP-probe level (lo / med / hi) we form the agent-averaged mu(s*)
trajectory per run, measure the restoring drift in the contested band exactly as
analyze_snr.py does (same crossing + bootstrap machinery, so the comparison with
the failed grand-mean observable is apples-to-apples), and report the w at which
the run-mean restoring drift changes sign. The legacy grand-mean mu is included
as the control that previously failed.

A probe coordinate is "predictive" if its restoring-drift sign change lands at
0.822 with the empirical boundary INSIDE the bootstrap CI. The script prints a
verdict per coordinate and writes a summary + figure.
"""
import os, glob, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same machinery as the first analyzer -- identical crossing/band/bootstrap so the
# probe coordinates are judged on exactly the test the grand mean was judged on.
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_snr import crossing, restoring_stats, boot_crossing

W_CRIT_EMPIRICAL = 0.822                 # 150k-matched boundary (see analyze_snr.py)
PROBE_LABELS = ("lo", "med", "hi")       # IP percentile probe levels (10/50/90)


def _series_for(z, key):
    """Return the per-update agent-averaged mu trajectory for a coordinate.
    key in {'lo','med','hi'} -> mu_probe[:, level, :] averaged over agents;
    key == 'grand' -> the legacy grand-mean mu (control)."""
    if key == "grand":
        return np.asarray(z["mu_grand"], float)
    level = PROBE_LABELS.index(key)
    mp = np.asarray(z["mu_probe"], float)            # [n_upd, n_probes, n_agents]
    return np.nanmean(mp[:, level, :], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/expSNR_figures/expSNR_probe_raw_data.csv")
    ap.add_argument("--out", default="docs/expSNR_figures")
    ap.add_argument("--kappa_c", type=float, default=0.0)
    ap.add_argument("--mu_coop", type=float, default=-0.40,
                    help="cooperative-optimum normalized action (matches analyze_snr.py)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    traj_dir = os.path.join(os.path.dirname(args.csv), "probe_trajectories")
    files = {int(os.path.basename(f).split("_")[1].split(".")[0]): f
             for f in glob.glob(os.path.join(traj_dir, "run_*.npz"))}
    ws = sorted(df["w"].unique())
    os.makedirs(args.out, exist_ok=True)

    coords = ["grand", "lo", "med", "hi"]
    # restoring drift per coordinate: dict coord -> {w -> [per-run drift]}
    restD = {c: {w: [] for w in ws} for c in coords}
    coll = {}
    for w in ws:
        coll[w] = df[df.w == w]["collapsed"].mean()

    for _, row in df.iterrows():
        f = files.get(int(row["run_idx"]))
        if f is None:
            continue
        z = np.load(f)
        w = float(row["w"])
        if w not in restD["grand"]:
            continue
        for c in coords:
            mu = _series_for(z, c)
            D, var, nb = restoring_stats(mu, args.mu_coop)
            restD[c][w].append(D)
    restD = {c: {w: np.array(v) for w, v in d.items()} for c, d in restD.items()}

    # collapse-fraction crossing -- the boundary-reproduction check (coord-free)
    cCF = crossing(ws, [coll[w] for w in ws], 0.5)

    print(f"Empirical structural boundary (150k)     w_crit = {W_CRIT_EMPIRICAL}")
    print(f"Collapse-fraction=0.5 crossing (check)   w = {cCF:.3f}")
    print()
    print(f"{'coordinate':>12} {'drift sign-change w':>20} {'95% CI':>20} {'w_crit in CI':>13}")
    results = {}
    for c in coords:
        cD, lo, hi = boot_crossing(restD[c], 0.0)
        inside = bool(np.isfinite(lo) and lo <= W_CRIT_EMPIRICAL <= hi)
        results[c] = (cD, lo, hi, inside)
        tag = "  <== PREDICTIVE" if inside else ""
        print(f"{c:>12} {cD:>20.3f} {f'[{lo:.3f}, {hi:.3f}]':>20} {str(inside):>13}{tag}")

    winners = [c for c in coords if c != "grand" and results[c][3]]
    print()
    if winners:
        print(f"RESULT: probe coordinate(s) {winners} locate the boundary -> "
              f"mechanism is PREDICTIVE on mu(s*) where the grand mean failed.")
    else:
        print("RESULT: no probe coordinate locates the boundary -> mechanism stays "
              "sign-and-form even on mu(s*). (grand mean control also shown.)")

    # figure: restoring drift vs w for each coordinate
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"grand": "#999999", "lo": "#1f77b4", "med": "#2e7d32", "hi": "#c00000"}
    for c in coords:
        m = [np.nanmean(restD[c][w]) if np.isfinite(restD[c][w]).any() else np.nan for w in ws]
        ax.plot(ws, m, "o-", color=colors[c],
                label=f"{c}  (cross={results[c][0]:.3f}{', PRED' if results[c][3] else ''})")
    ax.axhline(0, color="gray", lw=.8)
    ax.axvline(W_CRIT_EMPIRICAL, color="black", ls="--", label=f"$w_{{crit}}={W_CRIT_EMPIRICAL}$")
    ax.set_xlabel("centralization $w$")
    ax.set_ylabel("restoring drift of $\\mu(s^*)$ in contested band")
    ax.set_title("Probe-state Langevin coordinate vs grand-mean control")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_snr_probe.png"), dpi=160, bbox_inches="tight")

    with open(os.path.join(args.out, "snr_probe_summary.md"), "w") as f:
        f.write("# SNR probe-state validation\n\n")
        f.write(f"- Empirical structural w_crit (150k): **{W_CRIT_EMPIRICAL}**\n")
        f.write(f"- Collapse-fraction=0.5 crossing (boundary check): **{cCF:.3f}**\n\n")
        f.write("| coordinate | drift sign-change w | 95% CI | w_crit in CI |\n")
        f.write("|------------|--------------------|--------|---------------|\n")
        for c in coords:
            cD, lo, hi, inside = results[c]
            f.write(f"| {c} | {cD:.3f} | [{lo:.3f}, {hi:.3f}] | {inside} |\n")
        f.write("\n")
        f.write(f"**Verdict:** {'PREDICTIVE on ' + str(winners) if winners else 'still sign-and-form (no probe coordinate locates the boundary)'}.\n")
    print("\nwrote fig_snr_probe.png and snr_probe_summary.md")


if __name__ == "__main__":
    main()
