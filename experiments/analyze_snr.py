"""
Analyze exp_snr_validation output: does the measured signal-to-noise crossing
land at the empirical structural boundary w_crit = 0.666?

IMPORTANT (observable choice): PPO initialises near mu~0 (base-stock S~250),
while the cooperative optimum is at S~150 (mu~-0.4) -- BELOW the init. So the
*raw* early-training drift of mu is negative for almost all w (mu falls toward
S=150 when healthy and on to S=0 when collapsing), and has no zero-crossing.
The boundary is instead the sign of the RESTORING drift in the contested band
between the cooperative and lazy fixed points: once mu dips below the
cooperative level, is it pushed back up (survive) or does it keep sliding to
the lazy corner (collapse)?  We estimate that from the saved per-run mu
trajectories (no re-training needed).

Outputs the restoring-drift crossing and SNR crossing with bootstrap CIs, the
collapse-fraction crossing as an independent check, a per-w diagnostic table,
and a figure.
"""
import os, glob, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Compare against the boundary at the MATCHED training horizon:
#   150k timesteps -> 0.822 ;  300k -> 0.666  (see Results).
# exp_snr_validation.py defaults to 150k, so the matched reference is 0.822.
W_CRIT_EMPIRICAL = 0.822


def crossing(xs, ys, level=0.0):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    for i in range(len(xs) - 1):
        a, b = ys[i] - level, ys[i + 1] - level
        if a == 0:
            return xs[i]
        if a < 0 <= b or a > 0 >= b:
            return xs[i] + (xs[i + 1] - xs[i]) * (level - ys[i]) / (ys[i + 1] - ys[i])
    return float("nan")


def restoring_stats(mu, mu_coop, lazy=-1.0, margin=0.05):
    """Mean increment (drift) and increment-variance of mu over updates that
    START in the contested band (lazy+margin, mu_coop-margin)."""
    mu = np.asarray(mu, float)
    if mu.size < 3:
        return np.nan, np.nan, 0
    d = np.diff(mu)
    start = mu[:-1]
    band = (start > lazy + margin) & (start < mu_coop - margin)
    if band.sum() < 2:
        return np.nan, np.nan, int(band.sum())
    return float(np.mean(d[band])), float(np.var(d[band], ddof=1)), int(band.sum())


def boot_crossing(per_w_values, level, n_boot=2000, seed=0):
    """per_w_values: dict w -> array of run-level scalars. Bootstrap the
    w-crossing of the run-mean curve."""
    rng = np.random.RandomState(seed)
    ws = sorted(per_w_values)
    out = []
    for _ in range(n_boot):
        means = []
        for w in ws:
            v = per_w_values[w]
            v = v[np.isfinite(v)]
            means.append(np.mean(rng.choice(v, len(v), replace=True)) if len(v) else np.nan)
        out.append(crossing(ws, means, level))
    out = np.array([c for c in out if np.isfinite(c)])
    if out.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmedian(out)), float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/expSNR_figures/expSNR_raw_data.csv")
    ap.add_argument("--out", default="docs/expSNR_figures")
    ap.add_argument("--kappa_c", type=float, default=0.0)
    ap.add_argument("--mu_coop", type=float, default=-0.40,
                    help="normalized action at the cooperative optimum (S=150 -> -0.40)")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    traj_dir = os.path.join(os.path.dirname(args.csv), "trajectories")
    os.makedirs(args.out, exist_ok=True)
    ws = sorted(df["w"].unique())

    # per-run restoring drift / snr from trajectories
    rest_D = {w: [] for w in ws}
    rest_S = {w: [] for w in ws}
    nband = {w: [] for w in ws}
    files = {int(os.path.basename(f).split("_")[1].split(".")[0]): f
             for f in glob.glob(os.path.join(traj_dir, "run_*.npz"))}
    for _, row in df.iterrows():
        f = files.get(int(row["run_idx"]))
        if f is None:
            continue
        z = np.load(f)
        D, var, nb = restoring_stats(z["mu"], args.mu_coop)
        w = float(row["w"])
        if w in rest_D:
            rest_D[w].append(D)
            rest_S[w].append(D / np.sqrt(var) if (var and var > 0) else np.nan)
            nband[w].append(nb)
    rest_D = {w: np.array(v) for w, v in rest_D.items()}
    rest_S = {w: np.array(v) for w, v in rest_S.items()}

    # diagnostics table
    print(f"{'w':>6} {'n':>3} {'collapse':>8} {'net_disp':>9} "
          f"{'restD':>9} {'restSNR':>8} {'band_n':>6}")
    coll = {}
    for w in ws:
        sub = df[df.w == w]
        cfrac = sub["collapsed"].mean()
        coll[w] = cfrac
        nd = sub["net_displacement"].mean()
        rd = np.nanmean(rest_D[w]) if len(rest_D[w]) else np.nan
        rs = np.nanmean(rest_S[w]) if len(rest_S[w]) else np.nan
        bn = int(np.nanmean(nband[w])) if len(nband[w]) else 0
        print(f"{w:>6.3f} {len(sub):>3d} {cfrac:>8.2f} {nd:>9.3f} "
              f"{rd:>9.4f} {rs:>8.3f} {bn:>6d}")

    cD, cD_lo, cD_hi = boot_crossing(rest_D, 0.0)
    cS, cS_lo, cS_hi = boot_crossing(rest_S, args.kappa_c)
    # collapse-fraction crossing 0.5 (independent reproduction of the boundary)
    cf_curve = [coll[w] for w in ws]
    cCF = crossing(ws, cf_curve, 0.5)

    print()
    print(f"Empirical structural boundary           w_crit = {W_CRIT_EMPIRICAL}")
    print(f"Restoring-drift sign change             w = {cD:.3f}  [{cD_lo:.3f}, {cD_hi:.3f}]")
    print(f"Restoring-SNR=kappa_c crossing          w = {cS:.3f}  [{cS_lo:.3f}, {cS_hi:.3f}]")
    print(f"Collapse-fraction=0.5 crossing (check)  w = {cCF:.3f}")
    inside = (np.isfinite(cD_lo) and cD_lo <= W_CRIT_EMPIRICAL <= cD_hi)
    print(f"Empirical w_crit inside restoring-drift CI: {inside}")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    rd_m = [np.nanmean(rest_D[w]) for w in ws]
    rd_lo = [np.nanpercentile(rest_D[w][np.isfinite(rest_D[w])], 16) if np.isfinite(rest_D[w]).any() else np.nan for w in ws]
    rd_hi = [np.nanpercentile(rest_D[w][np.isfinite(rest_D[w])], 84) if np.isfinite(rest_D[w]).any() else np.nan for w in ws]
    ax[0].axhline(0, color="gray", lw=.8)
    ax[0].plot(ws, rd_m, "o-", color="#1f4e79")
    ax[0].fill_between(ws, rd_lo, rd_hi, color="#1f4e79", alpha=.15)
    ax[0].axvline(W_CRIT_EMPIRICAL, color="#c00000", ls="--", label="empirical $w_{crit}=0.666$")
    if np.isfinite(cD):
        ax[0].axvline(cD, color="#2e7d32", ls=":", label=f"measured crossing={cD:.3f}")
    ax[0].set_xlabel("centralization $w$"); ax[0].set_ylabel("restoring drift in contested band")
    ax[0].set_title("Restoring drift (Langevin, from PPO)"); ax[0].legend(fontsize=8)

    ax[1].plot(ws, cf_curve, "s-", color="#7030a0")
    ax[1].axhline(0.5, color="gray", lw=.8)
    ax[1].axvline(W_CRIT_EMPIRICAL, color="#c00000", ls="--", label="empirical $w_{crit}=0.666$")
    if np.isfinite(cCF):
        ax[1].axvline(cCF, color="#2e7d32", ls=":", label=f"collapse-0.5={cCF:.3f}")
    ax[1].set_xlabel("centralization $w$"); ax[1].set_ylabel("collapse fraction")
    ax[1].set_title("Boundary reproduced by this sweep"); ax[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_snr_validation.png"), dpi=160, bbox_inches="tight")

    with open(os.path.join(args.out, "snr_summary.md"), "w") as f:
        f.write("# SNR validation\n\n")
        f.write(f"- Empirical structural w_crit: **{W_CRIT_EMPIRICAL}**\n")
        f.write(f"- Restoring-drift sign change: **{cD:.3f}** [{cD_lo:.3f}, {cD_hi:.3f}]\n")
        f.write(f"- Restoring-SNR crossing: **{cS:.3f}** [{cS_lo:.3f}, {cS_hi:.3f}]\n")
        f.write(f"- Collapse-fraction=0.5 (check): **{cCF:.3f}**\n")
        f.write(f"- Empirical w_crit inside restoring-drift CI: **{inside}**\n")
    print("wrote fig_snr_validation.png and snr_summary.md")


if __name__ == "__main__":
    main()
