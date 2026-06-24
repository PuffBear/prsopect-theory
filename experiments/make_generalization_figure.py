"""
Build the single 'generalization crux' figure: w_crit(lambda) for three settings
on one axis --- supply chain (PPO), supply chain (MAPPO), MPE Cooperative
Navigation (MAPPO) --- with bootstrap 95% CIs. This is the figure that shows the
loss-aversion DIRECTION generalizing across environment and algorithm, while
absolute levels differ.

Run: python experiments/make_generalization_figure.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(0)
OUT = os.path.join(ROOT, "docs/generalization_figures/generalization_wcrit_vs_lambda.png")


def b(s):
    return 1 if str(s).strip().lower() in ("1", "true", "1.0") else 0


def rows(p):
    return list(csv.DictReader(open(os.path.join(ROOT, p)))) if os.path.exists(os.path.join(ROOT, p)) else []


def firth(w, y, it=500):
    X = np.column_stack([np.ones_like(w, float), w.astype(float)]); beta = np.zeros(2)
    for _ in range(it):
        pr = 1 / (1 + np.exp(-(X @ beta))); W = pr * (1 - pr)
        Ii = np.linalg.inv((X.T * W) @ X); h = np.einsum("ij,jk,ik->i", X, Ii, X) * W
        s = Ii @ (X.T @ (y - pr + h * (0.5 - pr))); beta = beta + s
        if np.max(np.abs(s)) < 1e-10:
            break
    return -beta[0] / beta[1]


def wcrit_ci(pairs):
    w = np.array([a for a, _ in pairs]); y = np.array([c for _, c in pairs])
    pt = firth(w, y); bs = []
    for _ in range(2000):
        i = RNG.integers(0, len(w), len(w))
        try:
            bs.append(firth(w[i], y[i]))
        except Exception:
            pass
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def series_supply_ppo():
    data = {}
    for r in rows("docs/expB_reduced_figures/expB_reduced_raw_data.csv"):
        data.setdefault(int(float(r["lambda_loss"])), []).append((float(r["w"]), b(r["collapsed"])))
    for r in rows("docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv"):
        data.setdefault(int(float(r["lambda_loss"])), []).append((float(r["w"]), b(r["collapsed"])))
    return data


def series_supply_mappo():
    data = {}
    for r in rows("docs/expMAPPO_orgym_figures/mappo_orgym_raw_data.csv"):
        data.setdefault(int(float(r["lambda_loss"])), []).append((float(r["w"]), b(r["collapsed"])))
    return data


def series_mpe_mappo(theta=0.144):
    """MPE 20-seed re-run (AAAI rev.). Reads the SAME files and uses the SAME
    occupancy threshold (theta=0.144, recalibrated from this run's null endpoints)
    as analyze_mpe_decomposition.py, so the standalone and combined figures agree.
    Sources: v2 decomposition (curvature=lambda1, lambda3) + lambda-fill sweep
    (lambda 2,4,5,6,7). The old mappo_mpe_raw_data.csv / highlambda are NOT used
    (stale 10-seed / would double-count)."""
    data = {}
    for r in rows("docs/expMAPPO_mpe_figures/mappo_mpe_v2_raw_data.csv"):
        lam = {"curvature": 1, "lambda3": 3}.get(r.get("condition"))
        if lam:
            data.setdefault(lam, []).append(
                (float(r["w"]), 1 if float(r["mean_occupancy"]) < theta else 0))
    for r in rows("docs/expMAPPO_mpe_figures/mappo_mpe_lambda_sweep_raw_data.csv"):
        data.setdefault(int(float(r["lambda_loss"])), []).append(
            (float(r["w"]), 1 if float(r["mean_occupancy"]) < theta else 0))
    return data


def series_lbf_mappo(theta=0.626):
    """Third environment. theta=0.626 is the collapsed-vs-coordinated midpoint
    (analyze_mappo_lbf.recalibrate_theta) on the MERGED main + low-w + high-w
    sweep. The low-w extension (w in {0.0,0.1,0.2}) lands the null cliff in-grid
    and the high-w extension (w=0.90) refines the high-lambda boundary; here only
    the alpha=beta=0.88 sweep (curvature lambda=1 + loss aversion lambda=2..7) is
    plotted, on the same lambda axis as the other environments. Must match
    analyze_mappo_lbf._load() (same three CSVs) so the standalone and combined
    figures agree."""
    data = {}
    for path in ("docs/expMAPPO_lbf_figures/mappo_lbf_raw_data.csv",
                 "docs/expMAPPO_lbf_figures/mappo_lbf_loww_raw_data.csv",
                 "docs/expMAPPO_lbf_figures/mappo_lbf_highw_raw_data.csv"):
        for r in rows(path):
            if r.get("condition") == "null":
                continue
            try:
                lam = int(float(r["lambda_loss"])); mr = float(r["mean_reward"])
            except (ValueError, KeyError, TypeError):
                continue  # skip malformed/partial-write rows (blank or None fields)
            data.setdefault(lam, []).append((float(r["w"]), 1 if mr < theta else 0))
    return data


def collect(data):
    lams = sorted(data); pts = []; los = []; his = []
    for l in lams:
        p, lo, hi = wcrit_ci(data[l])
        # boundaries near the grid ceiling can bootstrap slightly above 1; clip
        # for display (the effect is saturation, not an out-of-range w_crit).
        p, lo, hi = min(p, 1.0), min(lo, 1.0), min(hi, 1.0)
        # clipping to the ceiling can invert p vs lo/hi for saturated points;
        # clamp error bars to be non-negative (matplotlib rejects negative yerr).
        pts.append(p); los.append(max(0.0, p - lo)); his.append(max(0.0, hi - p))
    return np.array(lams), np.array(pts), np.array([los, his])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    specs = [
        ("Supply chain (PPO)", series_supply_ppo(), "tab:blue", "o-"),
        ("Supply chain (MAPPO)", series_supply_mappo(), "tab:green", "s--"),
        ("MPE Coop-Nav (MAPPO)", series_mpe_mappo(), "tab:red", "^:"),
        ("LBF foraging (MAPPO)", series_lbf_mappo(), "tab:purple", "D-."),
    ]
    for label, data, color, fmt in specs:
        if not data:
            continue
        lam, pts, err = collect(data)
        ax.errorbar(lam, pts, yerr=err, fmt=fmt, color=color, capsize=4, lw=2,
                    markersize=7, label=label)
    ax.set_xlabel(r"loss aversion $\lambda$")
    ax.set_ylabel(r"$w_{\mathrm{crit}}$  (collapse boundary)")
    ax.set_title("Loss aversion raises the coordination boundary\nacross environments and algorithms")
    ax.set_xticks(range(1, 8))
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


if __name__ == "__main__":
    main()
