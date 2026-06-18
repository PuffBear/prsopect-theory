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


def series_mpe_mappo(theta=0.164):
    data = {}
    for r in rows("docs/expMAPPO_mpe_figures/mappo_mpe_raw_data.csv"):
        cond = r.get("condition")
        lam = {"curvature": 1, "lambda3": 3}.get(cond)
        if lam:
            data.setdefault(lam, []).append((float(r["w"]), 1 if float(r["mean_occupancy"]) < theta else 0))
    for r in rows("docs/expMAPPO_mpe_figures/mappo_mpe_lambda_sweep_raw_data.csv"):
        data.setdefault(int(float(r["lambda_loss"])), []).append(
            (float(r["w"]), 1 if float(r["mean_occupancy"]) < theta else 0))
    return data


def collect(data):
    lams = sorted(data); pts = []; los = []; his = []
    for l in lams:
        p, lo, hi = wcrit_ci(data[l]); pts.append(p); los.append(p - lo); his.append(hi - p)
    return np.array(lams), np.array(pts), np.array([los, his])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    specs = [
        ("Supply chain (PPO)", series_supply_ppo(), "tab:blue", "o-"),
        ("Supply chain (MAPPO)", series_supply_mappo(), "tab:green", "s--"),
        ("MPE Coop-Nav (MAPPO)", series_mpe_mappo(), "tab:red", "^:"),
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
