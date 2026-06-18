"""
Comprehensive PPO supply-chain behavioral figure: the structural NULL baseline
(alpha=beta=1, lambda=1) overlaid with the FULL loss-aversion sweep
(alpha=beta=0.88, lambda=1..7). Shows in one panel:
  - curvature shifts w_crit LEFT of the null (lambda=1, ab=0.88),
  - loss aversion shifts it monotonically RIGHT (lambda=2..7),
i.e. the opposite-sign decomposition AND the full sweep together.

Replaces the old reduced {1,3,7} decomposition figure and the separate 7-curve
plot with a single complete figure.

Run: python experiments/make_ppo_behavioral_figure.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs/expPPO_lambda_sweep_figures/ppo_behavioral_full.png")


def b(s): return 1 if str(s).strip().lower() in ("1", "true", "1.0") else 0
def rows(p):
    fp = os.path.join(ROOT, p)
    return list(csv.DictReader(open(fp))) if os.path.exists(fp) else []


def firth(w, y, it=600):
    X = np.column_stack([np.ones_like(w, float), w.astype(float)]); beta = np.zeros(2)
    for _ in range(it):
        pr = 1 / (1 + np.exp(-(X @ beta))); W = pr * (1 - pr)
        Ii = np.linalg.inv((X.T * W) @ X); h = np.einsum("ij,jk,ik->i", X, Ii, X) * W
        s = Ii @ (X.T @ (y - pr + h * (0.5 - pr))); beta = beta + s
        if np.max(np.abs(s)) < 1e-11: break
    return beta


def fit(pairs):
    w = np.array([a for a, _ in pairs]); y = np.array([c for _, c in pairs])
    bt = firth(w, y); return bt, -bt[0] / bt[1]


def curve(bt, wg): return 1 / (1 + np.exp(-(bt[0] + bt[1] * wg)))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # null
    null = [(float(r["w"]), b(r["collapsed"])) for r in rows("docs/expN_figures/expN_raw_data.csv")]
    # full ab=0.88 sweep
    byl = {}
    for r in rows("docs/expB_reduced_figures/expB_reduced_raw_data.csv"):
        byl.setdefault(int(float(r["lambda_loss"])), []).append((float(r["w"]), b(r["collapsed"])))
    for r in rows("docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv"):
        byl.setdefault(int(float(r["lambda_loss"])), []).append((float(r["w"]), b(r["collapsed"])))

    wg = np.linspace(0.3, 1.0, 300)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # null baseline
    bt0, wc0 = fit(null)
    ax.plot(wg, curve(bt0, wg), color="0.35", lw=2.4, ls="--", label="null ($\\alpha{=}\\beta{=}1$)")
    ax.axvline(wc0, color="0.35", ls=":", lw=1.0)

    # lambda sweep, colour gradient
    cmap = cm.get_cmap("viridis")
    lams = sorted(byl)
    for i, lam in enumerate(lams):
        bt, wc = fit(byl[lam])
        c = cmap(i / max(1, len(lams) - 1))
        ax.plot(wg, curve(bt, wg), color=c, lw=2.0, label=f"$\\lambda={lam}$")
        ax.axvline(wc, color=c, ls=":", lw=0.8, alpha=0.7)

    ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("$w$ (reward centralization)")
    ax.set_ylabel(r"$P(\mathrm{collapse}\mid w)$")
    ax.set_title("Supply chain (PPO): null baseline vs full loss-aversion sweep\n"
                 "curvature ($\\lambda{=}1$) sits left of null; loss aversion shifts right")
    ax.legend(ncol=2, fontsize=8, frameon=True, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT, "| null w_crit=%.3f" % wc0,
          "| lambda w_crit:", {l: round(fit(byl[l])[1], 3) for l in lams})


if __name__ == "__main__":
    main()
