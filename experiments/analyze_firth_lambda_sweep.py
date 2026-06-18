"""
Firth penalized logistic regression for the full loss-aversion sweep.

Motivation: at high lambda the collapse transition is very steep and falls
between two adjacent w-grid points, so the *unpenalized* MLE logistic slope
diverges (the ~ -150 slopes / p~0.95 seen in ppo_lambda_sweep_results.md are a
fitting artifact, NOT quasi-complete separation -- the lambda=7 w=0.85 cell is
interior, 11/20 collapsed). Firth's penalized likelihood (Jeffreys-prior penalty)
is the standard, stable fix and yields finite, significant slopes for all lambda.

Reproduces the slope/p columns in Table `tab:lambda_sweep`. No external deps
beyond numpy.

Run:
    python experiments/analyze_firth_lambda_sweep.py
"""
import csv
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    os.path.join(ROOT, "docs/expB_reduced_figures/expB_reduced_raw_data.csv"),       # lambda 1,3,7
    os.path.join(ROOT, "docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv"),  # lambda 2,4,5,6
]


def _as_bool(s):
    return 1 if str(s).strip().lower() in ("1", "true", "1.0") else 0


def load():
    rows = []
    for path in FILES:
        if not os.path.exists(path):
            continue
        for x in csv.DictReader(open(path)):
            lam = x.get("lambda_loss")
            rows.append((float(x["w"]), int(float(lam)) if lam else None, _as_bool(x["collapsed"])))
    return rows


def firth_logit(w, y, iters=500, tol=1e-11):
    """Firth penalized logistic regression for design [1, w]. Returns dict."""
    X = np.column_stack([np.ones_like(w, dtype=float), w.astype(float)])
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        pr = 1.0 / (1.0 + np.exp(-eta))
        W = pr * (1 - pr)
        I = (X.T * W) @ X
        Iinv = np.linalg.inv(I)
        # hat-matrix diagonals (leverages)
        h = np.einsum("ij,jk,ik->i", X, Iinv, X) * W
        # Firth-adjusted score
        U = X.T @ (y - pr + h * (0.5 - pr))
        step = Iinv @ U
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta
    pr = 1.0 / (1.0 + np.exp(-eta))
    W = pr * (1 - pr)
    cov = np.linalg.inv((X.T * W) @ X)
    se = np.sqrt(np.diag(cov))
    from math import erf, sqrt
    z = beta[1] / se[1]
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return dict(b0=beta[0], b1=beta[1], se1=se[1], z=z, p=p, wcrit=-beta[0] / beta[1])


def main():
    data = load()
    print(f"{'lambda':>6} {'wcrit':>7} {'slope':>9} {'SE':>7} {'z':>7} {'p':>11}")
    for lam in range(1, 8):
        sub = [(w, yy) for (w, l, yy) in data if l == lam]
        if not sub:
            continue
        w = np.array([a for a, _ in sub])
        y = np.array([b for _, b in sub])
        r = firth_logit(w, y)
        print(f"{lam:>6} {r['wcrit']:7.3f} {r['b1']:9.2f} {r['se1']:7.2f} {r['z']:7.2f} {r['p']:11.2e}")


if __name__ == "__main__":
    main()
