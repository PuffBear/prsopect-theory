"""
MPE/MAPPO behavioral decomposition -- full five-condition version.

Merges the base MPE sweep {null, curvature, lambda3} with the high-lambda
extension {lambda5, lambda7} and reports, per condition:
  - bootstrap w_crit (P(collapse|w) logistic midpoint) with 95% CI
  - Firth penalized slope + p (stable under steep transitions)
  - excess vs null  (w_crit_cond - w_crit_null), with a bootstrap CI on the diff

This is the MPE analogue of the supply-chain Table `tab:decomposition`. It tells
you whether the loss-aversion (rightward) effect that holds in the supply chain
replicates in MPE once high lambda is included.

Run (after the high-lambda sweep finishes):
    python experiments/analyze_mpe_decomposition.py
"""
import csv
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [
    os.path.join(ROOT, "docs/expMAPPO_mpe_figures/mappo_mpe_raw_data.csv"),
    os.path.join(ROOT, "docs/expMAPPO_mpe_highlambda_figures/mappo_mpe_highlambda_raw_data.csv"),
]
OUT_MD = os.path.join(ROOT, "docs/expMPE_combined_figures/mappo_mpe_decomposition_full.md")
COND_ORDER = ["null", "curvature", "lambda3", "lambda5", "lambda7"]
N_BOOT = 2000
RNG = np.random.default_rng(0)


def _as_bool(s):
    return 1 if str(s).strip().lower() in ("1", "true", "1.0") else 0


def load():
    by_cond = {}
    for path in SOURCES:
        if not os.path.exists(path):
            continue
        for x in csv.DictReader(open(path)):
            c = x["condition"]
            by_cond.setdefault(c, []).append((float(x["w"]), _as_bool(x["collapsed"])))
    return by_cond


def firth_logit(w, y, iters=500, tol=1e-11):
    X = np.column_stack([np.ones_like(w, dtype=float), w.astype(float)])
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        pr = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = pr * (1 - pr)
        Iinv = np.linalg.inv((X.T * W) @ X)
        h = np.einsum("ij,jk,ik->i", X, Iinv, X) * W
        step = Iinv @ (X.T @ (y - pr + h * (0.5 - pr)))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    pr = 1.0 / (1.0 + np.exp(-(X @ beta)))
    W = pr * (1 - pr)
    se = np.sqrt(np.diag(np.linalg.inv((X.T * W) @ X)))
    from math import erf, sqrt
    z = beta[1] / se[1]
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return dict(b1=beta[1], p=p, wcrit=-beta[0] / beta[1])


def _wcrit_mle(w, y):
    # plain logistic via IRLS for bootstrap point estimate; falls back to Firth
    try:
        r = firth_logit(w, y)
        return r["wcrit"]
    except Exception:
        return np.nan


def bootstrap_wcrit(pairs):
    w = np.array([a for a, _ in pairs]); y = np.array([b for _, b in pairs])
    point = _wcrit_mle(w, y)
    boots = []
    n = len(w)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        wc = _wcrit_mle(w[idx], y[idx])
        if np.isfinite(wc):
            boots.append(wc)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return point, lo, hi, np.array(boots)


def main():
    by_cond = load()
    if "null" not in by_cond:
        print("No 'null' condition found; cannot compute excess."); return
    results = {}
    for c in COND_ORDER:
        if c not in by_cond:
            continue
        pt, lo, hi, boots = bootstrap_wcrit(by_cond[c])
        w = np.array([a for a, _ in by_cond[c]]); y = np.array([b for _, b in by_cond[c]])
        fr = firth_logit(w, y)
        results[c] = dict(wcrit=pt, lo=lo, hi=hi, boots=boots, slope=fr["b1"], p=fr["p"], n=len(by_cond[c]))

    null_boots = results["null"]["boots"]
    lines = ["| condition | n | w_crit | 95% CI | Firth slope | p | excess vs null | excess 95% CI |",
             "|---|---|---|---|---|---|---|---|"]
    for c in COND_ORDER:
        if c not in results:
            continue
        r = results[c]
        if c == "null":
            exc, exc_ci = "0.000", "---"
        else:
            m = min(len(r["boots"]), len(null_boots))
            diff = r["boots"][:m] - null_boots[:m]
            elo, ehi = np.percentile(diff, [2.5, 97.5])
            exc = f"{r['wcrit'] - results['null']['wcrit']:+.3f}"
            exc_ci = f"[{elo:+.3f}, {ehi:+.3f}]"
        lines.append(f"| {c} | {r['n']} | {r['wcrit']:.3f} | "
                     f"[{r['lo']:.3f}, {r['hi']:.3f}] | {r['slope']:.1f} | "
                     f"{r['p']:.2e} | {exc} | {exc_ci} |")
    md = "# MPE/MAPPO behavioral decomposition (five conditions)\n\n" + "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w").write(md)
    print(md)
    print(f"written -> {OUT_MD}")


if __name__ == "__main__":
    main()
