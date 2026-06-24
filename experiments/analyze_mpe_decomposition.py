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
# 20-seed re-run sources (AAAI rev.). Keyed on lambda_loss, NOT the 'condition'
# string, because the lambda-fill file has no 'condition' column. We take the
# null + curvature(lambda1) + lambda3 cells from the v2 decomposition sweep and
# lambda2,4,5,6,7 from the lambda-fill sweep -- a full 7-point 20-seed curve.
# The old highlambda file is intentionally EXCLUDED: its lambda5/7 cells overlap
# the lambda-fill sweep and would double-count.
V2_CSV  = os.path.join(ROOT, "docs/expMAPPO_mpe_figures/mappo_mpe_v2_raw_data.csv")
LAM_CSV = os.path.join(ROOT, "docs/expMAPPO_mpe_figures/mappo_mpe_lambda_sweep_raw_data.csv")
OUT_MD = os.path.join(ROOT, "docs/expMPE_combined_figures/mappo_mpe_decomposition_full.md")
# canonical row order: null baseline, then the alpha=beta=0.88 lambda sweep 1..7
COND_ORDER = ["null", "lambda1", "lambda2", "lambda3", "lambda4",
              "lambda5", "lambda6", "lambda7"]
N_BOOT = 2000
RNG = np.random.default_rng(0)


def _as_bool(s):
    return 1 if str(s).strip().lower() in ("1", "true", "1.0") else 0


def _occ(x):
    try:
        return float(x["mean_occupancy"])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def recalibrate_theta(occ_by_cond):
    """Collapse threshold = midpoint of the null condition's w=0 (collapsed) and
    w=1 (coordinated) mean occupancy, computed from THIS run. The shipped json
    theta=0.164 was calibrated on an earlier run that reached w1 occ=0.276; this
    20-seed re-run tops out at ~0.245, so the stale theta sits too high in the
    transition and pins the boundary at the grid ceiling. Recalibrating from the
    run's own endpoints (same collapsed-vs-coordinated midpoint method) lands
    theta~0.144 and puts the boundary in the interior. The loss-aversion
    DIRECTION is invariant to the choice (theta 0.13/0.144/0.164 all rise)."""
    null = occ_by_cond.get("null", [])
    if not null:
        return 0.164
    w0 = [o for w, o in null if w == 0.0 and o == o]
    w1 = [o for w, o in null if w == 1.0 and o == o]
    if not (w0 and w1):
        return 0.164
    return float((sum(w0) / len(w0) + sum(w1) / len(w1)) / 2.0)


def load():
    """Merge the 20-seed re-run into condition buckets keyed on lambda, and label
    collapse by RE-THRESHOLDING mean_occupancy at a theta recalibrated from this
    run's null endpoints (NOT the stored 'collapsed' column, which used the stale
    run-time theta=0.164).
      - v2 file: 'condition' col; 'null' baseline, 'curvature'=lambda1@0.88,
        'lambda3'=lambda3@0.88.
      - lambda-fill file: no 'condition' col; bucket each row as lambda<L>@0.88.
    """
    occ_by_cond = {}

    def add_occ(cond, x):
        occ_by_cond.setdefault(cond, []).append((float(x["w"]), _occ(x)))

    # v2 decomposition sweep (null, curvature=lambda1, lambda3)
    if os.path.exists(V2_CSV):
        for x in csv.DictReader(open(V2_CSV)):
            c = x["condition"]
            if c == "null":
                add_occ("null", x)
            elif c == "curvature":
                add_occ("lambda1", x)      # curvature condition IS lambda=1 @ 0.88
            elif c == "lambda3":
                add_occ("lambda3", x)
    # lambda-fill sweep (lambda 2,4,5,6,7 @ alpha=beta=0.88; no condition column)
    if os.path.exists(LAM_CSV):
        for x in csv.DictReader(open(LAM_CSV)):
            lam = int(round(float(x["lambda_loss"])))
            add_occ(f"lambda{lam}", x)

    # recalibrate theta from this run's null endpoints, then label collapse by
    # re-thresholding occupancy at that theta (drops NaN-occupancy rows).
    theta = recalibrate_theta(occ_by_cond)
    load.theta = theta  # expose for the report header
    by_cond = {}
    for cond, pairs in occ_by_cond.items():
        by_cond[cond] = [(w, 1 if o < theta else 0) for w, o in pairs if o == o]
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
    theta = getattr(load, "theta", 0.164)
    md = ("# MPE/MAPPO behavioral decomposition (20-seed, full lambda sweep)\n\n"
          "Null (alpha=beta=1) + curvature/loss-aversion sweep lambda=1..7 "
          "(alpha=beta=0.88). Sources: mappo_mpe_v2 + mappo_mpe_lambda_sweep.\n"
          f"Collapse threshold theta={theta:.3f} (occupancy), recalibrated from "
          "this run's null w=0/w=1 endpoints.\n\n"
          + "\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w").write(md)
    print(md)
    print(f"written -> {OUT_MD}")

    _save_lambda_curves(by_cond, results, theta)


def _save_lambda_curves(by_cond, results, theta):
    """Regenerate docs/expMAPPO_mpe_figures/mappo_mpe_lambda_curves.png (the paper
    figure) from the SAME 20-seed fits so it can never drift from the table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(ROOT, "docs/expMAPPO_mpe_figures/mappo_mpe_lambda_curves.png")
    colors = {1: "tab:blue", 2: "tab:cyan", 3: "tab:orange", 4: "tab:olive",
              5: "tab:green", 6: "tab:purple", 7: "tab:red"}
    wg = np.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lam in range(1, 8):
        c = f"lambda{lam}"
        if c not in by_cond or c not in results:
            continue
        fr = firth_logit(np.array([a for a, _ in by_cond[c]]),
                         np.array([b for _, b in by_cond[c]], float))
        # logistic curve from the Firth fit (b0 = -wcrit*b1)
        b1 = fr["b1"]; b0 = -results[c]["wcrit"] * b1
        pr = 1.0 / (1.0 + np.exp(-(b0 + b1 * wg)))
        ax.plot(wg, pr, color=colors[lam], lw=2.0, label=f"$\\lambda={lam}$")
        wc = results[c]["wcrit"]
        if np.isfinite(wc):
            ax.axvline(wc, color=colors[lam], ls="--", lw=1.0, alpha=0.7)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("$w$ (reward centralization)")
    ax.set_ylabel(r"$P(\mathrm{collapse}\mid w)$")
    ax.set_title(f"MAPPO on MPE: collapse transition by $\\lambda$ "
                 f"(20 seeds, $\\theta={theta:.3f}$)")
    ax.legend(title="loss aversion", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
