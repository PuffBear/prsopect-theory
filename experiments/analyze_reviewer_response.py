"""
Analysis for the three reviewer-response experiments (run the moment the queued
PBS arrays finish). Each block is independent and skips gracefully if its CSV
is absent.

  Point 2 -- Shapley/difference credit assignment (exp_shapley.py)
      RQ: does structural credit assignment make the system robust to loss
      aversion? Reports collapse fraction per (credit_mode, lambda). If
      difference/Shapley credit keeps collapse ~0 even at lambda=7, the boundary
      is an artifact of scalar-w blending, not of loss aversion per se.

  Point 3 -- heterogeneous cognitive profiles (exp_heterogeneous.py)
      RQ: does a single loss-averse node shift w_crit for the whole system?
      Reports bootstrap w_crit per config vs homogeneous baselines
      (lambda=1 -> 0.586, lambda=7 -> 0.851).

  Point 4 -- entropy ablation (exp_entropy_ablation.py)
      RQ: is w_crit driven by behavioral bias or by exploration noise? Reports
      w_crit per ent_coef; the mean-field SNR account predicts w_crit rises with
      ent_coef.

Run:
    python experiments/analyze_reviewer_response.py
"""
import csv
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(0)
N_BOOT = 2000

SHAPLEY = os.path.join(ROOT, "docs/expShapley_figures/expShapley_raw_data.csv")
HET = os.path.join(ROOT, "docs/expHet_figures/expHet_raw_data.csv")
ENT = os.path.join(ROOT, "docs/expEnt_figures/expEnt_raw_data.csv")


def _b(s):
    return 1 if str(s).strip().lower() in ("1", "true", "1.0") else 0


def _read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def firth_wcrit(w, y, iters=500, tol=1e-11):
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
    return -beta[0] / beta[1]


def boot_wcrit(pairs):
    w = np.array([a for a, _ in pairs]); y = np.array([b for _, b in pairs])
    pt = firth_wcrit(w, y); n = len(w); boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        try:
            boots.append(firth_wcrit(w[idx], y[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return pt, lo, hi


def analyze_shapley():
    rows = _read(SHAPLEY)
    if not rows:
        return "## Point 2 -- Shapley/difference credit assignment\n_(no data yet)_\n"
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["credit_mode"], float(r["lambda_loss"]))].append(_b(r["collapsed"]))
    out = ["## Point 2 -- Shapley/difference credit assignment\n",
           "| credit_mode | lambda | collapse fraction | n |", "|---|---|---|---|"]
    for k in sorted(agg):
        v = agg[k]
        out.append(f"| {k[0]} | {k[1]:.0f} | {np.mean(v):.2f} | {len(v)} |")
    out.append("\n_Interpretation: if collapse fraction stays low at lambda=7 under "
               "difference/Shapley credit, the loss-aversion boundary is specific to "
               "scalar-w blending._\n")
    return "\n".join(out)


def _wcrit_table(rows, key, baselines=""):
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r[key]].append((float(r["w"]), _b(r["collapsed"])))
    out = [f"| {key} | w_crit | 95% CI | n |", "|---|---|---|---|"]
    for k in sorted(agg, key=lambda x: (len(x), x)):
        pt, lo, hi = boot_wcrit(agg[k])
        out.append(f"| {k} | {pt:.3f} | [{lo:.3f}, {hi:.3f}] | {len(agg[k])} |")
    if baselines:
        out.append(f"\n_{baselines}_")
    return "\n".join(out)


def analyze_het():
    rows = _read(HET)
    if not rows:
        return "## Point 3 -- heterogeneous cognitive profiles\n_(no data yet)_\n"
    return ("## Point 3 -- heterogeneous cognitive profiles\n\n"
            + _wcrit_table(rows, "config",
                           "Compare to homogeneous baselines: lambda=1 -> 0.586, lambda=7 -> 0.851. "
                           "A single loss-averse node shifting w_crit toward the lambda=7 baseline "
                           "is the cascade result.") + "\n")


def analyze_ent():
    rows = _read(ENT)
    if not rows:
        return "## Point 4 -- entropy ablation\n_(no data yet)_\n"
    return ("## Point 4 -- entropy ablation\n\n"
            + _wcrit_table(rows, "ent_coef",
                           "Mean-field SNR prediction: w_crit rises with ent_coef "
                           "(more exploration noise -> need more centralization).") + "\n")


def main():
    md = "# Reviewer-response analysis\n\n" + "\n".join(
        [analyze_shapley(), analyze_het(), analyze_ent()])
    out = os.path.join(ROOT, "docs/reviewer_response_analysis.md")
    open(out, "w").write(md)
    print(md)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
