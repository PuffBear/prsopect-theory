"""
The bimodality/"genuine
attractor" diagnostic uses two point-estimate Sarle coefficients (0.59, 0.57). This adds, from the SAME existing raw CSVs
(no new training):
  - bootstrap CI on Sarle's bimodality coefficient at each w, both substrates
  - Hartigan's dip test (unimodal null) at each w, both substrates
Reads existing raw CSVs; NO new training.
"""
import numpy as np
import diptest

from make_bimodality_figure import load_supply, load_lbf, bimodality_coefficient

N_BOOT = 2000
RNG = np.random.default_rng(0)


def bootstrap_ci(x, stat_fn, n_boot=N_BOOT):
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        boots.append(stat_fn(x[idx]))
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return np.nan, np.nan
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def report(name, d, value_col, ceiling=None):
    print(f"\n=== {name}: Sarle BC (bootstrap 95% CI) + Hartigan dip test ===")
    for w in sorted(d.w.unique()):
        v = d[d.w == w][value_col].values
        v = v[~np.isnan(v)]
        if len(v) < 4:
            continue
        bc = bimodality_coefficient(v)
        lo, hi = bootstrap_ci(v, bimodality_coefficient)
        dip, pval = diptest.diptest(v)
        sig = "*" if pval < 0.05 else ""
        print(f"  w={w:.2f}  n={len(v):2d}  BC={bc:.3f} [{lo:.3f},{hi:.3f}]  "
              f"dip={dip:.4f}  p={pval:.3f}{sig}")
        if ceiling is not None:
            n_ceil = int((v >= ceiling).sum())
            below = v[v < ceiling]
            if n_ceil > 0 and len(below) >= 4:
                dip2, p2 = diptest.diptest(below)
                print(f"           [{n_ceil} runs at ceiling >={ceiling}; excluding them: "
                      f"dip={dip2:.4f} p={p2:.3f}]")


def main():
    supply, lbf = load_supply(), load_lbf()
    report("Supply chain (mean base-stock S)", supply, "mean_S")
    report("LBF (team return)", lbf, "mean_reward", ceiling=0.999)


if __name__ == "__main__":
    main()
