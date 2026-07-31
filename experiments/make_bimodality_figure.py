"""
Bimodality figure (reviewer Q1 / section 2, all three reviews): show the CONTINUOUS
behavioral metric distribution across w, to test whether "collapse" is a distinct
attractor (bimodal) or a thresholded continuum. Honest per-substrate finding:
  - Supply chain (mean base-stock S): BIMODAL -- runs sit at S~0 (lazy attractor) or
    S~coordinated, with a gap. Supports an attractor / collapse reading.
  - LBF (team return): SMOOTH -- a continuous ramp, so "collapse" there is a
    threshold-defined behavioral crossing, not a distinct regime.
Reads existing raw CSVs; NO new training.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "docs/reanalysis_controls"
os.makedirs(OUTDIR, exist_ok=True)


def load_supply():
    d = pd.concat([pd.read_csv("docs/expB_reduced_figures/expB_reduced_raw_data.csv"),
                   pd.read_csv("docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv")],
                  ignore_index=True)
    d["lambda_loss"] = d["lambda_loss"].round().astype(int)
    return d[d.lambda_loss == 1]        # risk-neutral, the primary boundary


def load_lbf():
    d = pd.read_csv("docs/expMAPPO_lbf_figures/mappo_lbf_raw_data.csv", keep_default_na=False)
    for c in ("w", "lambda_loss", "mean_reward"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["w", "lambda_loss", "mean_reward"])
    d["lambda_loss"] = d["lambda_loss"].round().astype(int)
    return d[d.lambda_loss == 1]


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient: (skew^2 + 1) / kurtosis. >0.555 suggests bimodal."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or np.std(x) == 0:
        return np.nan
    m = x.mean(); s = x.std()
    skew = np.mean(((x - m) / s) ** 3)
    kurt = np.mean(((x - m) / s) ** 4)
    g = skew; k = kurt - 3.0
    return (g * g + 1) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def main():
    supply, lbf = load_supply(), load_lbf()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Supply chain: mean_S per w (strip + violin) ---
    ax = axes[0]
    ws = sorted(supply.w.unique())
    data = [supply[supply.w == w].mean_S.values for w in ws]
    parts = ax.violinplot(data, positions=range(len(ws)), showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor("tab:blue"); pc.set_alpha(0.3)
    for i, (w, vals) in enumerate(zip(ws, data)):
        jit = np.random.uniform(-0.08, 0.08, len(vals))
        ax.scatter(i + jit, vals, s=14, color="tab:blue", alpha=0.6)
        bc = bimodality_coefficient(vals)
        ax.text(i, ax.get_ylim()[1] * 0.96 if i == 0 else max(vals) + 12,
                f"BC={bc:.2f}" if np.isfinite(bc) else "", ha="center", fontsize=7)
    ax.axhline(10, color="red", ls="--", lw=1, label=r"collapse floor $\bar S<10$")
    ax.set_xticks(range(len(ws))); ax.set_xticklabels([f"{w:g}" for w in ws])
    ax.set_xlabel("w (reward centralization)")
    ax.set_ylabel(r"mean base-stock $\bar S$ (per run)")
    ax.set_title("Supply chain: bimodal (lazy attractor vs coordinated)")
    ax.legend(fontsize=8)

    # --- LBF: mean_reward per w ---
    ax = axes[1]
    ws = sorted(lbf.w.unique())
    data = [lbf[lbf.w == w].mean_reward.values for w in ws]
    parts = ax.violinplot(data, positions=range(len(ws)), showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor("tab:orange"); pc.set_alpha(0.3)
    for i, (w, vals) in enumerate(zip(ws, data)):
        jit = np.random.uniform(-0.08, 0.08, len(vals))
        ax.scatter(i + jit, vals, s=10, color="tab:orange", alpha=0.5)
    ax.set_xticks(range(len(ws))); ax.set_xticklabels([f"{w:g}" for w in ws])
    ax.set_xlabel("w (reward centralization)")
    ax.set_ylabel("team return (per run)")
    ax.set_title("LBF: smooth continuum (threshold-defined crossing)")

    plt.tight_layout()
    p = os.path.join(OUTDIR, "bimodality_supply_vs_lbf.png")
    plt.savefig(p, dpi=150); print(f"Saved {p}")

    # --- print bimodality coefficients for the paper text ---
    print("\n=== Sarle bimodality coefficient (>0.555 => bimodal) ===")
    print("Supply chain mean_S by w:")
    for w in sorted(supply.w.unique()):
        v = supply[supply.w == w].mean_S.values
        print(f"  w={w}: BC={bimodality_coefficient(v):.3f} (n={len(v)})")
    print("LBF team return by w:")
    for w in sorted(lbf.w.unique()):
        v = lbf[lbf.w == w].mean_reward.values
        print(f"  w={w}: BC={bimodality_coefficient(v):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
