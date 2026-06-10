"""
Regenerate Figure 1 - the (w, T) collapse heatmap with ALL rows at alpha=beta=1.

Replaces the curvature-confounded version (whose 600k row was alpha=beta=0.88 / Exp T).
Rows are now matched:
  150k -> docs/expA_figures/expA_raw_data.csv          (alpha=beta=1, lambda=1)
  300k -> docs/expA300k_figures/expA300k_raw_data.csv   (alpha=beta=1, lambda=1)
  600k -> docs/exp0b_extended/exp0b_600k_raw_data.csv    (alpha=beta=1, lambda=1, Exp 1)

Output: docs/exp0b_extended/collapse_heatmap_matched.png
(Update the main.tex includegraphics path + caption to point here and drop the
curvature-inflation caveat once Exp 1 has run.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCES = {  # horizon -> (csv, optional lambda filter)
    "150k": ("docs/expA_figures/expA_raw_data.csv", 1.0),
    "300k": ("docs/expA300k_figures/expA300k_raw_data.csv", 1.0),
    "600k": ("docs/exp0b_extended/exp0b_600k_raw_data.csv", None),  # Exp 1 (lambda=1 only)
}
OUT = "docs/exp0b_extended/collapse_heatmap_matched.png"


def _coerce_collapsed(s):
    return s.map(lambda v: 1 if str(v).strip().lower() in ("1", "true", "1.0") else 0).astype(int)


def main():
    frac = {}        # horizon -> {w: collapse_fraction}
    for hz, (csv, lam) in SOURCES.items():
        if not os.path.exists(csv):
            print(f"[skip] {hz}: missing {csv}")
            continue
        df = pd.read_csv(csv)
        if lam is not None and "lambda" in df.columns:
            df = df[df["lambda"] == lam]
        if lam is not None and "lambda_loss" in df.columns:
            df = df[df["lambda_loss"] == lam]
        df["collapsed"] = _coerce_collapsed(df["collapsed"])
        frac[hz] = df.groupby("w")["collapsed"].mean().to_dict()

    if "600k" not in frac:
        print("\nExp 1 (600k matched) data not found yet - run experiments/exp1_null_600k.py "
              "(or submit docs/slurm/run_exp1.pbs) first. Heatmap not generated.")
        return

    horizons = [h for h in ["150k", "300k", "600k"] if h in frac]
    ws = sorted({w for h in horizons for w in frac[h]})
    M = np.full((len(horizons), len(ws)), np.nan)
    for i, h in enumerate(horizons):
        for j, w in enumerate(ws):
            if w in frac[h]:
                M[i, j] = frac[h][w]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(ws))); ax.set_xticklabels([f"{w:.2f}" for w in ws])
    ax.set_yticks(range(len(horizons))); ax.set_yticklabels(horizons)
    ax.set_xlabel("w (reward centralization)"); ax.set_ylabel("training horizon T")
    ax.set_title("Collapse fraction P(collapse) in (w, T) - matched alpha=beta=1 at every row")
    for i in range(len(horizons)):
        for j in range(len(ws)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="P(collapse)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")

    # quick success check from the spec
    for w in (0.60, 0.70):
        c3 = frac.get("300k", {}).get(w)
        c6 = frac.get("600k", {}).get(w)
        if c3 is not None and c6 is not None:
            print(f"  w={w}: 300k={c3:.2f} -> 600k={c6:.2f}  "
                  f"({'SLIDE confirmed' if c6 < c3 else 'no slide'})")


if __name__ == "__main__":
    main()
