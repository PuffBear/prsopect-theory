"""Generate fig1_heatmap_draft.png — P(collapse) heatmap in (w, T) space."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── load data ──────────────────────────────────────────────────────────────
BASE = "/Users/Agriya/Desktop/prospect theory/docs"
df150 = pd.read_csv(f"{BASE}/expA_figures/expA_raw_data.csv")
df300 = pd.read_csv(f"{BASE}/expA300k_figures/expA300k_raw_data.csv")

# ── compute P(collapse) + Wilson 95% CI per w ─────────────────────────────
def collapse_stats(df):
    rows = []
    z = 1.96
    for w, grp in df.groupby("w"):
        n = len(grp)
        k = grp["collapsed"].sum()
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        rows.append(dict(w=float(w), p=p, n=n,
                         lo=max(centre - half, 0.0),
                         hi=min(centre + half, 1.0)))
    return pd.DataFrame(rows).sort_values("w").reset_index(drop=True)

s150_full = collapse_stats(df150)   # all w including 0.9, 1.0
s300_full = collapse_stats(df300)

# ── intersection for heatmap display ──────────────────────────────────────
w_common = sorted(set(s150_full["w"]) & set(s300_full["w"]))  # 0.2 … 0.8

s150_c = s150_full.set_index("w")
s300_c = s300_full.set_index("w")

Z = np.array([
    [s150_c.loc[w, "p"] for w in w_common],
    [s300_c.loc[w, "p"] for w in w_common],
])
w_arr = np.array(w_common)

# ── w_crit via linear interpolation (uses full w range for each T) ─────────
def wcrit_interp(stats_df):
    """Interpolate w where P=0.5 using central and CI curves."""
    w_vals = stats_df["w"].values
    p      = stats_df["p"].values
    lo_ci  = stats_df["lo"].values
    hi_ci  = stats_df["hi"].values

    def first_crossing(curve, xs):
        for i in range(len(curve) - 1):
            if (curve[i] - 0.5) * (curve[i + 1] - 0.5) <= 0:
                frac = (0.5 - curve[i]) / (curve[i + 1] - curve[i])
                return xs[i] + frac * (xs[i + 1] - xs[i])
        return None

    wc    = first_crossing(p,     w_vals)
    # upper CI (hi) crossing gives the lower bound on w_crit
    wc_lo = first_crossing(hi_ci, w_vals)
    # lower CI (lo) crossing gives the upper bound on w_crit
    wc_hi = first_crossing(lo_ci, w_vals)
    return wc, wc_lo, wc_hi

wc150, wc150_lo, wc150_hi = wcrit_interp(s150_full)
wc300, wc300_lo, wc300_hi = wcrit_interp(s300_full)

for label, wc, lo, hi in [("150k", wc150, wc150_lo, wc150_hi),
                           ("300k", wc300, wc300_lo, wc300_hi)]:
    lo_s = f"{lo:.3f}" if lo is not None else "n/a"
    hi_s = f"{hi:.3f}" if hi is not None else "n/a"
    wc_s = f"{wc:.3f}" if wc is not None else "n/a"
    print(f"w_crit {label}: {wc_s}  [{lo_s}, {hi_s}]")

# ── figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 3.0))

cmap = mcolors.LinearSegmentedColormap.from_list(
    "white_darkred", ["#ffffff", "#8b0000"]
)

# extent: left/right edges of cells
cell_w = w_arr[1] - w_arr[0]   # 0.1
x_left  = w_arr[0]  - cell_w / 2
x_right = w_arr[-1] + cell_w / 2
# extend x-axis slightly to accommodate 150k w_crit marker if > 0.8
x_right_plot = max(x_right, (wc150 or 0) + 0.03)

im = ax.imshow(
    Z,
    aspect="auto",
    cmap=cmap,
    vmin=0, vmax=1,
    extent=[x_left, x_right, -0.5, 1.5],
    origin="lower",
    interpolation="nearest",
)

# ── P=0.5 contour drawn on a fine grid ────────────────────────────────────
w_fine = np.linspace(w_arr[0], w_arr[-1], 500)
p150_fine = np.interp(w_fine, w_arr, [s150_c.loc[w, "p"] for w in w_common])
p300_fine = np.interp(w_fine, w_arr, [s300_c.loc[w, "p"] for w in w_common])
Z_fine = np.array([p150_fine, p300_fine])
W_fine, Y_fine = np.meshgrid(w_fine, [0.0, 1.0])
ax.contour(W_fine, Y_fine, Z_fine, levels=[0.5],
           colors=["#2196F3"], linewidths=1.8, linestyles="--")

# ── w_crit markers (may fall outside heatmap cells if w_crit > 0.8) ───────
BLUE = "#2196F3"

def xerr_or_zero(wc, lo, hi):
    lo_e = max(wc - lo, 0.0) if lo is not None else 0.0
    hi_e = max(hi - wc, 0.0) if hi is not None else 0.0
    return [[lo_e], [hi_e]]

if wc150 is not None:
    ax.errorbar(wc150, 0.0,
                xerr=xerr_or_zero(wc150, wc150_lo, wc150_hi),
                fmt="o", color=BLUE, markersize=7, capsize=5,
                linewidth=1.8, zorder=6)

if wc300 is not None:
    ax.errorbar(wc300, 1.0,
                xerr=xerr_or_zero(wc300, wc300_lo, wc300_hi),
                fmt="o", color=BLUE, markersize=7, capsize=5,
                linewidth=1.8, zorder=6)

# ── axes ───────────────────────────────────────────────────────────────────
ax.set_xlim(x_left, x_right_plot)
ax.set_ylim(-0.5, 1.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(["150 k", "300 k"], fontsize=11)
ax.set_xticks(sorted(list(w_arr) + (
    [round(wc150, 2)] if wc150 is not None and wc150 > w_arr[-1] + 0.01 else []
)))
ax.set_xticklabels([f"{x:.2f}".rstrip("0").rstrip(".") if x in w_arr
                    else f"{x:.2f}*"
                    for x in ax.get_xticks()], fontsize=9)
ax.set_xlabel(r"Loss-aversion weight $w$", fontsize=12)
ax.set_ylabel("Training steps $T$", fontsize=12)
ax.set_title(
    r"$P(\mathrm{collapse})$ in $(w,\,T)$ space — draft (600 k pending)",
    fontsize=11
)

cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
cbar.set_label(r"$P(\mathrm{collapse})$", fontsize=11)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

from matplotlib.lines import Line2D
ax.legend(
    handles=[
        Line2D([0], [0], color=BLUE, lw=1.8, ls="--",
               label=r"$P=0.5$ isocline"),
        Line2D([0], [0], marker="o", color=BLUE, lw=1.8, ms=7,
               label=r"$\hat{w}_{\rm crit}$ ± 95 % CI (Wilson)"),
    ],
    loc="lower left", fontsize=9, framealpha=0.85
)

plt.tight_layout()
out = f"{BASE}/figures/fig1_heatmap_draft.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
