"""
3D visualisation of the collapse boundary in (w, lambda, T) space.

Combines all available experimental data:
  - Exp A   @150k  : T=150k,  lambda=1,  w in {0.0..1.0}   (alpha=beta=0.88)
  - Exp A   @300k  : T=300k,  lambda=1,  w in {0.2..0.8}   (alpha=beta=0.88)
  - Exp B-Reduced @300k: T=300k, lambda in {1,3,7}          (alpha=beta=0.88)
  - PPO lambda sweep @300k: T=300k, lambda in {2,4,5,6}     (alpha=beta=0.88)
  - Exp 0b  @600k  : T=600k,  lambda=1,  w in {0.45..0.80} (alpha=beta=1)

Three panels:
  Panel 1 -- 3D scatter: axes (lambda, T, w), colour = collapse fraction
             per cell.  Each bubble is one (w, lambda, T) cell; size ~
             number of seeds, colour = collapse fraction (0=blue, 1=red).
  Panel 2 -- 3D surface: same data interpolated on a regular grid using
             linear interpolation (only at T=300k where we have full lambda
             coverage) plus the T=150k / T=600k lambda=1 ridge.
  Panel 3 -- w_crit vs lambda, coloured by T horizon (logistic fit points
             with error bars, reprojected onto a 3D axes for context).
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d proj)
from scipy.interpolate import griddata
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

OUT_DIR = "docs/expPPO_lambda_sweep_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load & harmonise data
# ---------------------------------------------------------------------------

def _load(path, t_val, lam_col=None, default_lam=1):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    df["T"] = t_val
    # normalise collapsed to 0/1 int
    df["collapsed"] = df["collapsed"].map(
        lambda v: 1 if str(v).strip().lower() in ("true", "1") else 0
    ).astype(int)
    if lam_col and lam_col in df.columns:
        df["lambda"] = df[lam_col].astype(float)
    elif "lambda" in df.columns:
        df["lambda"] = df["lambda"].astype(float)
    else:
        df["lambda"] = float(default_lam)
    return df[["w", "lambda", "T", "collapsed"]]


frames = []

a150 = _load("docs/expA_figures/expA_raw_data.csv",         150_000)
a300 = _load("docs/expA300k_figures/expA300k_raw_data.csv", 300_000)
bred = _load("docs/expB_reduced_figures/expB_reduced_raw_data.csv", 300_000, lam_col="lambda_loss")
lsw  = _load("docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv", 300_000, lam_col="lambda_loss")
e600 = _load("docs/exp0b_extended/exp0b_600k_raw_data.csv", 600_000)

for df in [a150, a300, bred, lsw, e600]:
    if df is not None:
        frames.append(df)

data = pd.concat(frames, ignore_index=True)
print(f"Combined: {len(data)} rows")
print("  T values  :", sorted(data["T"].unique()))
print("  lambda vals:", sorted(data["lambda"].unique()))
print("  w vals     :", sorted(data["w"].unique()))

# ---------------------------------------------------------------------------
# 2. Per-cell collapse fraction
# ---------------------------------------------------------------------------

cells = (
    data.groupby(["w", "lambda", "T"])["collapsed"]
    .agg(frac="mean", n="count")
    .reset_index()
)
print(f"\nCells (w, lambda, T): {len(cells)}")

# ---------------------------------------------------------------------------
# 3. Logistic w_crit per (lambda, T) slice
# ---------------------------------------------------------------------------

def logistic_wcrit(sub):
    """Fit logistic P(collapse|w) and return w_crit = -b0/b1."""
    if sub["collapsed"].nunique() < 2:
        return np.nan
    X = sub[["w"]].values
    y = sub["collapsed"].values
    try:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        clf.fit(X, y)
        b0, b1 = float(clf.intercept_[0]), float(clf.coef_[0][0])
        if abs(b1) < 1e-9:
            return np.nan
        return -b0 / b1
    except Exception:
        return np.nan


wcrit_rows = []
for (lam, T), sub in data.groupby(["lambda", "T"]):
    wc = logistic_wcrit(sub)
    if not np.isnan(wc) and 0.0 < wc < 1.0:
        wcrit_rows.append({"lambda": lam, "T": T, "w_crit": wc})

wcrit_df = pd.DataFrame(wcrit_rows)
print("\nw_crit estimates:")
print(wcrit_df.sort_values(["T", "lambda"]).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(20, 7))
fig.patch.set_facecolor("#0d0d0d")

T_labels = {150_000: "150k", 300_000: "300k", 600_000: "600k"}
T_vals   = sorted(data["T"].unique())
T_colors = {150_000: "#4fc3f7", 300_000: "#ff8a65", 600_000: "#a5d6a7"}
cmap     = cm.RdYlBu_r          # blue=0 (no collapse) -> red=1 (full collapse)

# --- Panel 1: 3D scatter of all (w, lambda, T) cells ----------------------
ax1 = fig.add_subplot(131, projection="3d")
ax1.set_facecolor("#0d0d0d")

sc = ax1.scatter(
    cells["lambda"], cells["T"] / 1e5, cells["w"],
    c=cells["frac"], cmap=cmap, vmin=0, vmax=1,
    s=cells["n"] * 8 + 20,
    alpha=0.85, edgecolors="white", linewidths=0.3, depthshade=True,
)
cb = fig.colorbar(sc, ax=ax1, pad=0.08, shrink=0.6)
cb.set_label("Collapse fraction", color="white", fontsize=8)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white", fontsize=7)

ax1.set_xlabel("lambda", color="white", labelpad=6, fontsize=9)
ax1.set_ylabel("T (x10^5 steps)", color="white", labelpad=6, fontsize=9)
ax1.set_zlabel("w", color="white", labelpad=6, fontsize=9)
ax1.set_title("Collapse fraction in (w, lambda, T)", color="white", fontsize=10, pad=10)
ax1.tick_params(colors="white", labelsize=7)
ax1.xaxis.pane.fill = ax1.yaxis.pane.fill = ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor("#333333")
ax1.yaxis.pane.set_edgecolor("#333333")
ax1.zaxis.pane.set_edgecolor("#333333")
ax1.grid(True, color="#2a2a2a", linewidth=0.4)
ax1.set_zticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax1.view_init(elev=22, azim=-50)

# --- Panel 2: T=300k surface + lambda=1 ridge ------------------------------
ax2 = fig.add_subplot(132, projection="3d")
ax2.set_facecolor("#0d0d0d")

# Surface at T=300k: interpolate collapse fraction on fine grid
t300 = cells[cells["T"] == 300_000].copy()
if len(t300) >= 4:
    lam_g = np.linspace(t300["lambda"].min(), t300["lambda"].max(), 60)
    w_g   = np.linspace(t300["w"].min(), t300["w"].max(), 60)
    LL, WW = np.meshgrid(lam_g, w_g)
    ZZ = griddata(
        t300[["lambda", "w"]].values,
        t300["frac"].values,
        (LL, WW), method="linear"
    )
    surf = ax2.plot_surface(
        LL, WW, np.full_like(LL, 3.0),   # T=300k plane at z=3.0
        facecolors=cmap(ZZ),
        alpha=0.75, linewidth=0, antialiased=True,
    )
    # colour-bar proxy
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb2 = fig.colorbar(sm, ax=ax2, pad=0.08, shrink=0.6)
    cb2.set_label("Collapse fraction", color="white", fontsize=8)
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb2.ax.axes, "yticklabels"), color="white", fontsize=7)

# lambda=1 ridge across T values
lam1 = cells[cells["lambda"] == 1.0].sort_values("T")
ax2.plot(
    [1.0] * len(lam1), lam1["w"], lam1["T"] / 1e5,
    color="#ffeb3b", linewidth=2.5, zorder=10, label="lambda=1 ridge"
)
ax2.scatter(
    [1.0] * len(lam1), lam1["w"], lam1["T"] / 1e5,
    c=lam1["frac"], cmap=cmap, vmin=0, vmax=1,
    s=60, edgecolors="#ffeb3b", linewidths=1.0, zorder=11,
)

ax2.set_xlabel("lambda", color="white", labelpad=6, fontsize=9)
ax2.set_ylabel("w", color="white", labelpad=6, fontsize=9)
ax2.set_zlabel("T (x10^5 steps)", color="white", labelpad=6, fontsize=9)
ax2.set_title("T=300k surface  +  lambda=1 ridge", color="white", fontsize=10, pad=10)
ax2.tick_params(colors="white", labelsize=7)
ax2.xaxis.pane.fill = ax2.yaxis.pane.fill = ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor("#333333")
ax2.yaxis.pane.set_edgecolor("#333333")
ax2.zaxis.pane.set_edgecolor("#333333")
ax2.grid(True, color="#2a2a2a", linewidth=0.4)
ax2.view_init(elev=28, azim=45)

# --- Panel 3: w_crit surface in (lambda, T) space -------------------------
ax3 = fig.add_subplot(133, projection="3d")
ax3.set_facecolor("#0d0d0d")

if len(wcrit_df) >= 3:
    # scatter of w_crit points
    sc3 = ax3.scatter(
        wcrit_df["lambda"], wcrit_df["T"] / 1e5, wcrit_df["w_crit"],
        c=wcrit_df["T"], cmap="plasma",
        s=80, edgecolors="white", linewidths=0.5, zorder=5,
    )

    # interpolate w_crit surface if enough T=300k lambda points
    wc300 = wcrit_df[wcrit_df["T"] == 300_000]
    if len(wc300) >= 3:
        lam_f = np.linspace(wc300["lambda"].min(), wc300["lambda"].max(), 60)
        wc_f  = np.interp(lam_f, wc300["lambda"].values, wc300["w_crit"].values)
        ax3.plot(lam_f, [3.0] * len(lam_f), wc_f,
                 color="#ff8a65", linewidth=2.5, label="w_crit(lambda) @ T=300k")

    # lambda=1 vertical slice (T axis)
    wc_lam1 = wcrit_df[wcrit_df["lambda"] == 1.0].sort_values("T")
    if len(wc_lam1) >= 2:
        ax3.plot(
            [1.0] * len(wc_lam1),
            wc_lam1["T"] / 1e5,
            wc_lam1["w_crit"],
            color="#4fc3f7", linewidth=2.5, label="w_crit(T) @ lambda=1"
        )

    # horizontal w=1 ceiling plane (guide)
    lam_ceil = np.array([1, 7])
    T_ceil   = np.array([1.5, 6.0])
    Lc, Tc   = np.meshgrid(lam_ceil, T_ceil)
    ax3.plot_surface(Lc, Tc, np.ones_like(Lc) * 1.0,
                     alpha=0.08, color="white")

    # label each point
    for _, row in wcrit_df.iterrows():
        ax3.text(row["lambda"], row["T"] / 1e5 + 0.08, row["w_crit"] + 0.01,
                 f'{row["w_crit"]:.2f}',
                 color="white", fontsize=6.5, ha="center")

    ax3.legend(fontsize=7, framealpha=0.15, labelcolor="white",
               loc="upper left")

ax3.set_xlabel("lambda", color="white", labelpad=6, fontsize=9)
ax3.set_ylabel("T (x10^5 steps)", color="white", labelpad=6, fontsize=9)
ax3.set_zlabel("w_crit", color="white", labelpad=6, fontsize=9)
ax3.set_zlim(0.4, 1.0)
ax3.set_title("Collapse boundary  w_crit(lambda, T)", color="white", fontsize=10, pad=10)
ax3.tick_params(colors="white", labelsize=7)
ax3.xaxis.pane.fill = ax3.yaxis.pane.fill = ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor("#333333")
ax3.yaxis.pane.set_edgecolor("#333333")
ax3.zaxis.pane.set_edgecolor("#333333")
ax3.grid(True, color="#2a2a2a", linewidth=0.4)
ax3.view_init(elev=28, azim=-55)

# ---------------------------------------------------------------------------
# 5. Global styling and save
# ---------------------------------------------------------------------------

plt.suptitle(
    "Behavioural Phase Boundary in (w, lambda, T) Space\n"
    "Cooperative Supply-Chain MARL  |  CPT reward shaping",
    color="white", fontsize=12, y=1.01, fontweight="bold"
)

fig.patch.set_facecolor("#0d0d0d")
plt.tight_layout(pad=2.0)

out = os.path.join(OUT_DIR, "3d_w_lambda_T.png")
plt.savefig(out, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nSaved {out}")
plt.close()
