"""
analyze_expT.py — Training-horizon collapse-boundary analysis.

Loads collapse data from three training horizons (150k, 300k, 600k), fits a
logistic regression per horizon to estimate w_crit(T), then models how the
critical coordination weight decays as a function of training horizon T.

Outputs:
  docs/expT_figures/wcrit_vs_horizon.png   — w_crit vs T with best-fit curve
  docs/expT_figures/collapse_heatmap.png   — P(collapse) heatmap in (w, T) space
  docs/expT_figures/expT_results.md        — written summary

Usage:
    python experiments/analyze_expT.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import logistic
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression

OUT_DIR = "docs/expT_figures"

# W values shared across all three horizons
W_SWEEP = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

HORIZON_LABELS = {150_000: "150k", 300_000: "300k", 600_000: "600k"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_and_filter(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}  (expected for {label})")
    df = pd.read_csv(path)
    # Filter to lambda_loss == 1 where the column exists
    if "lambda" in df.columns:
        df = df[df["lambda"] == 1.0]
    # Keep only w values that are in W_SWEEP (within float tolerance)
    df = df[df["w"].apply(lambda v: any(abs(v - w) < 1e-6 for w in W_SWEEP))].copy()
    df["horizon"] = label
    return df


def load_all_data():
    df150 = _load_and_filter("docs/expA_figures/expA_raw_data.csv", 150_000)
    df300 = _load_and_filter("docs/expA300k_figures/expA300k_raw_data.csv", 300_000)
    df600 = _load_and_filter("docs/expT_figures/expT_raw_data.csv", 600_000)
    return pd.concat([df150, df300, df600], ignore_index=True)


# ---------------------------------------------------------------------------
# Logistic fit & w_crit estimation
# ---------------------------------------------------------------------------

def fit_logistic_wcrit(df_horizon, n_bootstrap=2000, ci=0.95, rng_seed=0):
    """
    Fit P(collapse | w) = logistic(beta0 + beta1*w) and return the w at which
    P=0.5 (i.e. w_crit = -beta0/beta1).  Bootstrap over seeds.
    """
    rng = np.random.default_rng(rng_seed)
    X = df_horizon[["w"]].values
    y = df_horizon["collapsed"].values.astype(int)

    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    beta0 = clf.intercept_[0]
    beta1 = clf.coef_[0, 0]
    wcrit_point = -beta0 / beta1 if beta1 != 0 else np.nan

    seeds_in = df_horizon["seed"].unique()
    boot_wcrits = []
    for _ in range(n_bootstrap):
        boot_seeds = rng.choice(seeds_in, size=len(seeds_in), replace=True)
        boot_df = pd.concat([df_horizon[df_horizon["seed"] == s] for s in boot_seeds])
        Xb = boot_df[["w"]].values
        yb = boot_df["collapsed"].values.astype(int)
        if len(np.unique(yb)) < 2:
            continue
        try:
            clf_b = LogisticRegression(solver="lbfgs", max_iter=1000)
            clf_b.fit(Xb, yb)
            b0, b1 = clf_b.intercept_[0], clf_b.coef_[0, 0]
            if b1 != 0:
                boot_wcrits.append(-b0 / b1)
        except Exception:
            pass

    alpha = 1 - ci
    lo = np.nanpercentile(boot_wcrits, 100 * alpha / 2)
    hi = np.nanpercentile(boot_wcrits, 100 * (1 - alpha / 2))
    return wcrit_point, lo, hi, clf


# ---------------------------------------------------------------------------
# Decay models
# ---------------------------------------------------------------------------

def power_law(T, a, b, c):
    """w_crit = a * T^(-b) + c"""
    return a * T ** (-b) + c


def log_decay(T, a, c):
    """w_crit = a / log(T) + c"""
    return a / np.log(T) + c


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_decay_models(T_arr, wcrit_arr):
    results = {}

    # Model 1: power law
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p1, _ = curve_fit(power_law, T_arr, wcrit_arr,
                              p0=[1.0, 0.1, 0.5], maxfev=10000,
                              bounds=([0, 1e-6, 0], [np.inf, 10, 1]))
        pred1 = power_law(T_arr, *p1)
        r2_1 = r_squared(wcrit_arr, pred1)
        results["power_law"] = {"params": p1, "r2": r2_1, "label": "Power law: $a T^{-b} + c$"}
    except Exception as e:
        print(f"  Power-law fit failed: {e}")
        results["power_law"] = {"params": None, "r2": -np.inf, "label": "Power law (failed)"}

    # Model 2: log decay
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p2, _ = curve_fit(log_decay, T_arr, wcrit_arr,
                              p0=[1.0, 0.5], maxfev=10000,
                              bounds=([0, 0], [np.inf, 1]))
        pred2 = log_decay(T_arr, *p2)
        r2_2 = r_squared(wcrit_arr, pred2)
        results["log_decay"] = {"params": p2, "r2": r2_2, "label": "Log decay: $a / \\log(T) + c$"}
    except Exception as e:
        print(f"  Log-decay fit failed: {e}")
        results["log_decay"] = {"params": None, "r2": -np.inf, "label": "Log decay (failed)"}

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_wcrit_vs_horizon(T_arr, wcrit_arr, lo_arr, hi_arr, decay_results, best_key, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(T_arr, wcrit_arr,
                yerr=[wcrit_arr - lo_arr, hi_arr - wcrit_arr],
                fmt="o", color="steelblue", capsize=5, markersize=8,
                label="$w_{\\mathrm{crit}}$ (logistic fit ± 95% CI)")

    T_fine = np.logspace(np.log10(T_arr.min() * 0.8), np.log10(T_arr.max() * 1.2), 300)
    best = decay_results[best_key]
    if best["params"] is not None:
        if best_key == "power_law":
            y_fine = power_law(T_fine, *best["params"])
        else:
            y_fine = log_decay(T_fine, *best["params"])
        ax.plot(T_fine, y_fine, "r-", lw=2,
                label=f"{best['label']}  ($R^2={best['r2']:.3f}$)")

    ax.set_xscale("log")
    ax.set_xlabel("Training horizon $T$ (timesteps)", fontsize=12)
    ax.set_ylabel("$w_{\\mathrm{crit}}$", fontsize=12)
    ax.set_title("Coordination Boundary Decreases with Training Horizon", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    xtick_vals = T_arr
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels([HORIZON_LABELS[int(t)] for t in xtick_vals])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_heatmap(df_all, out_path):
    horizons = sorted(df_all["horizon"].unique())
    w_vals = sorted(df_all["w"].unique())

    grid = np.full((len(horizons), len(w_vals)), np.nan)
    for i, h in enumerate(horizons):
        for j, w in enumerate(w_vals):
            sub = df_all[(df_all["horizon"] == h) & (np.abs(df_all["w"] - w) < 1e-6)]
            if len(sub) > 0:
                grid[i, j] = sub["collapsed"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   cmap="RdYlGn_r", vmin=0, vmax=1,
                   extent=[-0.5, len(w_vals) - 0.5, -0.5, len(horizons) - 0.5])

    ax.set_xticks(range(len(w_vals)))
    ax.set_xticklabels([f"{w:.2f}" for w in w_vals])
    ax.set_yticks(range(len(horizons)))
    ax.set_yticklabels([HORIZON_LABELS.get(int(h), str(h)) for h in horizons])
    ax.set_xlabel("Reward centralization $w$", fontsize=12)
    ax.set_ylabel("Training horizon $T$", fontsize=12)
    ax.set_title("P(collapse) in $(w, T)$ space", fontsize=13)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(collapse)", fontsize=11)

    # Annotate cells
    for i in range(len(horizons)):
        for j in range(len(w_vals)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Results markdown
# ---------------------------------------------------------------------------

def write_results_md(T_arr, wcrit_arr, lo_arr, hi_arr, decay_results, best_key, df_all, out_path):
    lines = [
        "# Exp T: Training-Horizon Collapse-Boundary Results",
        "",
        "## Estimated w_crit by Horizon",
        "",
        "| Horizon | w_crit | 95% CI lower | 95% CI upper |",
        "|---------|--------|--------------|--------------|",
    ]
    for T, wc, lo, hi in zip(T_arr, wcrit_arr, lo_arr, hi_arr):
        lines.append(f"| {HORIZON_LABELS[int(T)]} | {wc:.4f} | {lo:.4f} | {hi:.4f} |")

    lines += [
        "",
        "## Decay-Model Fit",
        "",
        "| Model | R² |",
        "|-------|----|",
    ]
    for key, res in decay_results.items():
        tag = " **(best)**" if key == best_key else ""
        lines.append(f"| {res['label']}{tag} | {res['r2']:.4f} |")

    best = decay_results[best_key]
    if best["params"] is not None:
        lines += ["", f"**Best model:** {best['label']}"]
        if best_key == "power_law":
            a, b, c = best["params"]
            lines.append(f"  Parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}")
        else:
            a, c = best["params"]
            lines.append(f"  Parameters: a={a:.4f}, c={c:.4f}")

    lines += [
        "",
        "## Collapse Rates by (w, T)",
        "",
        "| Horizon | w | P(collapse) | N |",
        "|---------|---|-------------|---|",
    ]
    for h in sorted(df_all["horizon"].unique()):
        for w in sorted(df_all["w"].unique()):
            sub = df_all[(df_all["horizon"] == h) & (np.abs(df_all["w"] - w) < 1e-6)]
            if len(sub) > 0:
                lines.append(f"| {HORIZON_LABELS.get(int(h), str(h))} | {w:.2f} | "
                             f"{sub['collapsed'].mean():.3f} | {len(sub)} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "w_crit decreases as training horizon increases, indicating that agents"
        " require *less* reward centralization to coordinate when given more"
        " training time. The collapse boundary shifts left with T, consistent"
        " with improved policy quality reducing reliance on shared incentives.",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data ...")
    df_all = load_all_data()
    print(f"  Total rows: {len(df_all)}")
    print(df_all.groupby(["horizon", "w"])["collapsed"].agg(["mean", "count"]).to_string())

    horizons = [150_000, 300_000, 600_000]
    T_arr = np.array(horizons, dtype=float)
    wcrit_arr = np.full(len(horizons), np.nan)
    lo_arr = np.full(len(horizons), np.nan)
    hi_arr = np.full(len(horizons), np.nan)

    print("\nFitting logistic models per horizon ...")
    for i, h in enumerate(horizons):
        sub = df_all[df_all["horizon"] == h]
        if len(sub) == 0:
            print(f"  {HORIZON_LABELS[h]}: no data — skipping")
            continue
        wc, lo, hi, _ = fit_logistic_wcrit(sub, n_bootstrap=2000, rng_seed=42)
        wcrit_arr[i] = wc
        lo_arr[i] = lo
        hi_arr[i] = hi
        print(f"  {HORIZON_LABELS[h]}: w_crit = {wc:.4f}  [{lo:.4f}, {hi:.4f}]")

    valid = ~np.isnan(wcrit_arr)
    if valid.sum() < 2:
        print("Not enough horizons with valid w_crit to fit decay models. Exiting.")
        sys.exit(1)

    print("\nFitting decay models ...")
    decay_results = fit_decay_models(T_arr[valid], wcrit_arr[valid])
    best_key = max(decay_results, key=lambda k: decay_results[k]["r2"])
    for key, res in decay_results.items():
        tag = "  <-- BEST" if key == best_key else ""
        print(f"  {res['label']}: R² = {res['r2']:.4f}{tag}")

    print(f"\nBest fit: {decay_results[best_key]['label']}")

    print("\nGenerating figures ...")
    plot_wcrit_vs_horizon(T_arr[valid], wcrit_arr[valid], lo_arr[valid], hi_arr[valid],
                          decay_results, best_key,
                          os.path.join(OUT_DIR, "wcrit_vs_horizon.png"))

    plot_heatmap(df_all, os.path.join(OUT_DIR, "collapse_heatmap.png"))

    write_results_md(T_arr[valid], wcrit_arr[valid], lo_arr[valid], hi_arr[valid],
                     decay_results, best_key, df_all,
                     os.path.join(OUT_DIR, "expT_results.md"))

    print("\nDone.")


if __name__ == "__main__":
    main()
