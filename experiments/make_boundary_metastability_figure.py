"""Regenerate the main-paper boundary/horizon figure at one-column AAAI size."""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "legend.fontsize": 9.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from analyze_expN import _coerce, fit_logistic

OUT_PNG = os.path.join(ROOT, "AuthorKit27", "Figures", "boundary_metastability.png")
OUT_PDF = os.path.join(ROOT, "AuthorKit27", "Figures", "boundary_metastability.pdf")

HORIZON_POINTS = np.array([150_000, 300_000, 450_000, 900_000], dtype=float)
WCRIT = np.array([0.822, 0.666, 0.622, 0.573])
CI_LO = np.array([0.785, 0.635, 0.603, 0.547])
CI_HI = np.array([0.858, 0.694, 0.641, 0.598])


def load_fit(path):
    df = pd.read_csv(path)
    if "lambda" in df.columns:
        df = df[df["lambda"] == 1.0]
    if "lambda_loss" in df.columns:
        df = df[df["lambda_loss"] == 1.0]
    df = _coerce(df)
    model, _, _, _, _ = fit_logistic(df)
    return float(model.params["const"]), float(model.params["w"])


def logistic_curve(intercept, slope, x):
    return 1.0 / (1.0 + np.exp(-(intercept + slope * x)))


def main():
    b150 = load_fit(os.path.join(ROOT, "docs", "expA_figures", "expA_raw_data.csv"))
    b300 = load_fit(os.path.join(ROOT, "docs", "expA300k_figures", "expA300k_raw_data.csv"))

    x = np.linspace(0.30, 1.00, 400)
    red = "#c44e52"
    blue = "#4c72b0"
    gray = "#7a7a7a"

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.45))

    ax = axes[0]
    ax.plot(x, logistic_curve(*b150, x), color=blue, lw=1.7)
    ax.plot(x, logistic_curve(*b300, x), color=red, lw=1.7)
    ax.axhline(0.5, color="#b0b0b0", lw=0.8, ls="--", zorder=0)
    ax.axvline(WCRIT[0], color=blue, lw=0.9, ls=":")
    ax.axvline(WCRIT[1], color=red, lw=0.9, ls=":")
    ax.text(0.86, 0.82, "150k", color=blue, ha="left", va="center")
    ax.text(0.48, 0.22, "300k", color=red, ha="left", va="center")
    ax.text(WCRIT[0] + 0.01, 0.06, r"$0.822$", color=blue, ha="left", va="bottom")
    ax.text(WCRIT[1] - 0.01, 0.06, r"$0.666$", color=red, ha="right", va="bottom")
    ax.set_title("(a) Boundary")
    ax.set_xlabel(r"centralization $w$")
    ax.set_ylabel(r"$P(\mathrm{collapse})$")
    ax.set_xlim(0.30, 1.00)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_yticks([0, 0.5, 1.0])

    ax = axes[1]
    x_pos = np.log(HORIZON_POINTS)
    yerr = np.vstack([WCRIT - CI_LO, CI_HI - WCRIT])
    ax.errorbar(
        x_pos,
        WCRIT,
        yerr=yerr,
        fmt="o",
        color=blue,
        ecolor=blue,
        elinewidth=1.0,
        capsize=2.0,
        markersize=3.8,
        zorder=3,
    )
    x_grid = np.linspace(x_pos.min(), x_pos.max(), 200)
    ax.plot(x_grid, 2.44 - 0.138 * x_grid, color=red, lw=1.6)
    ax.text(12.78, 0.752, r"$2.44-0.138\ln T$", color=red, ha="left", va="center")
    ax.text(12.60, 0.612, "measured", color=blue, ha="left", va="center")
    ax.set_title("(b) Horizon Slide")
    ax.set_xlabel(r"training horizon $T$")
    ax.set_ylabel(r"$w_{\mathrm{crit}}$")
    ax.set_xlim(x_pos.min() - 0.08, x_pos.max() + 0.08)
    ax.set_ylim(0.50, 0.87)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["150k", "300k", "450k", "900k"])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])

    for ax in axes:
        ax.spines["top"].set_linewidth(0.8)
        ax.spines["right"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=2.8)

    fig.tight_layout(pad=0.20, h_pad=0.55)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
