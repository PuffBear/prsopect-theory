"""
Cross-environment, cross-algorithm grand comparison.

Produces the master 2x2 factorial figure: {PPO, MAPPO} x {or-gym, MPE}
showing w_crit(lambda) across all four cells.

Reads all four raw data CSVs and produces:
  - Grand 2x2 panel figure (one subplot per cell)
  - Overlay figure: all four w_crit(lambda) curves on one axes
  - Summary table: w_crit at each lambda for all conditions
  - grand_comparison_results.md
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyze_expN import fit_logistic, bootstrap_wcrit, _coerce, per_w_table, curve

OUTDIR = "docs/grand_comparison_figures"
LAMBDAS = [1, 2, 3, 4, 5, 6, 7]

# Data sources for each cell of the 2x2 factorial
CELLS = {
    ("PPO", "or-gym"): {
        "csvs": [
            "docs/expB_reduced_figures/expB_reduced_raw_data.csv",
            "docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv",
        ],
        "color": "tab:blue",
        "marker": "s",
    },
    ("MAPPO", "or-gym"): {
        "csvs": ["docs/expMAPPO_orgym_figures/mappo_orgym_raw_data.csv"],
        "color": "tab:red",
        "marker": "o",
    },
    ("PPO", "MPE"): {
        "csvs": ["docs/expPPO_mpe_figures/ppo_mpe_raw_data.csv"],
        "color": "tab:green",
        "marker": "^",
    },
    ("MAPPO", "MPE"): {
        "csvs": ["docs/expMAPPO_mpe_figures/mappo_mpe_raw_data.csv"],
        "color": "tab:purple",
        "marker": "D",
    },
}


def load_and_merge(csvs):
    dfs = []
    for p in csvs:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["lambda_loss"] = df["lambda_loss"].astype(float).round().astype(int)
    return _coerce(df)


def analyze_lambda(df, lam):
    sub = df[df["lambda_loss"] == lam]
    if len(sub) < 5:
        return None
    try:
        m, slope, pval, r2, w_crit = fit_logistic(sub)
        _, wc_lo, wc_hi = bootstrap_wcrit(sub)
    except Exception:
        return None
    return {"lambda": lam, "n": len(sub), "model": m, "slope": slope,
            "pval": pval, "r2": r2, "w_crit": w_crit, "ci_lo": wc_lo, "ci_hi": wc_hi}


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Analyze each cell
    all_res = {}
    for (algo, env), cfg in CELLS.items():
        df = load_and_merge(cfg["csvs"])
        if df is None:
            print(f"No data for {algo} x {env}")
            continue
        res = {}
        for lam in LAMBDAS:
            r = analyze_lambda(df, lam)
            if r is not None:
                res[lam] = r
        if res:
            all_res[(algo, env)] = res
            print(f"{algo} x {env}: {len(df)} runs, {len(res)} lambda-levels fitted")

    if not all_res:
        print("No data available for any cell.")
        sys.exit(0)

    sns.set_style("whitegrid")

    # --- Figure 1: Overlay w_crit(lambda) ------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for (algo, env), res in all_res.items():
        cfg = CELLS[(algo, env)]
        valid_lams = sorted(res.keys())
        xs = valid_lams
        ys = [res[l]["w_crit"] for l in valid_lams]
        lo = [res[l]["w_crit"] - res[l]["ci_lo"] for l in valid_lams]
        hi = [res[l]["ci_hi"] - res[l]["w_crit"] for l in valid_lams]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=f"{cfg['marker']}-",
                    color=cfg["color"], capsize=5, lw=2, markersize=8,
                    label=f"{algo} x {env}")
    ax.set_xlabel("loss aversion lambda", fontsize=12)
    ax.set_ylabel("w_crit (P=0.5)", fontsize=12)
    ax.set_xticks(LAMBDAS)
    ax.set_title("Grand Comparison: w_crit vs lambda across algorithms and environments",
                 fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    f1 = os.path.join(OUTDIR, "grand_wcrit_overlay.png")
    plt.savefig(f1, dpi=150)
    print(f"\nSaved {f1}")

    # --- Figure 2: 2x2 panel -------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    panel_map = {
        ("PPO", "or-gym"): axes[0, 0],
        ("MAPPO", "or-gym"): axes[0, 1],
        ("PPO", "MPE"): axes[1, 0],
        ("MAPPO", "MPE"): axes[1, 1],
    }
    wg = np.linspace(0.0, 1.0, 200)
    LCOLS = {1: "tab:blue", 2: "tab:cyan", 3: "tab:orange", 4: "tab:olive",
             5: "tab:green", 6: "tab:purple", 7: "tab:red"}

    for (algo, env), ax in panel_map.items():
        ax.set_title(f"{algo} x {env}", fontsize=12)
        if (algo, env) not in all_res:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray")
            continue
        res = all_res[(algo, env)]
        for lam in LAMBDAS:
            if lam not in res:
                continue
            r = res[lam]; c = LCOLS[lam]
            ax.plot(wg, curve(r["model"], wg), color=c, lw=2, label=f"lambda={lam}")
            ax.axvline(r["w_crit"], color=c, ls="--", lw=1, alpha=0.6)
        ax.set_xlim(0.0, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, ncol=2)

    for ax in axes[1, :]:
        ax.set_xlabel("w (reward centralization)")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(collapse)")
    fig.suptitle("Collapse Transition: 2x2 Factorial (Algorithm x Environment)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    f2 = os.path.join(OUTDIR, "grand_2x2_panel.png")
    plt.savefig(f2, dpi=150, bbox_inches="tight")
    print(f"Saved {f2}")

    # --- Results markdown -----------------------------------------------
    md = ["# Grand Comparison: {PPO, MAPPO} x {or-gym, MPE}\n",
          "## w_crit by lambda across all conditions\n"]
    header = "| lambda |"
    sep = "|---|"
    for (algo, env) in CELLS:
        header += f" {algo}x{env} |"
        sep += "---|"
    md += [header, sep]
    for lam in LAMBDAS:
        row_str = f"| {lam} |"
        for (algo, env) in CELLS:
            if (algo, env) in all_res and lam in all_res[(algo, env)]:
                r = all_res[(algo, env)][lam]
                row_str += f" {r['w_crit']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |"
            else:
                row_str += " - |"
        md.append(row_str)
    md += ["\nFigures: [grand_wcrit_overlay.png](grand_wcrit_overlay.png), "
           "[grand_2x2_panel.png](grand_2x2_panel.png)"]
    with open(os.path.join(OUTDIR, "grand_comparison_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {os.path.join(OUTDIR, 'grand_comparison_results.md')}")


if __name__ == "__main__":
    main()
