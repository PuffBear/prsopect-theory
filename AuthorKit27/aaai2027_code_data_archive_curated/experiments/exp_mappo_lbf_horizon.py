"""
Exp MAPPO-LBF horizon (metastability generalization).

Tests whether the metastability result --- w_crit slides LEFT as the training
horizon T grows, so collapse at intermediate w is a metastable attractor rather
than an absorbing state --- generalizes from the supply chain (Exp T) to LBF.

Mirrors exp_T_horizon.py (supply chain) in design: fix the behavioral condition,
sweep w x seed, and run the SAME grid at two horizons. Metastability = w_crit at
the short horizon > w_crit at the long horizon (boundary slides left with T), the
two-point comparison the supply chain used originally (150k vs 300k).

Reuses the validated MAPPO-LBF machinery (exp_mappo_lbf): same env builder,
centralized critic, collapse label (team return < theta), 20 seeds. The ONLY new
axis is the training horizon.

GRID
----
  w        in {0.3, 0.5, 0.65, 0.75, 0.85, 0.95}   (6, the LBF w-grid)
  horizon  in {1_000_000, 4_000_000}               (2: short vs long)
  seeds    42-61                                    (20)
  Total: 6 x 2 x 20 = 240 runs   ->  PBS array 0..239

Condition fixed at the curvature point (alpha=beta=0.88, lambda=1), matching the
supply-chain Exp-T condition, so the slide is measured on the same behavioral
setting.

    python experiments/exp_mappo_lbf_horizon.py --wiring_check
"""
import os
import sys
import argparse
import warnings
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from filelock import FileLock

from exp_mappo_lbf import build_mappo_model, _build_mappo_env, wiring_check
from exp_ppo_lbf import evaluate_lbf, MetricsCallback
from env.lbf_wrapper import is_collapsed_lbf


def _append_row(row, out_path, columns):
    """Local append that writes THIS driver's columns (the imported
    exp_mappo_lbf._append_row uses that module's COLUMNS, which has no 'horizon'
    field and silently drops it -- the bug that lost the axis on the first run)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=columns).to_csv(
            out_path, mode="a", header=write_header, index=False)

W_LIST = [0.3, 0.5, 0.65, 0.75, 0.85, 0.95]
HORIZONS = [1_000_000, 4_000_000]           # short vs long (metastability slide)
SEEDS = list(range(42, 62))                 # 20
ALPHA, BETA, LAMBDA = 0.88, 0.88, 1.0       # curvature point (matches supply Exp-T)
N_W, N_H, N_SEED = len(W_LIST), len(HORIZONS), len(SEEDS)
N_RUNS = N_W * N_H * N_SEED                  # 240

COLUMNS = ["run_idx", "w", "horizon", "alpha", "beta", "lambda_loss", "seed",
           "algo", "env", "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


def map_run_idx(run_idx):
    """w (outer) x horizon (middle) x seed (inner)."""
    block = N_H * N_SEED
    w_idx = run_idx // block
    rem = run_idx % block
    h_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], HORIZONS[h_idx], SEEDS[seed_idx]


def run_one(run_idx, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, horizon, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, _inner = build_mappo_model(w, LAMBDA, ALPHA, BETA, seed)
    cb = MetricsCallback()
    _inner.reset(seed=int(seed))
    model.learn(total_timesteps=horizon, callback=cb)

    eval_env = _build_mappo_env(w, LAMBDA, ALPHA, BETA)
    mean_reward = evaluate_lbf(model, eval_env, num_episodes=20, base_seed=seed * 10)
    collapsed = is_collapsed_lbf(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "horizon": horizon, "alpha": ALPHA,
           "beta": BETA, "lambda_loss": LAMBDA, "seed": seed, "algo": "MAPPO",
           "env": "LBF_5x5_2p_2f", "mean_reward": mean_reward, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path, COLUMNS)
    print(f"[MAPPO-LBF-horizon] run_idx={run_idx} w={w} T={horizon} seed={seed} "
          f"-> team_return={mean_reward:.3f} collapsed={collapsed}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--out", default="docs/expMAPPO_lbf_horizon_figures/mappo_lbf_horizon_raw_data.csv")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); print(f"  (LBF horizon: {N_RUNS} runs, T in {HORIZONS})"); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.out)


if __name__ == "__main__":
    main()
