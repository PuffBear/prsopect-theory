"""
Exp MAPPO-MPE horizon (metastability generalization).

Tests whether the metastability result --- w_crit slides LEFT as the training
horizon T grows --- generalizes from the supply chain (Exp T) to MPE Cooperative
Navigation. Two-point comparison (short vs long), as the supply chain used.

Reuses the validated MAPPO-MPE machinery (exp_mappo_mpe): same env builder,
centralized critic, occupancy collapse label, herding scale. The ONLY new axis is
the training horizon. The standard MPE budget is 4M; we bracket it with 2M (short)
and 8M (long) at the curvature condition (alpha=beta=0.88, lambda=1).

GRID
----
  w        in {0.0, 0.1, ..., 1.0}        (11, the MPE w-grid)
  horizon  in {2_000_000, 8_000_000}      (2: short vs long)
  seeds    42-61                          (20)
  Total: 11 x 2 x 20 = 440 runs   ->  PBS array 0..439

NOTE: collapse is occupancy < theta; theta is recalibrated at analysis time from
this run's null endpoints (see analyze_mpe_decomposition.py). mean_occupancy is
logged raw so the label can be recomputed.

    python experiments/exp_mappo_mpe_horizon.py --wiring_check
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
from exp_mappo_mpe import train_mappo_model, _build_env


def _append_row(row, out_path, columns):
    """Local append writing THIS driver's columns (imported _append_row uses the
    base module's COLUMNS and silently drops 'horizon')."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=columns).to_csv(
            out_path, mode="a", header=write_header, index=False)
from exp_ppo_mpe import evaluate_mpe
from env.mpe_coop_nav import is_collapsed_mpe_occupancy, HERDING_SCALE

W_LIST = [round(0.1 * i, 1) for i in range(11)]   # 11 levels (MPE axis)
HORIZONS = [2_000_000, 8_000_000]                 # short vs long
SEEDS = list(range(42, 62))                       # 20
ALPHA, BETA, LAMBDA = 0.88, 0.88, 1.0             # curvature point
N_W, N_H, N_SEED = len(W_LIST), len(HORIZONS), len(SEEDS)
N_RUNS = N_W * N_H * N_SEED                        # 440

COLUMNS = ["run_idx", "w", "horizon", "alpha", "beta", "lambda_loss", "seed",
           "algo", "env", "mean_reward", "mean_occupancy", "collapsed",
           "final_value_loss", "final_explained_variance"]


def map_run_idx(run_idx):
    block = N_H * N_SEED
    w_idx = run_idx // block
    rem = run_idx % block
    h_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], HORIZONS[h_idx], SEEDS[seed_idx]


def run_one(run_idx, out_path, herding_scale=HERDING_SCALE):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, horizon, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, cb = train_mappo_model(w, ALPHA, BETA, LAMBDA, seed, horizon,
                                  herding_scale=herding_scale)
    eval_env = _build_env(w, LAMBDA, ALPHA, BETA, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe(model, eval_env, num_episodes=10,
                                         base_seed=seed * 10)
    collapsed = is_collapsed_mpe_occupancy(mean_occ)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "horizon": horizon, "alpha": ALPHA,
           "beta": BETA, "lambda_loss": LAMBDA, "seed": seed, "algo": "MAPPO",
           "env": "MPE_CoopNav", "mean_reward": mean_reward,
           "mean_occupancy": mean_occ, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path, COLUMNS)
    print(f"[MAPPO-MPE-horizon] run_idx={run_idx} w={w} T={horizon} seed={seed} "
          f"-> occ={mean_occ:.3f} collapsed={collapsed}", flush=True)
    return row


def wiring_check():
    """Build a model directly (no training) and assert the centralized critic."""
    from agents.mappo_policy import _CentralCriticExtractor, CentralizedCriticPolicy
    from exp_mappo_mpe import build_mappo_vec
    from exp_ppo_mpe import PPO_KWARGS
    from stable_baselines3 import PPO
    vec, _inner, local_dim = build_mappo_vec(0.5, ALPHA, BETA, LAMBDA, seed=42)
    m = PPO(CentralizedCriticPolicy, vec, seed=42,
            policy_kwargs=dict(local_dim=local_dim,
                               net_arch=dict(pi=[64, 64], vf=[128, 128])),
            **PPO_KWARGS)
    assert isinstance(m.policy.vf_features_extractor, _CentralCriticExtractor), \
        "vf extractor is NOT the centralized critic!"
    print(f"  centralized critic wired (local_dim={local_dim})")
    print(f"  WIRING CHECK PASS  (MPE horizon: {N_RUNS} runs, T in {HORIZONS})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--out", default="docs/expMAPPO_mpe_horizon_figures/mappo_mpe_horizon_raw_data.csv")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.out)


if __name__ == "__main__":
    main()
