"""
MAPPO-MPE full loss-aversion sweep (fill-in) -- mirrors the supply-chain
ppo_lambda_sweep so MPE gets a complete w_crit(lambda) curve.

We already have lambda=1 (curvature condition) and lambda=3 (lambda3 condition)
at alpha=beta=0.88 from the decomposition sweep (exp_mappo_mpe.py). This script
runs the MISSING lambda values {2,4,5,6,7}, also at alpha=beta=0.88, on the same
w-grid / seeds, under the same centralized-critic MAPPO + 50-step horizon config
(shared train_mappo_model). Combine with the decomposition CSV in analysis to
build the 7-point curve.

Grid: lambda in {2,4,5,6,7} x W_LIST(11) x seeds 42-61 -> 1100 runs.
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from filelock import FileLock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from exp_mappo_mpe import train_mappo_model, _build_env
from exp_ppo_mpe import evaluate_mpe, W_LIST, SEEDS, HERDING_SCALE
from env.mpe_coop_nav import is_collapsed_mpe_occupancy

LAMBDA_FILL = [2.0, 4.0, 5.0, 6.0, 7.0]   # 1 and 3 already done in the decomposition sweep
ALPHA, BETA = 0.88, 0.88                  # curvature held fixed; vary loss aversion
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_FILL), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED             # 11 * 5 * 20 = 1100  (20 seeds; AAAI rev.)

COLUMNS = ["run_idx", "w", "alpha", "beta", "lambda_loss", "seed",
           "algo", "env", "mean_reward", "mean_occupancy", "collapsed",
           "final_value_loss", "final_explained_variance"]


def map_run_idx(run_idx):
    """w (outer) x lambda (middle) x seed (inner)."""
    block = N_LAM * N_SEED
    w_idx = run_idx // block
    rem = run_idx % block
    lam_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], LAMBDA_FILL[lam_idx], SEEDS[seed_idx]


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def run_one(run_idx, timesteps, out_path, herding_scale=HERDING_SCALE):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, lam, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, cb = train_mappo_model(w, ALPHA, BETA, lam, seed, timesteps,
                                  herding_scale=herding_scale)
    eval_env = _build_env(w, lam, ALPHA, BETA, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe(model, eval_env, num_episodes=10,
                                         base_seed=seed * 10)
    collapsed = is_collapsed_mpe_occupancy(mean_occ)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "alpha": ALPHA, "beta": BETA,
           "lambda_loss": lam, "seed": seed, "algo": "MAPPO", "env": "MPE_CoopNav",
           "mean_reward": mean_reward, "mean_occupancy": mean_occ,
           "collapsed": collapsed, "final_value_loss": fvl,
           "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"[MAPPO-MPE-lam] run_idx={run_idx} w={w} lam={lam} seed={seed} "
          f"-> r_global={mean_reward:.1f} occ={mean_occ:.3f} collapsed={collapsed}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx",   type=int, required=True, help=f"0..{N_RUNS-1}")
    ap.add_argument("--timesteps", type=int, default=4_000_000)
    ap.add_argument("--out", default="docs/expMAPPO_mpe_figures/mappo_mpe_lambda_sweep_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
