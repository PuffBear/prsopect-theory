"""
Exp B-Reduced — reduced w×λ collapse surface with standard CPT curvature.

5 w-levels × 3 λ-levels × 20 seeds = 300 runs. Fixed CPT curvature α=β=0.88
(standard Tversky-Kahneman); λ ∈ {1,3,7} adds loss aversion on top. Same [0,500]
base-stock substrate, eval, and collapse label as expA_interior_w.py / exp_null_model.py
so results are directly comparable to Exp N (the α=β=1 null).

SLURM array job: one task per run_idx 0..299, each appends one row under a filelock.
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from filelock import FileLock

import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Reuse the exact eval / callback / collapse label / substrate bounds from the reference.
from expA_interior_w import (evaluate_and_extract_metrics, is_collapsed, MetricsCallback,
                             LOW_BOUND, HIGH_BOUND)
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

# Frozen grid (spec)
W_LIST = [0.5, 0.65, 0.75, 0.85, 0.95]
LAMBDA_LIST = [1, 3, 7]
SEEDS = list(range(42, 62))  # 20
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED  # 300

# Fixed CPT curvature (standard Tversky-Kahneman)
ALPHA, BETA = 0.88, 0.88

# Collapse label (spec): (mean_S < 10) AND (profit <= -127.1)  [== is_collapsed]
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

COLUMNS = ["run_idx", "w", "lambda_loss", "seed", "mean_S", "profit", "lost_sales",
           "collapsed", "final_value_loss", "final_explained_variance"]


def _build_env(w, lam, alpha, beta, seed):
    """Same as expA._build_env but with configurable PT curvature (alpha,beta)."""
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def map_run_idx(run_idx):
    lam_idx = run_idx // (N_W * N_SEED)
    remainder = run_idx % (N_W * N_SEED)
    w_idx = remainder // N_SEED
    seed_idx = remainder % N_SEED
    return W_LIST[w_idx], LAMBDA_LIST[lam_idx], SEEDS[seed_idx]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, lam, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    train_env = _build_env(w, lam, ALPHA, BETA, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, ALPHA, BETA, seed)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)

    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "seed": seed,
           "mean_S": mean_S, "profit": profit, "lost_sales": lost_sales,
           "collapsed": collapsed, "final_value_loss": fvl,
           "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} w={w} λ={lam} seed={seed} -> mean_S={mean_S:.1f} "
          f"profit={profit:.1f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help="0..299")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="docs/expB_reduced_figures/expB_reduced_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
