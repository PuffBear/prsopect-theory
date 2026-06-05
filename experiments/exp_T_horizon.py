"""
Exp T — Training-horizon sweep for collapse boundary (experiment_spec_v1 sec.T).

Sweeps reward centralization w across [0.5, 0.8] at 600k training steps with
CPT shaping (lambda=1.0, alpha=0.88, beta=0.88). Designed as a SLURM array job:
one task per run_idx 0..139. Each task trains one PPO agent and appends a single
row to the shared CSV under a filelock.

Companion data at 150k steps: docs/expA_figures/expA_raw_data.csv (filtered to
lambda==1, w in [0.5,0.8]).
Companion data at 300k steps: docs/expA300k_figures/expA300k_raw_data.csv.

CLI:
    python experiments/exp_T_horizon.py --run_idx 0
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

from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from expA_interior_w import evaluate_and_extract_metrics, MetricsCallback

# --- Frozen grid (spec) ---
W_LIST = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
SEEDS = list(range(42, 62))       # 20 seeds
N_RUNS = len(W_LIST) * len(SEEDS) # 7 * 20 = 140

TOTAL_TIMESTEPS = 600_000

# CPT parameters for this experiment
LAMBDA_LOSS = 1.0
ALPHA = 0.88
BETA = 0.88

# Substrate range — must match expA for a consistent collapse label
LOW_BOUND, HIGH_BOUND = 0.0, 500.0

# Collapse definition: (mean_S < 10) AND (profit <= -127.1)
S_FLOOR = 10.0
PROFIT_CEIL = -127.1

OUT_PATH = "docs/expT_figures/expT_raw_data.csv"
COLUMNS = ["run_idx", "w", "seed", "mean_S", "profit", "lost_sales", "collapsed"]


def _build_env(w, seed):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": LAMBDA_LOSS, "alpha": ALPHA, "beta": BETA}
              for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")

    w = W_LIST[run_idx // len(SEEDS)]
    seed = SEEDS[run_idx % len(SEEDS)]
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    print(f"[run_idx={run_idx}] w={w:.2f} seed={seed} timesteps={timesteps:,}", flush=True)

    train_env = _build_env(w, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, seed)
    mean_S, _order, _inv, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)

    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))

    row = {
        "run_idx": run_idx,
        "w": w,
        "seed": seed,
        "mean_S": mean_S,
        "profit": profit,
        "lost_sales": lost_sales,
        "collapsed": int(collapsed),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lock = FileLock(out_path + ".lock")
    with lock:
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"[run_idx={run_idx}] done  mean_S={mean_S:.1f} profit={profit:.1f} "
          f"lost_sales={lost_sales:.3f} collapsed={collapsed}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description="Exp T: 600k horizon sweep")
    ap.add_argument("--run_idx", type=int, required=True,
                    help="SLURM array index 0..139")
    ap.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
