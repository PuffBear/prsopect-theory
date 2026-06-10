"""
Exp MPE - collapse-threshold calibration (theta).

The MPE collapse metric is the cumulative TRUE (raw simple_spread) reward over an
eval episode. To label a run "collapsed" we need a threshold theta. Per the AAAI
spec, theta is set from the distribution of trained agents at the two extremes:
  w = 1.0  -> fully global reward  -> best coordination (high true reward)
  w = 0.0  -> fully local reward   -> worst coordination (low true reward)
theta is then placed between the two regimes (see analyze_mpe_calibrate.py).

This runner trains those extreme conditions with the SAME pipeline as the main
sweep (exp_ppo_mpe: PPO, alpha=beta=0.88, lambda=1, 300k) so the calibration
distribution matches the sweep. The collapse metric (true_reward) is independent
of alpha/beta/lambda, so the threshold transfers to the whole sweep.

Grid: w in {0.0, 1.0} x seeds 42-61 (20)  ->  40 runs.
PBS array: one task per run_idx 0..39.  Run this BEFORE (or alongside) the sweep,
then run analyze_mpe_calibrate.py to write the threshold file.
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
from exp_ppo_mpe import _build_env, evaluate_mpe, MetricsCallback, ALPHA, BETA

W_CAL = [0.0, 1.0]            # worst (local) vs best (global) coordination
SEEDS = list(range(42, 62))  # 20
N_RUNS = len(W_CAL) * len(SEEDS)  # 40
LAMBDA = 1                   # null CPT (same as Exp 2a)

COLUMNS = ["run_idx", "w", "seed", "mean_reward", "final_value_loss", "final_explained_variance"]


def map_run_idx(run_idx):
    return W_CAL[run_idx // len(SEEDS)], SEEDS[run_idx % len(SEEDS)]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    train_env = _build_env(w, LAMBDA, ALPHA, BETA)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, LAMBDA, ALPHA, BETA)
    mean_reward = evaluate_mpe(model, eval_env, num_episodes=10, base_seed=seed * 10)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "seed": seed, "mean_reward": mean_reward,
           "final_value_loss": fvl, "final_explained_variance": fev}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(out_path, mode="a", header=write_header, index=False)
    print(f"[MPE-calib] run_idx={run_idx} w={w} seed={seed} -> mean_reward={mean_reward:.1f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help="0..39")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="docs/exp_mpe/mpe_calibration_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
