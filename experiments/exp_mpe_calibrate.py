"""
Exp MPE - collapse-threshold calibration (theta).

The MPE collapse metric is the cumulative TRUE reward (raw global team score
from simple_spread at local_ratio=0.0) over an eval episode.  To label a run
"collapsed" we need a threshold theta placed between the two extremes:

  w = 0.0  -> fully local (herding dominant) -> agents cluster -> worst coverage
  w = 1.0  -> fully global                   -> agents spread  -> best coverage

Expected ranges with HERDING_SCALE=4.0 and local_ratio=0.0:
  w=0: episode true_reward ≈ -200...-250   (25 steps x 3 agents x -3.0/step)
  w=1: episode true_reward ≈ -20...-35     (25 steps x 3 agents x -0.3/step)
  midpoint theta ≈ -110...-140

This runner trains those extreme conditions with the SAME pipeline as the main
sweep (PPO, alpha=beta=0.88, lambda=1, 600k steps, herding_scale=4.0).
The collapse metric (true_reward) is independent of alpha/beta/lambda, so the
threshold transfers to all cells of the main sweep.

Grid: w in {0.0, 1.0} x seeds 42-61 (20 seeds each) -> 40 runs.
PBS array: run_idx 0..39.

WORKFLOW:
  1. qsub docs/slurm/run_mpe_calibrate.pbs   (or run locally: see run_mpe_pilot_local.py)
  2. python experiments/analyze_mpe_calibrate.py
     -> writes docs/exp_mpe/mpe_collapse_threshold.json
  3. qsub docs/slurm/run_ppo_mpe.pbs          (main 1020-run sweep)
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
from exp_ppo_mpe import (
    _build_env, evaluate_mpe, MetricsCallback, HERDING_SCALE,
    N_ENV_COPIES, PPO_KWARGS,
)

# v2: calibrate under the NULL condition (no behavioral bias). The occupancy
# metric is independent of alpha/beta/lambda, so the threshold transfers.
ALPHA, BETA = 1.0, 1.0

W_CAL  = [0.0, 1.0]          # worst (herding) vs best (global) coordination
SEEDS  = list(range(42, 52)) # 10 seeds each extreme (v2)
N_RUNS = len(W_CAL) * len(SEEDS)   # 40
LAMBDA = 1                   # null CPT; threshold is independent of lambda

COLUMNS = ["run_idx", "w", "seed", "mean_reward", "mean_occupancy",
           "final_value_loss", "final_explained_variance"]


def map_run_idx(run_idx):
    """run_idx -> (w, seed).  First 20 = w=0, last 20 = w=1."""
    return W_CAL[run_idx // len(SEEDS)], SEEDS[run_idx % len(SEEDS)]


def run_one(run_idx, timesteps, out_path, herding_scale=HERDING_SCALE):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    train_env = _build_env(w, LAMBDA, ALPHA, BETA, herding_scale=herding_scale)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                base_class='stable_baselines3')
    # --- reproducible env-spawn seeding (v3 fix) --------------------------
    # np.random.seed() does NOT reach the gymnasium env's internal np_random,
    # so v2 runs were not reproducible across seeds. ConcatVecEnv.reset(seed)
    # assigns seed+i per copy (decorrelated); SB3's later reset(None) keeps
    # that np_random, making the whole run reproducible.
    _inner_vec = getattr(vec, "venv", vec)
    vec.seed = lambda s, _v=_inner_vec: _v.reset(seed=int(s))

    model = PPO("MlpPolicy", vec, verbose=0, seed=seed, **PPO_KWARGS)
    cb = MetricsCallback()
    _inner_vec.reset(seed=int(seed))
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, LAMBDA, ALPHA, BETA, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe(model, eval_env, num_episodes=10,
                                         base_seed=seed * 10)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "seed": seed,
           "mean_reward": mean_reward, "mean_occupancy": mean_occ,
           "final_value_loss": fvl,
           "final_explained_variance": fev}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)
    print(f"[MPE-calib] run_idx={run_idx} w={w} seed={seed} "
          f"-> r_global={mean_reward:.1f} occ={mean_occ:.2f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx",      type=int,   required=True, help="0..39")
    ap.add_argument("--timesteps",    type=int,   default=2_000_000)
    ap.add_argument("--herding_scale",type=float, default=HERDING_SCALE)
    ap.add_argument("--out", default="docs/exp_mpe/mpe_calibration_v2_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out,
            herding_scale=args.herding_scale)


if __name__ == "__main__":
    main()
