"""
LBF collapse-threshold calibration (v2, gated).

Trains the endpoints (w=0 own-share vs w=1 team return, NULL condition)
with the exact main-sweep config; logs mean team return per eval episode.
Grid: w in {0.0, 1.0} x seeds 42-51 -> 20 runs. PBS array 0..19.
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
from exp_ppo_lbf import (
    _build_env, evaluate_lbf, MetricsCallback, N_ENV_COPIES, PPO_KWARGS,
)

W_CAL  = [0.0, 1.0]
SEEDS  = list(range(42, 52))
N_RUNS = len(W_CAL) * len(SEEDS)
ALPHA, BETA, LAMBDA = 1.0, 1.0, 1.0   # null condition

COLUMNS = ["run_idx", "w", "seed", "mean_reward",
           "final_value_loss", "final_explained_variance"]


def map_run_idx(run_idx):
    return W_CAL[run_idx // len(SEEDS)], SEEDS[run_idx % len(SEEDS)]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    train_env = _build_env(w, LAMBDA, ALPHA, BETA)
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

    eval_env = _build_env(w, LAMBDA, ALPHA, BETA)
    mean_reward = evaluate_lbf(model, eval_env, num_episodes=10,
                               base_seed=seed * 10)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "seed": seed,
           "mean_reward": mean_reward,
           "final_value_loss": fvl, "final_explained_variance": fev}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)
    print(f"[LBF-calib] run_idx={run_idx} w={w} seed={seed} "
          f"-> team_return={mean_reward:.3f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx",   type=int, required=True, help="0..19")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--out", default="docs/exp_lbf/lbf_calibration_v2_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
