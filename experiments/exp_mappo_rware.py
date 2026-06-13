"""
Exp MAPPO-RWARE -- MAPPO on Robot Warehouse with CPT reward shaping.

Cross-algorithm replication: MAPPO on RWARE (Christianos et al. 2020).
The environment and wrapper stack are IDENTICAL to exp_ppo_rware.py so that
the only variable between the two experiments is the algorithm.

=== WRAPPER STACK (same as PPO-RWARE) ===

    rware gymnasium env
      -> RWAREParallelWrapper       (gym -> PettingZoo)
      -> RWARELocalRewardWrapper    (r_local=carry_bonus, w-blend)
      -> CPTRewardWrapper           (alpha/beta/lambda)
      -> SuperSuit vectorisation

=== SWEEP GRID (same as PPO-RWARE) ===

    w      in {0.10, 0.15, ..., 0.90}  (17 levels)
    lambda in {1, 2, 3}                (null CPT + two loss-aversion levels)
    seeds  42-61                       (20 seeds)
    Total: 17 x 3 x 20 = 1020 runs

PBS array: run_idx 0..1019

=== REFERENCES ===

    Christianos et al. (2020). Shared Experience Multi-Agent Reinforcement
    Learning. NeurIPS.  (RWARE environment)
    Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative
    Multi-Agent Games. NeurIPS.  (MAPPO)
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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env.rware_wrapper import make_rware_env_with_w, is_collapsed_rware
from agents.cpt_wrapper import CPTRewardWrapper
from exp_ppo_rware import MetricsCallback, evaluate_rware

# --- Sweep grid (must match exp_ppo_rware.py exactly) -----------------------
W_LIST      = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10..0.90
LAMBDA_LIST = [1, 2, 3]
SEEDS       = list(range(42, 62))
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED  # 1020

ALPHA, BETA = 0.88, 0.88
ENV_ID      = "rware-tiny-3ag-easy-v2"
MAX_CYCLES  = 500

COLUMNS = ["run_idx", "w", "lambda_loss", "seed", "algo", "env",
           "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


# --- MAPPO: centralised critic ----------------------------------------------

class MAPPOCriticExtractor(BaseFeaturesExtractor):
    """Wider centralised critic for MAPPO (Yu et al. 2022)."""
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 128), nn.ReLU(),
            nn.Linear(128, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# --- Index mapping -----------------------------------------------------------

def map_run_idx(run_idx):
    lam_seed_block = N_LAM * N_SEED
    w_idx    = run_idx // lam_seed_block
    rem      = run_idx % lam_seed_block
    lam_idx  = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], LAMBDA_LIST[lam_idx], SEEDS[seed_idx]


# --- Env builder (identical to exp_ppo_rware._build_env) --------------------

def _build_env(w, lam, alpha, beta):
    local_w = make_rware_env_with_w(w, env_id=ENV_ID, max_cycles=MAX_CYCLES)
    params  = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
               for a in local_w.possible_agents}
    return CPTRewardWrapper(local_w, params, reward_scale=1.0,
                            global_reward_weight=0.0)


# --- Main job ----------------------------------------------------------------

def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, lam, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    train_env = _build_env(w, lam, ALPHA, BETA)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    policy_kwargs = dict(
        features_extractor_class=MAPPOCriticExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[128, 128]),
    )
    model = PPO("MlpPolicy", vec, verbose=0,
                n_steps=256, batch_size=512, seed=seed,
                policy_kwargs=policy_kwargs)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env    = _build_env(w, lam, ALPHA, BETA)
    mean_reward = evaluate_rware(model, eval_env, num_episodes=10,
                                 base_seed=seed * 10)

    collapsed = is_collapsed_rware(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "seed": seed,
           "algo": "MAPPO", "env": "RWARE_tiny_3ag",
           "mean_reward": mean_reward, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"[MAPPO-RWARE] run_idx={run_idx} w={w} lam={lam} seed={seed} "
          f"-> mean_reward={mean_reward:.3f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx",   type=int, required=True, help="0..1019")
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--out", default="docs/exp_rware/mappo_rware_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
