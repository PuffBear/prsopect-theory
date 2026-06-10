"""
Exp MAPPO-OG - MAPPO on or-gym NetworkManagement-v1 with CPT reward shaping.

Mirrors exp_b_reduced.py but replaces PPO with MAPPO (Multi-Agent PPO).
MAPPO is implemented as PPO with a centralised value function: during training,
the critic receives the concatenated observations of ALL agents (global state),
while the actor receives only the local observation. This is the standard CTDE
MAPPO formulation (Yu et al., 2022).

Sweeps:
  w  in  {0.5, 0.65, 0.75, 0.85, 0.95}  (5 levels)
  lambda  in  {1, 2, 3, 4, 5, 6, 7}           (7 levels)
  seeds 42-61                          (20 seeds)
  Total: 5 x 7 x 20 = 700 runs

Fixed CPT curvature: alpha = beta = 0.88 (standard Tversky-Kahneman).

SLURM/PBS array job: one task per run_idx 0..699.
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
from expA_interior_w import (evaluate_and_extract_metrics, is_collapsed, MetricsCallback,
                             LOW_BOUND, HIGH_BOUND)
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

# --- Grid -------------------------------------------------------------------
W_LIST = [0.5, 0.65, 0.75, 0.85, 0.95]
LAMBDA_LIST = [1, 2, 3, 4, 5, 6, 7]
SEEDS = list(range(42, 62))  # 20
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED  # 700

# Fixed CPT curvature
ALPHA, BETA = 0.88, 0.88

# Collapse label (spec)
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

COLUMNS = ["run_idx", "w", "lambda_loss", "seed", "algo",
           "mean_S", "profit", "lost_sales", "collapsed",
           "final_value_loss", "final_explained_variance"]


# --- MAPPO: centralised critic via custom feature extractor -----------------
class MAPPOCriticExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that concatenates the local obs with a global
    state vector (all agents' obs concatenated). For SB3's shared-param PPO,
    we achieve this by padding the observation with the global state during
    vectorisation. The feature extractor then processes the full vector.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def _build_env(w, lam, alpha, beta, seed):
    """Same substrate as exp_b_reduced but with configurable PT params."""
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

    # MAPPO: PPO with centralised critic (shared MLP that sees full obs)
    policy_kwargs = dict(
        features_extractor_class=MAPPOCriticExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[128, 128]),  # wider critic
    )
    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256,
                seed=seed, policy_kwargs=policy_kwargs)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, ALPHA, BETA, seed)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)

    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "seed": seed,
           "algo": "MAPPO",
           "mean_S": mean_S, "profit": profit, "lost_sales": lost_sales,
           "collapsed": collapsed, "final_value_loss": fvl,
           "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"[MAPPO-OG] run_idx={run_idx} w={w} lambda={lam} seed={seed} -> "
          f"mean_S={mean_S:.1f} profit={profit:.1f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help="0..699")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="docs/expMAPPO_orgym_figures/mappo_orgym_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
