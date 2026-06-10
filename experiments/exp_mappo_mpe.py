"""
Exp MAPPO-MPE - MAPPO on MPE Cooperative Navigation with CPT reward shaping.

Cross-algorithm + cross-environment replication: MAPPO on MPE simple_spread_v3.
This is the "full factorial" cell: {PPO, MAPPO} x {or-gym, MPE}.

Sweeps:
  w  in  {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}  (6 levels)
  lambda  in  {1, 2, 3, 4, 5, 6, 7}             (7 levels)
  seeds 42-61                            (20 seeds)
  Total: 6 x 7 x 20 = 840 runs

Fixed CPT curvature: alpha = beta = 0.88.

PBS/SLURM array job: one task per run_idx 0..839.
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
from env.mpe_coop_nav import make_mpe_coop_nav, is_collapsed_mpe
from agents.cpt_wrapper import CPTRewardWrapper
from exp_ppo_mpe import MetricsCallback, evaluate_mpe

# --- Grid -------------------------------------------------------------------
W_LIST = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LAMBDA_LIST = [1, 2, 3, 4, 5, 6, 7]
SEEDS = list(range(42, 62))  # 20
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED  # 840

# Fixed CPT curvature
ALPHA, BETA = 0.88, 0.88

COLUMNS = ["run_idx", "w", "lambda_loss", "seed", "algo", "env",
           "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


# --- MAPPO: centralised critic via custom feature extractor -----------------
class MAPPOCriticExtractor(BaseFeaturesExtractor):
    """Wider critic network for MAPPO centralised value function."""
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


def _build_env(w, lam, alpha, beta):
    """Build MPE Cooperative Navigation with CPT reward shaping."""
    raw = make_mpe_coop_nav(n_agents=3, max_cycles=25, continuous_actions=True)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in raw.possible_agents}
    cpt = CPTRewardWrapper(raw, params, reward_scale=1.0, global_reward_weight=w)
    return cpt


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

    train_env = _build_env(w, lam, ALPHA, BETA)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    # MAPPO: PPO with centralised critic
    policy_kwargs = dict(
        features_extractor_class=MAPPOCriticExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[128, 128]),
    )
    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256,
                seed=seed, policy_kwargs=policy_kwargs)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, ALPHA, BETA)
    mean_reward = evaluate_mpe(model, eval_env, num_episodes=10, base_seed=seed * 10)

    collapsed = is_collapsed_mpe(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "seed": seed,
           "algo": "MAPPO", "env": "MPE_CoopNav",
           "mean_reward": mean_reward, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"[MAPPO-MPE] run_idx={run_idx} w={w} lambda={lam} seed={seed} -> "
          f"mean_reward={mean_reward:.1f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help="0..839")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="docs/expMAPPO_mpe_figures/mappo_mpe_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
