"""
Exp PPO-MPE - PPO on MPE Cooperative Navigation with CPT reward shaping.

Replicates the or-gym collapse-surface experiment on a completely different
environment (MPE simple_spread_v3) to test whether the w_crit(lambda) relationship
generalises beyond supply chains.

Key differences from or-gym experiments:
  - Environment: MPE Cooperative Navigation (3 agents, 3 landmarks)
  - No DiagnosticWrapper / scripted nodes (all agents learn)
  - No base-stock wrapper (continuous actions are direct force vectors)
  - Collapse definition: mean TRUE reward < MPE_COLLAPSE_REWARD_THRESHOLD

REWARD STACK (innermost -> outermost):
  simple_spread_v3
    -> MPELocalRewardWrapper(w)      defines r_local(i), blends with global
    -> CPTRewardWrapper(lam,a,b)     applies PT value function to blended reward
    -> SuperSuit vectorisation

  r_local(i) = -distance(agent_i, nearest_landmark)
  r_global   = mean of simple_spread's shared team reward
  r_blend(i) = (1-w)*r_local(i) + w*r_global

  Evaluation uses info['true_reward'] (raw simple_spread reward) so the
  collapse metric is environment-grounded, not warped by CPT.

NOTE: Requires 'mpe' package for PettingZoo >= 1.24:
  pip install mpe

Sweeps (per AAAI_EXPERIMENT_SPECS.md Exp 2a):
  w      in  {0.30, 0.35, ..., 0.90}  (13 levels)
  lambda in  {1}  (null CPT for environment generalizability test)
  seeds 42-61                          (20 seeds)
  Total: 13 x 20 = 260 runs

  For full lambda sweep see Exp 2b (separate script).

PBS/SLURM array job: one task per run_idx 0..259.
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
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env.mpe_coop_nav import MPELocalRewardWrapper, make_mpe_coop_nav, is_collapsed_mpe
from agents.cpt_wrapper import CPTRewardWrapper

# --- Grid (Exp 2a: w-sweep, null CPT) ----------------------------------------
W_LIST = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30..0.90
LAMBDA_LIST = [1]   # null CPT; lambda sweep is Exp 2b
SEEDS = list(range(42, 62))  # 20
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED  # 260

# Fixed CPT curvature
ALPHA, BETA = 0.88, 0.88

COLUMNS = ["run_idx", "w", "lambda_loss", "seed", "algo", "env",
           "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


class MetricsCallback(BaseCallback):
    """Records PPO value loss and explained variance per rollout."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.value_losses = []
        self.explained_variances = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is not None:
            nv = self.logger.name_to_value
            if "train/value_loss" in nv:
                self.value_losses.append(nv["train/value_loss"])
            if "train/explained_variance" in nv:
                self.explained_variances.append(nv["train/explained_variance"])


def _build_env(w, lam, alpha, beta):
    """
    Build MPE Cooperative Navigation with correct w-blending + CPT transform.

    Wrapper stack:
      simple_spread_v3
        -> MPELocalRewardWrapper(w)   r_local(i) = -dist(agent_i, nearest_lm)
        -> CPTRewardWrapper(lam,a,b)  PT value function on the blended reward
    """
    raw = make_mpe_coop_nav(n_agents=3, max_cycles=25, continuous_actions=True)
    local_w = MPELocalRewardWrapper(raw, w=w)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in local_w.possible_agents}
    # global_reward_weight=0.0 because w-blending is already done by MPELocalRewardWrapper
    cpt = CPTRewardWrapper(local_w, params, reward_scale=1.0, global_reward_weight=0.0)
    return cpt


def evaluate_mpe(model, env, num_episodes=10, base_seed=42):
    """Evaluate a trained model on MPE and return mean reward."""
    total_rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_rew = 0.0
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    real_shape = env.action_space(agent).shape[0]
                    actions[agent] = action[:real_shape]
            obs, rewards, term, trunc, info = env.step(actions)
            for agent in rewards:
                # Use true_reward if available (from CPT wrapper), else raw
                if agent in info and "true_reward" in info[agent]:
                    ep_rew += info[agent]["true_reward"]
                else:
                    ep_rew += rewards[agent]
        total_rewards.append(ep_rew)
    return np.mean(total_rewards)


def map_run_idx(run_idx):
    # Layout: iterate over w (outer), seed (inner)
    # run_idx 0..19 -> w=0.30, seeds 42..61
    # run_idx 20..39 -> w=0.35, seeds 42..61  etc.
    w_idx = run_idx // N_SEED
    seed_idx = run_idx % N_SEED
    return W_LIST[w_idx], LAMBDA_LIST[0], SEEDS[seed_idx]


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

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, ALPHA, BETA)
    mean_reward = evaluate_mpe(model, eval_env, num_episodes=10, base_seed=seed * 10)

    collapsed = is_collapsed_mpe(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "seed": seed,
           "algo": "PPO", "env": "MPE_CoopNav",
           "mean_reward": mean_reward, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"[PPO-MPE] run_idx={run_idx} w={w} lambda={lam} seed={seed} -> "
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
    ap.add_argument("--run_idx", type=int, required=True, help="0..259")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="docs/expPPO_mpe_figures/ppo_mpe_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
