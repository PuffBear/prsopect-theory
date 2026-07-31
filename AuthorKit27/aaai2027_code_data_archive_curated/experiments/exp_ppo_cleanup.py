"""
Exp PPO-Cleanup -- PPO on the Cleanup Sequential Social Dilemma with CPT shaping.

Replicates the supply-chain incentive-misalignment experiment on the Cleanup
task (Leibo et al. 2017), an established benchmark for studying the tragedy
of the commons in MARL.

=== INCENTIVE MISALIGNMENT STRUCTURE ===

    r_local(i) = apples collected by agent i this step
    r_global   = mean apples collected per agent this step (team harvest rate)
    r_blend(i) = (1-w) * r_local(i)  +  w * r_global

  At w=0: all agents harvest -- no one cleans.  Waste accumulates, apple spawn
  rate collapses to ~0.  TRAGEDY OF THE COMMONS.
  At w=1: agents learn to specialise -- some clean (maintaining apple supply),
  others harvest.  COORDINATION.

This mirrors the supply-chain collapse exactly:
  supply chain:  hoard stock  ->  downstream nodes starve
  Cleanup:       harvest only ->  waste grows -> apple supply collapses

=== WRAPPER STACK ===

    CleanupEnv (PettingZoo ParallelEnv, custom implementation)
      -> CleanupLocalRewardWrapper    (r_local = apple harvest, w-blend)
      -> CPTRewardWrapper             (alpha/beta/lambda)
      -> SuperSuit vectorisation

=== SWEEPS ===

    w      in {0.10, 0.15, ..., 0.90}  (17 levels)
    lambda in {1, 2, 3}                (null CPT + two CPT conditions)
    seeds  42-61                       (20 seeds)
    Total: 17 x 3 x 20 = 1020 runs

PBS array job: run_idx 0..1019

=== REFERENCES ===

    Leibo, J.Z. et al. (2017). Multi-agent Reinforcement Learning in Sequential
    Social Dilemmas. AAMAS.  arXiv:1702.03037
    Hughes, E. et al. (2018). Inequity Aversion Improves Cooperation in
    Intertemporal Social Dilemmas. NeurIPS.
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
from env.cleanup_env import (
    make_cleanup_env_with_w, CleanupLocalRewardWrapper, is_collapsed_cleanup
)
from agents.cpt_wrapper import CPTRewardWrapper

# --- Grid -------------------------------------------------------------------
W_LIST    = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10..0.90
LAMBDA_LIST = [1, 2, 3]
SEEDS     = list(range(42, 62))
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED   # 1020

ALPHA, BETA = 0.88, 0.88
MAX_CYCLES = 200   # episode length (steps)

COLUMNS = ["run_idx", "w", "lambda_loss", "seed", "algo", "env",
           "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


# --- Index mapping -----------------------------------------------------------

def map_run_idx(run_idx):
    lam_seed_block = N_LAM * N_SEED  # 60
    w_idx    = run_idx // lam_seed_block
    rem      = run_idx % lam_seed_block
    lam_idx  = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], LAMBDA_LIST[lam_idx], SEEDS[seed_idx]


# --- Env builder -------------------------------------------------------------

def _build_env(w, lam, alpha, beta):
    local_w = make_cleanup_env_with_w(w, n_agents=3, max_cycles=MAX_CYCLES)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in local_w.possible_agents}
    cpt = CPTRewardWrapper(local_w, params, reward_scale=1.0,
                           global_reward_weight=0.0)
    return cpt


# --- Training callbacks ------------------------------------------------------

class MetricsCallback(BaseCallback):
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


# --- Evaluation --------------------------------------------------------------

def evaluate_cleanup(model, env, num_episodes=10, base_seed=42):
    """
    Evaluate on Cleanup.  Returns mean cumulative true reward per episode
    (= mean total apples harvested across the team).
    """
    total_rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_rew = 0.0
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    act, _ = model.predict(obs[agent], deterministic=True)
                    actions[agent] = int(act)
            obs, rewards, term, trunc, info = env.step(actions)
            for agent in rewards:
                if agent in info and "true_reward" in info[agent]:
                    ep_rew += info[agent]["true_reward"]
                else:
                    ep_rew += rewards[agent]
        total_rewards.append(ep_rew)
    return float(np.mean(total_rewards))


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

    model = PPO("MlpPolicy", vec, verbose=0,
                n_steps=256, batch_size=512, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, ALPHA, BETA)
    mean_reward = evaluate_cleanup(model, eval_env, num_episodes=10,
                                   base_seed=seed * 10)

    collapsed = is_collapsed_cleanup(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {
        "run_idx": run_idx, "w": w, "lambda_loss": lam, "seed": seed,
        "algo": "PPO", "env": "Cleanup_SSD",
        "mean_reward": mean_reward, "collapsed": collapsed,
        "final_value_loss": fvl, "final_explained_variance": fev,
    }
    _append_row(row, out_path)
    print(f"[PPO-Cleanup] run_idx={run_idx} w={w} lambda={lam} seed={seed} -> "
          f"mean_reward={mean_reward:.3f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help="0..1019")
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--out", default="docs/exp_cleanup/ppo_cleanup_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
