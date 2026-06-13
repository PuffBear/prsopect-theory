"""
Exp PPO-RWARE -- PPO on Robot Warehouse with CPT reward shaping.

Replicates the supply-chain incentive-misalignment experiment on RWARE
(Robot Warehouse), a well-established MARL benchmark.  Demonstrates that
the same coordination cliff observed in the supply chain appears in a
discrete-action warehouse logistics task.

=== INCENTIVE MISALIGNMENT STRUCTURE ===

    r_local(i) = CARRY_BONUS  if agent i is carrying a shelf  (else 0)
    r_global   = mean delivery reward across agents
    r_blend(i) = (1-w) * r_local(i)  +  w * r_global

  At w=0 agents learn to always carry (hoard) -- deliveries collapse to ~0.
  At w=1 agents learn the full pick-up-deliver cycle -- good throughput.
  A sharp phase transition appears at some w* depending on lambda.

This mirrors the supply chain exactly:
  supply chain:  hoard stock at your node  ->  downstream nodes starve
  RWARE:         carry shelf indefinitely  ->  warehouse throughput collapses

=== WRAPPER STACK ===

    rware gymnasium env
      -> RWAREParallelWrapper       (gym -> PettingZoo)
      -> RWARELocalRewardWrapper    (r_local + w-blend)
      -> CPTRewardWrapper           (alpha/beta/lambda)
      -> SuperSuit vectorisation

=== SWEEPS ===

    w          in {0.0, 0.1, ..., 1.0}        (11 levels)
    condition  in {null, curvature, lambda3}  (decomposition design)
    seeds      42-51                          (10 seeds)
    Total: 11 x 3 x 10 = 330 runs

PBS array job: run_idx 0..329

=== REFERENCES ===

    Christianos et al. (2020). "Shared Experience Multi-Agent Reinforcement
    Learning." NeurIPS. (RWARE environment)
    Yu et al. (2022). "The Surprising Effectiveness of PPO in Cooperative
    Multi-Agent Games." NeurIPS. (MAPPO baseline)
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
from env.rware_wrapper import (
    make_rware_env_with_w, RWARELocalRewardWrapper, is_collapsed_rware
)
from agents.cpt_wrapper import CPTRewardWrapper

# --- Grid (v2) ----------------------------------------------------------------
# Conditions mirror the supply-chain decomposition: true null, curvature-only,
# loss aversion. (v1 swept lambda in {1,2,3} with alpha=beta=0.88 always, so
# there was no unbiased null and the decomposition could not be replicated.)
#   name              alpha  beta  lambda
CONDITIONS = [
    ("null",      1.00, 1.00, 1.0),
    ("curvature", 0.88, 0.88, 1.0),
    ("lambda3",   0.88, 0.88, 3.0),
]
W_LIST = [round(0.1 * i, 1) for i in range(11)]   # 0.0 .. 1.0
SEEDS  = list(range(42, 52))                       # 10 seeds
N_W, N_COND, N_SEED = len(W_LIST), len(CONDITIONS), len(SEEDS)
N_RUNS = N_W * N_COND * N_SEED   # 330

# Training (v2): 8 parallel env copies + entropy bonus, 2M-step default.
# RWARE delivery reward is sparse; single-copy 600k is unlikely to learn the
# w=1 end at all. GATING RULE: run the calibration endpoints and the local
# smoke test BEFORE submitting this sweep.
N_ENV_COPIES = 8
PPO_KWARGS = dict(n_steps=256, batch_size=512, ent_coef=0.01)

ENV_ID    = "rware-tiny-3ag-easy-v2"
MAX_CYCLES = 500          # steps per episode; RWARE default

COLUMNS = ["run_idx", "w", "condition", "alpha", "beta", "lambda_loss",
           "seed", "algo", "env",
           "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


# --- Index mapping -----------------------------------------------------------

def map_run_idx(run_idx):
    """run_idx -> (w, condition_tuple, seed).
    Layout: w (outer) x condition (middle) x seed (inner); block = 30."""
    block = N_COND * N_SEED  # 30
    w_idx    = run_idx // block
    rem      = run_idx % block
    c_idx    = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], CONDITIONS[c_idx], SEEDS[seed_idx]


# --- Env builder -------------------------------------------------------------

def _build_env(w, lam, alpha, beta):
    """Build the full RWARE + CPT wrapper stack."""
    local_w = make_rware_env_with_w(w, env_id=ENV_ID, max_cycles=MAX_CYCLES)
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

def evaluate_rware(model, env, num_episodes=10, base_seed=42):
    """
    Evaluate a trained model on RWARE.

    Returns mean cumulative TRUE reward per episode (= mean deliveries).
    Uses info['true_reward'] (set by RWARELocalRewardWrapper) which is the
    raw delivery reward, not the blended/CPT reward.
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


# --- Main job -----------------------------------------------------------------

def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")

    w, (cond, alpha, beta, lam), seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    # Build + vectorise training env
    train_env = _build_env(w, lam, alpha, beta)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, seed=seed, **PPO_KWARGS)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    # Evaluate on a fresh env (raw PettingZoo, not vectorised)
    eval_env = _build_env(w, lam, alpha, beta)
    mean_reward = evaluate_rware(model, eval_env, num_episodes=10,
                                 base_seed=seed * 10)

    collapsed = is_collapsed_rware(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {
        "run_idx": run_idx, "w": w, "condition": cond,
        "alpha": alpha, "beta": beta, "lambda_loss": lam, "seed": seed,
        "algo": "PPO", "env": "RWARE_tiny_3ag",
        "mean_reward": mean_reward, "collapsed": collapsed,
        "final_value_loss": fvl, "final_explained_variance": fev,
    }
    _append_row(row, out_path)
    print(f"[PPO-RWARE] run_idx={run_idx} w={w} cond={cond} lam={lam} seed={seed} -> "
          f"deliveries={mean_reward:.3f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help="0..%d" % (N_RUNS-1))
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--out", default="docs/exp_rware/ppo_rware_v2_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
