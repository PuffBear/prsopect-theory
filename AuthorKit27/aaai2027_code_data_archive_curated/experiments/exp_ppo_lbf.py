"""
Exp PPO-LBF -- PPO on Level-Based Foraging with CPT reward shaping (v2 design).

Second transfer substrate (replaces RWARE; see env/lbf_wrapper.py docstring
for the rationale). Demonstrates the coordination cliff on a standard MARL
benchmark with NATIVE local-vs-team incentives.

=== SWEEP ===

    w          in {0.0, 0.1, ..., 1.0}        (11 levels)
    condition  in {null, curvature, lambda3}  (decomposition design)
    seeds      42-51                          (10 seeds)
    Total: 11 x 3 x 10 = 330 runs

PBS array job: run_idx 0..329

=== GATING ===

  1. Local smoke test: run_lbf_smoke_local.py --train must PASS.
  2. Calibration endpoints + analyze_lbf_calibrate.py must print "Gate PASSED".
  Only then submit run_ppo_lbf_v2.pbs.
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
from env.lbf_wrapper import (
    make_lbf_env_with_w, is_collapsed_lbf, DEFAULT_ENV_ID, MAX_CYCLES,
)
from agents.cpt_wrapper import CPTRewardWrapper

# --- Grid (v2 conditions, same as MPE v2) ------------------------------------
CONDITIONS = [
    ("null",      1.00, 1.00, 1.0),
    ("curvature", 0.88, 0.88, 1.0),
    ("lambda3",   0.88, 0.88, 3.0),
]
W_LIST = [round(0.1 * i, 1) for i in range(11)]
SEEDS  = list(range(42, 52))
N_W, N_COND, N_SEED = len(W_LIST), len(CONDITIONS), len(SEEDS)
N_RUNS = N_W * N_COND * N_SEED   # 330

N_ENV_COPIES = 8
PPO_KWARGS = dict(n_steps=256, batch_size=512, ent_coef=0.01)

COLUMNS = ["run_idx", "w", "condition", "alpha", "beta", "lambda_loss",
           "seed", "algo", "env",
           "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


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


def map_run_idx(run_idx):
    """w (outer) x condition (middle) x seed (inner); block = 30."""
    block = N_COND * N_SEED
    w_idx    = run_idx // block
    rem      = run_idx % block
    c_idx    = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], CONDITIONS[c_idx], SEEDS[seed_idx]


def _build_env(w, lam, alpha, beta):
    local_w = make_lbf_env_with_w(w, env_id=DEFAULT_ENV_ID, max_cycles=MAX_CYCLES)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in local_w.possible_agents}
    return CPTRewardWrapper(local_w, params, reward_scale=1.0,
                            global_reward_weight=0.0)


def evaluate_lbf(model, env, num_episodes=10, base_seed=42):
    """Mean TEAM RETURN per episode: sum of native rewards over agents+steps.
    Independent of w/alpha/beta/lambda (uses info['true_reward'])."""
    totals = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        R = 0.0
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    act, _ = model.predict(obs[agent], deterministic=True)
                    actions[agent] = int(act)
            obs, rewards, term, trunc, info = env.step(actions)
            for agent in rewards:
                if agent in info and "true_reward" in info[agent]:
                    R += float(info[agent]["true_reward"])
                else:
                    R += float(rewards[agent])
        totals.append(R)
    return float(np.mean(totals))


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, (cond, alpha, beta, lam), seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    train_env = _build_env(w, lam, alpha, beta)
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

    eval_env = _build_env(w, lam, alpha, beta)
    mean_reward = evaluate_lbf(model, eval_env, num_episodes=10,
                               base_seed=seed * 10)

    collapsed = is_collapsed_lbf(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {
        "run_idx": run_idx, "w": w, "condition": cond,
        "alpha": alpha, "beta": beta, "lambda_loss": lam, "seed": seed,
        "algo": "PPO", "env": "LBF_5x5_2p_2f",
        "mean_reward": mean_reward, "collapsed": collapsed,
        "final_value_loss": fvl, "final_explained_variance": fev,
    }
    _append_row(row, out_path)
    print(f"[PPO-LBF] run_idx={run_idx} w={w} cond={cond} lam={lam} seed={seed} "
          f"-> team_return={mean_reward:.3f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx",   type=int, required=True,
                    help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--out", default="docs/exp_lbf/ppo_lbf_v2_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
