"""
Exp PPO-MPE -- PPO on MPE Cooperative Navigation with CPT reward shaping.

Demonstrates that the same coordination cliff observed in the supply chain
also appears in MPE Cooperative Navigation, a standard robotics/navigation
benchmark (Lowe et al. 2017).

=== INCENTIVE MISALIGNMENT STRUCTURE ===

    r_local(i) = -HERDING_SCALE * mean_dist(agent_i, other_agents)
    r_global   = -sum_lm(min_dist(landmark, agents))  [pure team score]
    r_blend(i) = (1-w) * r_local(i)  +  w * r_global

  At w=0: herding instinct dominates.  Agents cluster, no landmark is
  covered, true_reward collapses.  COLLAPSE.
  At w=1: global team score dominates; some free-riding but agents learn
  to spread.  COORDINATION.

  Theoretical w_crit ≈ 0.62 with HERDING_SCALE=4.0 (see env/mpe_coop_nav.py).
  This places the transition centrally in W_LIST = [0.10..0.90].

=== WRAPPER STACK ===

    simple_spread_v3  (local_ratio=0.0 -- pure global score; enforced by
                      make_mpe_coop_nav)
      -> MPELocalRewardWrapper(w, herding_scale=4.0)
      -> CPTRewardWrapper(alpha, beta, lambda)
      -> SuperSuit vectorisation

=== SWEEPS ===

    w          in {0.0, 0.1, ..., 1.0}        (11 levels)
    condition  in {null, curvature, lambda3}  (decomposition design)
    seeds      42-51                          (10 seeds per cell)
    Total: 11 x 3 x 10 = 330 runs

PBS array job: run_idx 0..329

=== REPRODUCIBILITY NOTES ===

- local_ratio=0.0 is set in make_mpe_coop_nav(); do NOT change this.
  If local_ratio != 0, r_global and true_reward are contaminated.
- CPTRewardWrapper preserves info[agent]["true_reward"] set by the inner
  wrapper (raw global team score); it does NOT overwrite it.
- Collapse threshold is data-driven: run exp_mpe_calibrate.py (40 jobs)
  then analyze_mpe_calibrate.py BEFORE submitting this sweep.
  The placeholder threshold (-100) in env/mpe_coop_nav.py is approximate.
- Evaluation uses true_reward (raw global, not CPT-transformed) so the
  collapse metric is environment-grounded.
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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.mpe_coop_nav import (
    MPELocalRewardWrapper, make_mpe_coop_nav,
    is_collapsed_mpe, is_collapsed_mpe_occupancy, HERDING_SCALE,
)
from agents.cpt_wrapper import CPTRewardWrapper

# --- Sweep grid (v2) ---------------------------------------------------------
# Conditions mirror the supply-chain decomposition (Exp N / Exp B-Reduced /
# Exp T): a true null, a curvature-only cell, and a loss-aversion cell.
# The v1 grid (lambda in {1,2,3} with alpha=beta=0.88 ALWAYS) had NO null
# condition, so the curvature-vs-lambda decomposition could not be replicated.
#   name              alpha  beta  lambda
CONDITIONS = [
    ("null",      1.00, 1.00, 1.0),   # no behavioral bias
    ("curvature", 0.88, 0.88, 1.0),   # probability weighting only
    ("lambda3",   0.88, 0.88, 3.0),   # weighting + loss aversion
]
W_LIST = [round(0.1 * i, 1) for i in range(11)]   # 0.0, 0.1, ..., 1.0
SEEDS  = list(range(42, 52))                       # 10 seeds
N_W, N_COND, N_SEED = len(W_LIST), len(CONDITIONS), len(SEEDS)
N_RUNS = N_W * N_COND * N_SEED   # 330

# Backwards-compat constants (calibration script imports these)
ALPHA, BETA = 0.88, 0.88

# Training (v2): 600k steps with a single env copy did not learn the w=1
# (pure global) end -- trained policies scored BELOW a no-op baseline
# (-96 vs -46 episode r_global; see scripted-policy probe 2026-06-13).
# Fixes: 8 parallel env copies, larger rollout, entropy bonus, 2M steps.
# v3: DISCRETE actions. The continuous Gaussian policy blew up under the
# entropy bonus (std -> 1.6 by 2M steps; performance peaked at 300k then
# degraded -- local smoke test 2026-06-13). Discrete MPE is also the standard
# benchmark setup (Yu et al. 2022). ent_coef=0.01 is safe for categorical.
N_ENV_COPIES = 8

# --- Training strength (v4) --------------------------------------------------
# v3 was reproducible but UNDER-TRAINED at the coordination end: trained w=1
# reached only ~0.26 landmark occupancy vs the scripted-spread ceiling of ~0.64
# (calibration 2026-06-13), with value_loss ~40-60 and explained_variance ~0.5
# still mediocre at 2M steps -- the policy never sharpened into coverage.
# v4 levers (all CRUX-SAFE: occupancy is reward-scale-invariant, and a CONSTANT
# reward scale + decay schedules preserve the relative lambda / alpha-beta
# magnitudes the CPT decomposition depends on -- unlike VecNormalize, which
# would normalise those differences away per-run):
#   (1) entropy decay 0.01 -> 1e-3 -> the deterministic-eval policy commits to
#       covering landmarks late in training instead of staying diffuse
#   (2) larger rollout (n_steps 512) -> lower-variance advantages for the
#       harder coordination credit-assignment at high w
#   (3) longer horizon (default 4M; set per-PBS)
# NOTE: LR is kept CONSTANT at 3e-4. A v4a attempt with linear LR annealing to
# zero crippled training (the policy is far from optimal -- 0.26 vs 0.64 ceiling
# -- so it needs learning BUDGET, not late-stage refinement). Reward scaling was
# also dropped (REWARD_SCALE=1.0): it added no benefit under PPO's advantage
# normalisation and would perturb the absolute lambda/alpha-beta magnitudes the
# CPT decomposition compares. Occupancy (the gate metric) is scale-invariant.
REWARD_SCALE = 1.0
ENT_START, ENT_END = 0.01, 0.001


class EntCoefDecay(BaseCallback):
    """Linearly decay PPO's entropy coefficient start -> end over training."""

    def __init__(self, start=ENT_START, end=ENT_END):
        super().__init__()
        self.start, self.end = float(start), float(end)

    def _on_step(self) -> bool:
        total = getattr(self.model, "_total_timesteps", 0) or 1
        frac = min(self.num_timesteps / total, 1.0)
        self.model.ent_coef = self.start + (self.end - self.start) * frac
        return True


PPO_KWARGS = dict(
    n_steps=512,
    batch_size=512,
    ent_coef=ENT_START,
    learning_rate=3e-4,
)

COLUMNS = ["run_idx", "w", "condition", "alpha", "beta", "lambda_loss",
           "seed", "algo", "env",
           "mean_reward", "mean_occupancy", "collapsed",
           "final_value_loss", "final_explained_variance"]


# --- Callbacks --------------------------------------------------------------

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


# --- Index mapping ----------------------------------------------------------

def map_run_idx(run_idx):
    """
    run_idx -> (w, condition_tuple, seed).
    Layout: w (outer) x condition (middle) x seed (inner).
      run_idx 0..9    -> w=0.0, null,      seeds 42..51
      run_idx 10..19  -> w=0.0, curvature, seeds 42..51
      run_idx 20..29  -> w=0.0, lambda3,   seeds 42..51
      run_idx 30..39  -> w=0.1, null,      seeds 42..51   etc.
    """
    block = N_COND * N_SEED  # 30
    w_idx    = run_idx // block
    rem      = run_idx % block
    c_idx    = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], CONDITIONS[c_idx], SEEDS[seed_idx]


# --- Env builder ------------------------------------------------------------

def _build_env(w, lam, alpha, beta, herding_scale=HERDING_SCALE):
    """
    Build MPE Cooperative Navigation with w-blend + CPT transform.

    Wrapper stack:
      simple_spread_v3  (local_ratio=0.0)
        -> MPELocalRewardWrapper(w, herding_scale)
        -> CPTRewardWrapper(lam, alpha, beta)
    """
    raw = make_mpe_coop_nav(n_agents=3, max_cycles=50, continuous_actions=False)
    local_w = MPELocalRewardWrapper(raw, w=w, herding_scale=herding_scale)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in local_w.possible_agents}
    # global_reward_weight=0.0 because w-blending is already done by
    # MPELocalRewardWrapper; CPTRewardWrapper only applies the PT transform.
    cpt = CPTRewardWrapper(local_w, params, reward_scale=REWARD_SCALE,
                           global_reward_weight=0.0)
    return cpt


# --- Shared training setup (v4) ---------------------------------------------
# One code path for sweep + calibration + smoke so config CANNOT drift between
# them (the drift between smoke and HPC calibration is what produced the
# spurious LBF result). Returns (trained_model, metrics_callback).

def build_mpe_vec(w, alpha, beta, lam, seed, herding_scale=HERDING_SCALE):
    train_env = _build_env(w, lam, alpha, beta, herding_scale=herding_scale)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                base_class='stable_baselines3')
    # reproducible, decorrelated env-spawn seeding (v3 fix)
    _inner = getattr(vec, "venv", vec)
    vec.seed = lambda s, _v=_inner: _v.reset(seed=int(s))
    return vec, _inner


def train_mpe_model(w, alpha, beta, lam, seed, timesteps,
                    herding_scale=HERDING_SCALE, verbose=0):
    vec, _inner = build_mpe_vec(w, alpha, beta, lam, seed, herding_scale)
    model = PPO("MlpPolicy", vec, verbose=verbose, seed=seed, **PPO_KWARGS)
    metrics = MetricsCallback()
    _inner.reset(seed=int(seed))
    model.learn(total_timesteps=timesteps,
                callback=CallbackList([metrics, EntCoefDecay()]))
    return model, metrics


# --- Evaluation -------------------------------------------------------------

def evaluate_mpe(model, env, num_episodes=10, base_seed=42):
    """
    Evaluate a trained model using the PURE GLOBAL COVERAGE metric.

    Uses info[agent]['r_global'] set by MPELocalRewardWrapper — the raw
    simple_spread global team score (-sum_l min_a dist(a,l)), added ONCE
    per step (not per agent; all agents share the same r_global value).

    Returns (mean_episode_r_global, mean_per_step_occupancy).

    PRIMARY collapse metric is occupancy (see env/mpe_coop_nav.py v2 collapse
    definition). Measured scripted-policy brackets (2026-06-13, 20 seeds):
      cluster/still/random: episode r_global ≈ -46..-48, occupancy ≈ 0.0-0.1
      greedy spread:        episode r_global ≈ -21,      occupancy ≈ 0.8-1.0

    NOTE: do NOT sum true_reward per agent — that triple-counts the global
    score (one per agent × 3 agents) and produces inverted calibration results.
    """
    total_rewards, total_occs = [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_rew, ep_occ, n_steps = 0.0, 0.0, 0
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    space = env.action_space(agent)
                    if hasattr(space, "shape") and space.shape:
                        actions[agent] = action[:space.shape[0]]   # continuous
                    else:
                        actions[agent] = int(action)               # discrete
            obs, rewards, term, trunc, info = env.step(actions)
            # r_global / occupancy are identical across agents — count once/step
            added = False
            for agent in rewards:
                if agent in info and "r_global" in info[agent]:
                    ep_rew += float(info[agent]["r_global"])
                    ep_occ += float(info[agent].get("occupancy", float("nan")))
                    added = True
                    break
            if not added:
                # fallback: mean raw reward as proxy
                ep_rew += sum(rewards.values()) / max(len(rewards), 1)
                ep_occ += float("nan")
            n_steps += 1
        total_rewards.append(ep_rew)
        total_occs.append(ep_occ / max(n_steps, 1))
    return float(np.mean(total_rewards)), float(np.mean(total_occs))


# --- Main job ---------------------------------------------------------------

def run_one(run_idx, timesteps, out_path, herding_scale=HERDING_SCALE):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, (cond, alpha, beta, lam), seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, cb = train_mpe_model(w, alpha, beta, lam, seed, timesteps,
                                herding_scale=herding_scale)

    eval_env = _build_env(w, lam, alpha, beta, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe(model, eval_env, num_episodes=10,
                                         base_seed=seed * 10)

    collapsed = is_collapsed_mpe_occupancy(mean_occ)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {
        "run_idx": run_idx, "w": w, "condition": cond,
        "alpha": alpha, "beta": beta, "lambda_loss": lam, "seed": seed,
        "algo": "PPO", "env": "MPE_CoopNav",
        "mean_reward": mean_reward, "mean_occupancy": mean_occ,
        "collapsed": collapsed,
        "final_value_loss": fvl, "final_explained_variance": fev,
    }
    _append_row(row, out_path)
    print(f"[PPO-MPE] run_idx={run_idx} w={w} cond={cond} lam={lam} seed={seed} "
          f"-> r_global={mean_reward:.1f} occ={mean_occ:.2f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx",      type=int,   required=True, help="0..%d" % (N_RUNS-1))
    ap.add_argument("--timesteps",    type=int,   default=4_000_000)
    ap.add_argument("--herding_scale",type=float, default=HERDING_SCALE)
    ap.add_argument("--out", default="docs/expPPO_mpe_figures/ppo_mpe_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out,
            herding_scale=args.herding_scale)


if __name__ == "__main__":
    main()
