"""
Exp MAPPO-MPE -- MAPPO on MPE Cooperative Navigation with CPT reward shaping.

Cross-algorithm replication: MAPPO on MPE simple_spread_v3.
MAPPO is PPO with a wider centralised critic (CTDE; Yu et al. 2022).
The environment and wrapper stack are IDENTICAL to exp_ppo_mpe.py so that
the only variable between the two experiments is the algorithm.

=== WRAPPER STACK (same as PPO-MPE) ===

    simple_spread_v3  (local_ratio=0.0)
      -> MPELocalRewardWrapper(w, herding_scale=HERDING_SCALE)
      -> CPTRewardWrapper(alpha, beta, lambda)
      -> SuperSuit vectorisation

=== SWEEP GRID (same as PPO-MPE) ===

    w          in {0.0, 0.1, ..., 1.0}        (11 levels)
    condition  in {null, curvature, lambda3}  (decomposition design)
    seeds      42-61                          (20 seeds; AAAI rev.)
    Total: 11 x 3 x 20 = 660 runs

PBS array: run_idx 0..659
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
from env.mpe_coop_nav import (
    MPELocalRewardWrapper, make_mpe_coop_nav,
    is_collapsed_mpe, is_collapsed_mpe_occupancy, HERDING_SCALE,
)
from env.global_state_wrapper import GlobalStateWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from agents.mappo_policy import CentralizedCriticPolicy
from stable_baselines3.common.callbacks import CallbackList
from exp_ppo_mpe import MetricsCallback, evaluate_mpe, EntCoefDecay

# --- Sweep grid (must match exp_ppo_mpe.py exactly) -------------------------
# Grid, conditions, training config, and CSV schema are IMPORTED from
# exp_ppo_mpe so the two experiments can never drift apart (v2 design).
from exp_ppo_mpe import (
    CONDITIONS, W_LIST, SEEDS, N_W, N_COND, N_SEED, N_RUNS,
    N_ENV_COPIES, PPO_KWARGS, COLUMNS, map_run_idx as _map_run_idx,
)


# --- Index mapping -----------------------------------------------------------

def map_run_idx(run_idx):
    """Same layout as exp_ppo_mpe (w outer x condition middle x seed inner)."""
    return _map_run_idx(run_idx)


# --- Env builder: SAME stack as PPO + GlobalStateWrapper for the critic ------
# GlobalStateWrapper is OUTERMOST: it only augments observations (appends the
# joint state for the centralized critic); rewards / info (true_reward,
# r_global, occupancy) pass through unchanged, so eval is identical to PPO.

def _build_env(w, lam, alpha, beta, herding_scale=HERDING_SCALE):
    raw     = make_mpe_coop_nav(n_agents=3, max_cycles=50, continuous_actions=False)  # v4: 25->50 lifts occupancy ceiling
    local_w = MPELocalRewardWrapper(raw, w=w, herding_scale=herding_scale)
    params  = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
               for a in local_w.possible_agents}
    cpt = CPTRewardWrapper(local_w, params, reward_scale=1.0,
                           global_reward_weight=0.0)
    return GlobalStateWrapper(cpt)


# --- Shared MAPPO training setup (one path for sweep + calibrate + smoke) ----

def build_mappo_vec(w, alpha, beta, lam, seed, herding_scale=HERDING_SCALE):
    env = _build_env(w, lam, alpha, beta, herding_scale=herding_scale)
    local_dim = env.local_dim                      # set by GlobalStateWrapper
    vec = ss.pad_action_space_v0(env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                base_class='stable_baselines3')
    _inner = getattr(vec, "venv", vec)
    vec.seed = lambda s, _v=_inner: _v.reset(seed=int(s))
    return vec, _inner, local_dim


def train_mappo_model(w, alpha, beta, lam, seed, timesteps,
                      herding_scale=HERDING_SCALE, verbose=0):
    vec, _inner, local_dim = build_mappo_vec(w, alpha, beta, lam, seed,
                                             herding_scale)
    policy_kwargs = dict(local_dim=local_dim,
                         net_arch=dict(pi=[64, 64], vf=[128, 128]))
    model = PPO(CentralizedCriticPolicy, vec, verbose=verbose, seed=seed,
                policy_kwargs=policy_kwargs, **PPO_KWARGS)
    metrics = MetricsCallback()
    _inner.reset(seed=int(seed))
    model.learn(total_timesteps=timesteps,
                callback=CallbackList([metrics, EntCoefDecay()]))
    return model, metrics


# --- Main job ----------------------------------------------------------------

def run_one(run_idx, timesteps, out_path, herding_scale=HERDING_SCALE):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, (cond, alpha, beta, lam), seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, cb = train_mappo_model(w, alpha, beta, lam, seed, timesteps,
                                  herding_scale=herding_scale)

    eval_env  = _build_env(w, lam, alpha, beta, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe(model, eval_env, num_episodes=10,
                                         base_seed=seed * 10)

    collapsed = is_collapsed_mpe_occupancy(mean_occ)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "condition": cond,
           "alpha": alpha, "beta": beta, "lambda_loss": lam, "seed": seed,
           "algo": "MAPPO", "env": "MPE_CoopNav",
           "mean_reward": mean_reward, "mean_occupancy": mean_occ,
           "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"[MAPPO-MPE] run_idx={run_idx} w={w} cond={cond} lam={lam} seed={seed} "
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
    ap.add_argument("--run_idx",       type=int,   required=True, help="0..%d" % (N_RUNS-1))
    ap.add_argument("--timesteps",     type=int,   default=2_000_000)
    ap.add_argument("--herding_scale", type=float, default=HERDING_SCALE)
    ap.add_argument("--out", default="docs/expMAPPO_mpe_figures/mappo_mpe_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out,
            herding_scale=args.herding_scale)


if __name__ == "__main__":
    main()
