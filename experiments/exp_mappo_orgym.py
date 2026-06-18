"""
Exp MAPPO-OG - TRUE centralized-critic MAPPO on or-gym NetworkManagement-v1
with CPT reward shaping.

Second algorithm on the PRIMARY environment. Mirrors exp_b_reduced.py (same
substrate, same w/lambda grid, same collapse label, same PPO hyperparameters)
and changes ONE thing: the critic is centralized (CTDE MAPPO, Yu et al. 2022).
This isolates "does a centralized critic move the loss-aversion boundary?" from
every other factor, answering the single-algorithm reviewer concern.

REAL MAPPO (not a renamed PPO):
  - env.global_state_wrapper.GlobalStateWrapper sits OUTERMOST and augments every
    agent's obs to [ own_obs | onehot_id | global_state ] (global_state = concat
    of all learners' own_parts). It only touches observations.
  - agents.mappo_policy.CentralizedCriticPolicy:
        ACTOR  reads obs[:, :local_dim]      (decentralized execution)
        CRITIC reads obs[:, local_dim:]      (centralized training, joint state)
  This is the SAME validated machinery used by exp_mappo_mpe.py.

  >>> BEFORE COMMITTING CLUSTER TIME: run the wiring check
        python experiments/exp_mappo_orgym.py --wiring_check
      It asserts the centralized critic is built, trainable, and reads the
      joint-state slice (guards against silently falling back to IPPO).

Sweeps:
  w       in {0.5, 0.65, 0.75, 0.85, 0.95}   (5)
  lambda  in {1, 2, 3, 4, 5, 6, 7}           (7)
  seeds   42-61                              (20)
  Total: 5 x 7 x 20 = 700 runs               (PBS array 0..699)

Fixed CPT curvature: alpha = beta = 0.88.
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
from expA_interior_w import (evaluate_and_extract_metrics, is_collapsed, MetricsCallback,
                             LOW_BOUND, HIGH_BOUND)
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper
from env.global_state_wrapper import GlobalStateWrapper
from agents.mappo_policy import CentralizedCriticPolicy

# --- Grid (identical to exp_b_reduced / exp_ppo_lambda_sweep) ----------------
W_LIST = [0.5, 0.65, 0.75, 0.85, 0.95]
LAMBDA_LIST = [1, 2, 3, 4, 5, 6, 7]
SEEDS = list(range(42, 62))  # 20
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED  # 700

ALPHA, BETA = 0.88, 0.88
S_FLOOR, PROFIT_CEIL = 10.0, -127.1  # frozen collapse label

# Same PPO hyperparameters as the PPO supply-chain runs (only the critic differs)
PPO_KWARGS = dict(n_steps=128, batch_size=256)

COLUMNS = ["run_idx", "w", "lambda_loss", "seed", "algo",
           "mean_S", "profit", "lost_sales", "collapsed",
           "final_value_loss", "final_explained_variance"]


def _build_env(w, lam, alpha, beta):
    """Same substrate as exp_b_reduced, with GlobalStateWrapper OUTERMOST so the
    centralized critic can read the joint state. Rewards/eval pass through it."""
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta} for a in interv.possible_agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    base = ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)
    return GlobalStateWrapper(base)


def build_mappo_vec(w, lam, alpha, beta, seed):
    """Vectorised env + local_dim for the centralized-critic policy."""
    env = _build_env(w, lam, alpha, beta)
    local_dim = env.local_dim  # set by GlobalStateWrapper
    vec = ss.pad_action_space_v0(env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    _inner = getattr(vec, "venv", vec)
    vec.seed = lambda s, _v=_inner: _v.reset(seed=int(s))
    return vec, _inner, local_dim


def build_mappo_model(w, lam, alpha, beta, seed):
    vec, _inner, local_dim = build_mappo_vec(w, lam, alpha, beta, seed)
    policy_kwargs = dict(local_dim=local_dim,
                         net_arch=dict(pi=[64, 64], vf=[128, 128]))
    model = PPO(CentralizedCriticPolicy, vec, verbose=0, seed=seed,
                policy_kwargs=policy_kwargs, **PPO_KWARGS)
    return model, _inner


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

    model, _inner = build_mappo_model(w, lam, ALPHA, BETA, seed)
    cb = MetricsCallback()
    _inner.reset(seed=int(seed))
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, ALPHA, BETA)
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


def wiring_check():
    """Assert the centralized critic is built, trainable, and reads joint state."""
    from agents.mappo_policy import _CentralCriticExtractor
    model, _inner = build_mappo_model(0.75, 7, ALPHA, BETA, seed=42)
    pol = model.policy
    obs_dim = model.observation_space.shape[0]
    local_dim = pol._local_dim
    assert isinstance(pol.vf_features_extractor, _CentralCriticExtractor), \
        "vf extractor is NOT the centralized critic (silently IPPO!)"
    opt_params = {id(p) for g in pol.optimizer.param_groups for p in g["params"]}
    crit = list(pol.vf_features_extractor.parameters())
    assert crit and all(id(p) in opt_params for p in crit), \
        "centralized critic params are NOT in the optimizer!"
    assert pol.vf_features_extractor.net[0].in_features == obs_dim - local_dim, \
        "critic input != global-state slice"
    print(f"  local_dim={local_dim} augmented_obs={obs_dim} "
          f"global_slice={obs_dim - local_dim}")
    print("  WIRING CHECK PASS: centralized critic wired, trainable, reads joint state.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..699")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="docs/expMAPPO_orgym_figures/mappo_orgym_raw_data.csv")
    ap.add_argument("--wiring_check", action="store_true",
                    help="validate the centralized critic and exit (no training)")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
