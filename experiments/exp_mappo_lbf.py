"""
Exp MAPPO-LBF -- centralized-critic MAPPO on Level-Based Foraging with CPT
reward shaping. THIRD ENVIRONMENT (breadth result).

Why this experiment: the primary boundary is on the supply chain (PPO+MAPPO)
and replicates on MPE (MAPPO). Reviewers weak-rejected on environment breadth
("single supply-chain environment dominates"). LBF is a standard MARL benchmark
(Christianos et al. 2020 / Papoudakis et al.) in a third, structurally distinct
domain (grid foraging, public-goods incentive). The local smoke GATE passed
decisively -- MAPPO coordinates (w=1 -> 0.906) while w=0 free-rides (0.334),
a 0.57 separation with wide interior headroom for w_crit. This driver runs the
full sweep that turns that gate into the breadth result: w_crit rising with
loss aversion in a third environment.

WHY MAPPO (not PPO): PPO/IPPO never coordinates on LBF (the w=1 team-reward end
is a credit-assignment failure for independent learners). A centralized critic
fixes it -- the same situation, and the same fix, as MPE. Running LBF under
MAPPO also matches the algorithm used for MPE, so the breadth evidence reads as
"algorithm fixed (MAPPO), environment varied (MPE -> LBF)", cleanly isolating
the environment axis.

REAL MAPPO (not a renamed PPO): GlobalStateWrapper sits OUTERMOST and augments
each agent's obs to [ own_obs | onehot_id | global_state ]; CentralizedCriticPolicy
routes the actor to obs[:, :local_dim] and the critic to obs[:, local_dim:].
This is the identical, validated machinery from run_lbf_smoke_local.py (PART 0
wiring check) and exp_mappo_mpe.py / exp_mappo_orgym.py.

    >>> BEFORE COMMITTING CLUSTER TIME:
          python experiments/exp_mappo_lbf.py --wiring_check

Sweep (full decomposition, mirrors the supply-chain / MPE design):
  w          in {0.3, 0.5, 0.65, 0.75, 0.85, 0.95}        (6)
  condition  null (a=b=1, lam=1) + lambda-sweep a=b=0.88, lam in {1..7}  (8)
  seeds      42-61                                          (20)
  Total: 6 x 8 x 20 = 960 runs                              (PBS array 0..959)

The lam=1, a=b=0.88 condition is the curvature reference; null (a=b=1) anchors
the curvature shift. The loss-aversion direction is the lam=1..7 progression at
a=b=0.88. Collapse label: team return < theta (calibrated, env-grounded; see
env/lbf_wrapper.py and docs/exp_lbf/lbf_collapse_threshold.json).
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

# Validated LBF stack + helpers (identical to the PPO-LBF and smoke-gate code)
from exp_ppo_lbf import (
    _build_env as _build_ppo_env, evaluate_lbf, MetricsCallback,
    N_ENV_COPIES, PPO_KWARGS,
)
from env.lbf_wrapper import is_collapsed_lbf
from env.global_state_wrapper import GlobalStateWrapper
from agents.mappo_policy import CentralizedCriticPolicy

# --- Grid --------------------------------------------------------------------
# Wider w-grid than the supply chain because LBF's null boundary location is not
# yet pinned; this brackets an interior w_crit at every lambda and leaves room
# for the rightward loss-aversion shift. (Smoke: w=0 collapses, w=1 coordinates.)
W_LIST = [0.3, 0.5, 0.65, 0.75, 0.85, 0.95]
# AAAI rev. caveat: only ONE sampled level lies above 0.85 (w=0.95). The
# high-lambda (lambda>=5) boundaries fall in [0.85, 1], so their w_crit estimates
# are COARSELY RESOLVED and are reported as EXPLORATORY locations of a saturating
# boundary, not precise values. analyze_mappo_lbf.py flags lambda>=4 with '*'.
# To resolve them, add finer w in (0.85, 1) -- e.g. {0.90} -- and re-run.
# condition = (name, alpha, beta, lambda). null anchors the curvature shift;
# lambda1@0.88 is the curvature condition; lambda1..7@0.88 is the LA sweep.
CONDITIONS = [("null", 1.00, 1.00, 1.0)] + [
    (f"lambda{L}", 0.88, 0.88, float(L)) for L in range(1, 8)
]
SEEDS = list(range(42, 62))  # 20
N_W, N_COND, N_SEED = len(W_LIST), len(CONDITIONS), len(SEEDS)
N_RUNS = N_W * N_COND * N_SEED  # 960

COLUMNS = ["run_idx", "w", "condition", "alpha", "beta", "lambda_loss",
           "seed", "algo", "env",
           "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


def _build_mappo_env(w, lam, alpha, beta):
    """The validated LBF CPT stack with GlobalStateWrapper OUTERMOST so the
    centralized critic reads the joint state. Rewards/info pass through, so
    evaluate_lbf (uses info['true_reward']) is unchanged."""
    return GlobalStateWrapper(_build_ppo_env(w, lam, alpha, beta))


def build_mappo_model(w, lam, alpha, beta, seed):
    env = _build_mappo_env(w, lam, alpha, beta)
    local_dim = env.local_dim  # set by GlobalStateWrapper
    vec = ss.pad_action_space_v0(env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                base_class='stable_baselines3')
    _inner = getattr(vec, "venv", vec)
    vec.seed = lambda s, _v=_inner: _v.reset(seed=int(s))
    policy_kwargs = dict(local_dim=local_dim,
                         net_arch=dict(pi=[64, 64], vf=[128, 128]))
    model = PPO(CentralizedCriticPolicy, vec, verbose=0, seed=seed,
                policy_kwargs=policy_kwargs, **PPO_KWARGS)
    return model, _inner


def map_run_idx(run_idx):
    """w (outer) x condition (middle) x seed (inner); block = N_COND*N_SEED."""
    block = N_COND * N_SEED
    w_idx    = run_idx // block
    rem      = run_idx % block
    c_idx    = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], CONDITIONS[c_idx], SEEDS[seed_idx]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, (cond, alpha, beta, lam), seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, _inner = build_mappo_model(w, lam, alpha, beta, seed)
    cb = MetricsCallback()
    _inner.reset(seed=int(seed))
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_mappo_env(w, lam, alpha, beta)
    mean_reward = evaluate_lbf(model, eval_env, num_episodes=10,
                               base_seed=seed * 10)

    collapsed = is_collapsed_lbf(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {
        "run_idx": run_idx, "w": w, "condition": cond,
        "alpha": alpha, "beta": beta, "lambda_loss": lam, "seed": seed,
        "algo": "MAPPO", "env": "LBF_5x5_2p_2f",
        "mean_reward": mean_reward, "collapsed": collapsed,
        "final_value_loss": fvl, "final_explained_variance": fev,
    }
    _append_row(row, out_path)
    print(f"[MAPPO-LBF] run_idx={run_idx} w={w} cond={cond} lam={lam} seed={seed} "
          f"-> team_return={mean_reward:.3f} collapsed={collapsed}")
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
    model, _inner = build_mappo_model(0.75, 7.0, 0.88, 0.88, seed=42)
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
    print(f"  grid: {N_W} w x {N_COND} cond x {N_SEED} seeds = {N_RUNS} runs")
    print("  WIRING CHECK PASS: centralized critic wired, trainable, reads joint state.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--out", default="docs/expMAPPO_lbf_figures/mappo_lbf_raw_data.csv")
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
