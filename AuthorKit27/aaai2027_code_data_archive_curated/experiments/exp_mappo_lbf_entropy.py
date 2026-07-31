"""
Exp MAPPO-LBF entropy ablation (exploration-noise generalization).

Tests whether the exploration-noise result --- w_crit RISES as exploration noise
(the PPO entropy coefficient) grows, a direct prediction of the mean-field
signal-to-noise mechanism --- generalizes from the supply chain
(exp_entropy_ablation.py) to LBF.

Mirrors the supply-chain entropy ablation: sweep w x ent_coef x seed at a fixed
behavioral condition, estimate w_crit(ent_coef); if w_crit rises with ent_coef,
the boundary is an SNR crossing in this environment too (predict-then-confirm).

Reuses the validated MAPPO-LBF env builder + collapse label; the ONLY change is
that PPO's entropy coefficient is the swept variable (the LBF default ent_coef in
exp_ppo_lbf.PPO_KWARGS is overridden per run).

GRID
----
  w        in {0.3, 0.5, 0.65, 0.75, 0.85, 0.95}     (6, LBF w-grid)
  ent_coef in {0.0, 0.001, 0.01, 0.05}               (4, matches supply ablation)
  seeds    42-61                                      (20)
  Total: 6 x 4 x 20 = 480 runs   ->  PBS array 0..479

Condition fixed at the curvature point (alpha=beta=0.88, lambda=1), matching the
supply-chain entropy ablation's fixed-lambda design.

    python experiments/exp_mappo_lbf_entropy.py --wiring_check
"""
import os
import sys
import argparse
import warnings
import numpy as np

import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from filelock import FileLock
from exp_mappo_lbf import _build_mappo_env


def _append_row(row, out_path, columns):
    """Local append writing THIS driver's columns (imported _append_row uses the
    base module's COLUMNS and silently drops 'ent_coef')."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=columns).to_csv(
            out_path, mode="a", header=write_header, index=False)
from exp_ppo_lbf import evaluate_lbf, MetricsCallback, N_ENV_COPIES, PPO_KWARGS
from env.lbf_wrapper import is_collapsed_lbf
from agents.mappo_policy import CentralizedCriticPolicy

W_LIST = [0.3, 0.5, 0.65, 0.75, 0.85, 0.95]
ENT_COEFS = [0.0, 0.001, 0.01, 0.05]        # exploration noise (matches supply)
SEEDS = list(range(42, 62))                 # 20
ALPHA, BETA, LAMBDA = 0.88, 0.88, 1.0
HORIZON = 2_000_000                          # LBF's standard budget
N_W, N_ENT, N_SEED = len(W_LIST), len(ENT_COEFS), len(SEEDS)
N_RUNS = N_W * N_ENT * N_SEED               # 480

COLUMNS = ["run_idx", "w", "ent_coef", "alpha", "beta", "lambda_loss", "seed",
           "algo", "env", "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


def build_mappo_model(w, lam, alpha, beta, seed, ent_coef):
    """Same as exp_mappo_lbf.build_mappo_model but with ent_coef OVERRIDDEN (the
    swept exploration-noise term). All other PPO_KWARGS (n_steps, batch_size) and
    the centralized-critic policy are identical."""
    env = _build_mappo_env(w, lam, alpha, beta)
    local_dim = env.local_dim
    vec = ss.pad_action_space_v0(env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                base_class='stable_baselines3')
    _inner = getattr(vec, "venv", vec)
    vec.seed = lambda s, _v=_inner: _v.reset(seed=int(s))
    policy_kwargs = dict(local_dim=local_dim,
                         net_arch=dict(pi=[64, 64], vf=[128, 128]))
    kwargs = dict(PPO_KWARGS); kwargs["ent_coef"] = ent_coef   # override the swept term
    model = PPO(CentralizedCriticPolicy, vec, verbose=0, seed=seed,
                policy_kwargs=policy_kwargs, **kwargs)
    return model, _inner


def map_run_idx(run_idx):
    """w (outer) x ent_coef (middle) x seed (inner)."""
    block = N_ENT * N_SEED
    w_idx = run_idx // block
    rem = run_idx % block
    e_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], ENT_COEFS[e_idx], SEEDS[seed_idx]


def run_one(run_idx, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, ent, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, _inner = build_mappo_model(w, LAMBDA, ALPHA, BETA, seed, ent)
    cb = MetricsCallback()
    _inner.reset(seed=int(seed))
    model.learn(total_timesteps=HORIZON, callback=cb)

    eval_env = _build_mappo_env(w, LAMBDA, ALPHA, BETA)
    mean_reward = evaluate_lbf(model, eval_env, num_episodes=20, base_seed=seed * 10)
    collapsed = is_collapsed_lbf(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "ent_coef": ent, "alpha": ALPHA,
           "beta": BETA, "lambda_loss": LAMBDA, "seed": seed, "algo": "MAPPO",
           "env": "LBF_5x5_2p_2f", "mean_reward": mean_reward, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path, COLUMNS)
    print(f"[MAPPO-LBF-entropy] run_idx={run_idx} w={w} ent_coef={ent} seed={seed} "
          f"-> team_return={mean_reward:.3f} collapsed={collapsed}", flush=True)
    return row


def wiring_check():
    """Assert the centralized critic is built and ent_coef is actually applied."""
    from agents.mappo_policy import _CentralCriticExtractor
    model, _inner = build_mappo_model(0.75, LAMBDA, ALPHA, BETA, seed=42, ent_coef=0.05)
    pol = model.policy
    assert isinstance(pol.vf_features_extractor, _CentralCriticExtractor), \
        "vf extractor is NOT the centralized critic (silently IPPO!)"
    assert abs(model.ent_coef - 0.05) < 1e-9, \
        f"ent_coef not applied: model.ent_coef={model.ent_coef} != 0.05"
    print(f"  centralized critic wired; ent_coef override applied "
          f"(model.ent_coef={model.ent_coef})")
    print(f"  WIRING CHECK PASS  (LBF entropy: {N_RUNS} runs, ent in {ENT_COEFS})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--out", default="docs/expMAPPO_lbf_entropy_figures/mappo_lbf_entropy_raw_data.csv")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.out)


if __name__ == "__main__":
    main()
