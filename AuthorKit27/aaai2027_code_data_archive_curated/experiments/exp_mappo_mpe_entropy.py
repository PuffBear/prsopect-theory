"""
Exp MAPPO-MPE entropy ablation (exploration-noise generalization).

Tests whether w_crit RISES with exploration noise (PPO entropy coefficient) ---
the mean-field signal-to-noise prediction --- on MPE Cooperative Navigation, as
on the supply chain (exp_entropy_ablation.py).

IMPORTANT (subtle): the base MPE training (exp_mappo_mpe.train_mappo_model)
ANNEALS the entropy coefficient via an EntCoefDecay callback (0.01 -> 0.001 over
training). For an entropy ABLATION the swept ent_coef must be held FIXED, so this
driver uses its own train loop that sets a constant ent_coef and does NOT attach
EntCoefDecay. Everything else (env builder, centralized critic, occupancy label,
budget) is identical to exp_mappo_mpe.

GRID
----
  w        in {0.0, 0.1, ..., 1.0}        (11, MPE w-grid)
  ent_coef in {0.0, 0.001, 0.01, 0.05}    (4, matches supply ablation; FIXED)
  seeds    42-61                          (20)
  Total: 11 x 4 x 20 = 880 runs   ->  PBS array 0..879

Condition fixed at the curvature point (alpha=beta=0.88, lambda=1).

    python experiments/exp_mappo_mpe_entropy.py --wiring_check
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from filelock import FileLock
from exp_mappo_mpe import build_mappo_vec, _build_env


def _append_row(row, out_path, columns):
    """Local append writing THIS driver's columns (imported _append_row uses the
    base module's COLUMNS and silently drops 'ent_coef')."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=columns).to_csv(
            out_path, mode="a", header=write_header, index=False)
from exp_ppo_mpe import evaluate_mpe, MetricsCallback, PPO_KWARGS
from agents.mappo_policy import CentralizedCriticPolicy
from env.mpe_coop_nav import is_collapsed_mpe_occupancy, HERDING_SCALE

W_LIST = [round(0.1 * i, 1) for i in range(11)]
ENT_COEFS = [0.0, 0.001, 0.01, 0.05]        # exploration noise (FIXED, not annealed)
SEEDS = list(range(42, 62))                 # 20
ALPHA, BETA, LAMBDA = 0.88, 0.88, 1.0
HORIZON = 4_000_000                          # MPE standard budget
N_W, N_ENT, N_SEED = len(W_LIST), len(ENT_COEFS), len(SEEDS)
N_RUNS = N_W * N_ENT * N_SEED               # 880

COLUMNS = ["run_idx", "w", "ent_coef", "alpha", "beta", "lambda_loss", "seed",
           "algo", "env", "mean_reward", "mean_occupancy", "collapsed",
           "final_value_loss", "final_explained_variance"]


def train_fixed_entropy(w, alpha, beta, lam, seed, timesteps, ent_coef,
                        herding_scale=HERDING_SCALE):
    """Like exp_mappo_mpe.train_mappo_model but with a CONSTANT ent_coef and NO
    EntCoefDecay (so the swept exploration noise is actually held fixed)."""
    vec, _inner, local_dim = build_mappo_vec(w, alpha, beta, lam, seed, herding_scale)
    policy_kwargs = dict(local_dim=local_dim,
                         net_arch=dict(pi=[64, 64], vf=[128, 128]))
    kwargs = dict(PPO_KWARGS); kwargs["ent_coef"] = ent_coef   # fixed override
    model = PPO(CentralizedCriticPolicy, vec, verbose=0, seed=seed,
                policy_kwargs=policy_kwargs, **kwargs)
    metrics = MetricsCallback()
    _inner.reset(seed=int(seed))
    model.learn(total_timesteps=timesteps, callback=CallbackList([metrics]))  # NO decay
    return model, metrics


def map_run_idx(run_idx):
    block = N_ENT * N_SEED
    w_idx = run_idx // block
    rem = run_idx % block
    e_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], ENT_COEFS[e_idx], SEEDS[seed_idx]


def run_one(run_idx, out_path, herding_scale=HERDING_SCALE):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, ent, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, cb = train_fixed_entropy(w, ALPHA, BETA, LAMBDA, seed, HORIZON, ent,
                                    herding_scale=herding_scale)
    eval_env = _build_env(w, LAMBDA, ALPHA, BETA, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe(model, eval_env, num_episodes=10,
                                         base_seed=seed * 10)
    collapsed = is_collapsed_mpe_occupancy(mean_occ)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "ent_coef": ent, "alpha": ALPHA,
           "beta": BETA, "lambda_loss": LAMBDA, "seed": seed, "algo": "MAPPO",
           "env": "MPE_CoopNav", "mean_reward": mean_reward,
           "mean_occupancy": mean_occ, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path, COLUMNS)
    print(f"[MAPPO-MPE-entropy] run_idx={run_idx} w={w} ent_coef={ent} seed={seed} "
          f"-> occ={mean_occ:.3f} collapsed={collapsed}", flush=True)
    return row


def wiring_check():
    """Assert the centralized critic AND that ent_coef is fixed (no decay)."""
    from agents.mappo_policy import _CentralCriticExtractor
    model, _ = train_fixed_entropy(0.5, ALPHA, BETA, LAMBDA, seed=42,
                                   timesteps=0, ent_coef=0.05)
    assert isinstance(model.policy.vf_features_extractor, _CentralCriticExtractor), \
        "vf extractor is NOT the centralized critic!"
    assert abs(float(model.ent_coef) - 0.05) < 1e-9, \
        f"ent_coef not fixed at 0.05 (got {model.ent_coef}) -- decay may be active!"
    print(f"  centralized critic wired; ent_coef FIXED at {model.ent_coef} (no decay)")
    print(f"  WIRING CHECK PASS  (MPE entropy: {N_RUNS} runs, ent in {ENT_COEFS})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--out", default="docs/expMAPPO_mpe_entropy_figures/mappo_mpe_entropy_raw_data.csv")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.out)


if __name__ == "__main__":
    main()
