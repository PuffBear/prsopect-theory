"""
Low-w extension of the MAPPO-LBF entropy ablation.

The main entropy sweep used w in {0.3,...,0.95}; at low exploration noise
(ent_coef=0) the boundary sat BELOW w=0.3, so w_crit(ent=0) was sub-grid. This
adds w in {0.0, 0.1, 0.2} at every ent_coef so the low-noise boundary lands
in-grid and w_crit(ent=0) becomes a located value.

Identical learner / condition / collapse label / seeds to
exp_mappo_lbf_entropy; only W_LIST changes. analyze_lbf_entropy.py auto-merges.

GRID
----
  w        in {0.0, 0.1, 0.2}                  (3)
  ent_coef in {0.0, 0.001, 0.01, 0.05}         (4, same as main)
  seeds    42-61                                (20)
  Total: 3 x 4 x 20 = 240 runs   ->  PBS array 0..239

    python experiments/exp_mappo_lbf_entropy_loww.py --wiring_check
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

from exp_mappo_lbf import _build_mappo_env
from exp_ppo_lbf import evaluate_lbf, MetricsCallback, N_ENV_COPIES, PPO_KWARGS
from env.lbf_wrapper import is_collapsed_lbf
from agents.mappo_policy import CentralizedCriticPolicy
# reuse the entropy driver's model builder (overrides ent_coef) for consistency
from exp_mappo_lbf_entropy import build_mappo_model

W_LIST = [0.0, 0.1, 0.2]
ENT_COEFS = [0.0, 0.001, 0.01, 0.05]
SEEDS = list(range(42, 62))
ALPHA, BETA, LAMBDA = 0.88, 0.88, 1.0
HORIZON = 2_000_000
N_W, N_ENT, N_SEED = len(W_LIST), len(ENT_COEFS), len(SEEDS)
N_RUNS = N_W * N_ENT * N_SEED               # 240

COLUMNS = ["run_idx", "w", "ent_coef", "alpha", "beta", "lambda_loss", "seed",
           "algo", "env", "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


def _append_row(row, out_path, columns):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=columns).to_csv(
            out_path, mode="a", header=write_header, index=False)


def map_run_idx(run_idx):
    """w (outer) x ent_coef (middle) x seed (inner) -- same layout as main."""
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
    print(f"[MAPPO-LBF-entropy-loww] run_idx={run_idx} w={w} ent_coef={ent} seed={seed} "
          f"-> team_return={mean_reward:.3f} collapsed={collapsed}", flush=True)
    return row


def wiring_check():
    from agents.mappo_policy import _CentralCriticExtractor
    model, _ = build_mappo_model(0.1, LAMBDA, ALPHA, BETA, seed=42, ent_coef=0.05)
    assert isinstance(model.policy.vf_features_extractor, _CentralCriticExtractor), \
        "vf extractor is NOT the centralized critic!"
    assert abs(model.ent_coef - 0.05) < 1e-9, "ent_coef not applied!"
    print(f"  centralized critic wired; ent_coef override applied "
          f"(model.ent_coef={model.ent_coef})")
    print(f"  WIRING CHECK PASS  (LBF entropy low-w: {N_RUNS} runs, w in {W_LIST})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--out", default="docs/expMAPPO_lbf_entropy_figures/mappo_lbf_entropy_loww_raw_data.csv")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.out)


if __name__ == "__main__":
    main()
