"""
High-w extension of the MAPPO-LBF sweep -- resolves the saturated high-lambda
boundaries.

The main sweep sampled only ONE w-level above 0.85 (w=0.95), so the high-lambda
(lambda>=4) boundaries, which fall in (0.85, 1), are coarsely resolved and were
reported as EXPLORATORY locations of a saturating boundary rather than precise
values (see exp_mappo_lbf.py W_LIST caveat; analyze_mappo_lbf.py flags them '*').

This driver inserts the missing interior level -- w = 0.90 -- for every
condition, so the (0.85, 1) gap is bisected and the lambda>=4 logistic fits gain
an interior anchor that pulls them off the grid ceiling. Everything else --
learner, conditions, seeds, collapse label, hyperparameters -- is IDENTICAL to
exp_mappo_lbf.py (machinery imported directly, not reimplemented).

  1 w x (null + lambda=1..7 @ a=b=0.88) x 20 seeds = 160 runs   (PBS array 0..159)

analyze_mappo_lbf.py auto-merges this file (mappo_lbf_highw_raw_data.csv) exactly
as it already merges the low-w extension, so no analyzer change is needed beyond
adding the merge path.

    python experiments/exp_mappo_lbf_highw.py --wiring_check
"""
import os
import sys
import argparse
import warnings
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Reuse the EXACT validated machinery from the main MAPPO-LBF driver
from exp_mappo_lbf import (
    build_mappo_model, _build_mappo_env, CONDITIONS, SEEDS, COLUMNS,
    _append_row, wiring_check,
)
from exp_ppo_lbf import evaluate_lbf, MetricsCallback
from env.lbf_wrapper import is_collapsed_lbf

W_LIST_HIGH = [0.90]                          # bisects the (0.85, 0.95) gap
N_W, N_COND, N_SEED = len(W_LIST_HIGH), len(CONDITIONS), len(SEEDS)
N_RUNS = N_W * N_COND * N_SEED                # 160
OUT_DEFAULT = "docs/expMAPPO_lbf_figures/mappo_lbf_highw_raw_data.csv"


def map_run_idx(run_idx):
    block = N_COND * N_SEED
    w_idx    = run_idx // block
    rem      = run_idx % block
    c_idx    = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST_HIGH[w_idx], CONDITIONS[c_idx], SEEDS[seed_idx]


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
    mean_reward = evaluate_lbf(model, eval_env, num_episodes=10, base_seed=seed * 10)

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
    print(f"[MAPPO-LBF-highw] run_idx={run_idx} w={w} cond={cond} lam={lam} seed={seed} "
          f"-> team_return={mean_reward:.3f} collapsed={collapsed}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--wiring_check", action="store_true")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); print(f"  (high-w extension: {N_RUNS} runs, w in {W_LIST_HIGH})"); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
