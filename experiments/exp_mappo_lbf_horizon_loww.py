"""
Low-w extension of the MAPPO-LBF horizon (metastability) sweep.

The main horizon sweep used w in {0.3,...,0.95}; at the long horizon (4M) the
collapse boundary slid BELOW w=0.3 (collapse <15% even at w=0.3), so w_crit was
sub-grid and only reportable as a bound. This adds the missing low-w cells ---
w in {0.0, 0.1, 0.2} --- at BOTH horizons so the slid boundary lands in-grid and
w_crit(4M) becomes a located value, not just a direction.

Identical learner / condition / collapse label / seeds to exp_mappo_lbf_horizon;
only W_LIST changes. analyze_lbf_horizon.py auto-merges this file.

GRID
----
  w        in {0.0, 0.1, 0.2}                  (3, the missing low-w cells)
  horizon  in {1_000_000, 4_000_000}           (2, same as main)
  seeds    42-61                                (20)
  Total: 3 x 2 x 20 = 120 runs   ->  PBS array 0..119

NOTE: this driver writes its OWN columns (incl. 'horizon') via a local
_append_row -- the main horizon driver's first run dropped that column (it reused
the base module's COLUMNS); fixed here and recoverable from run_idx regardless.

    python experiments/exp_mappo_lbf_horizon_loww.py --wiring_check
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from filelock import FileLock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from exp_mappo_lbf import build_mappo_model, _build_mappo_env, wiring_check
from exp_ppo_lbf import evaluate_lbf, MetricsCallback
from env.lbf_wrapper import is_collapsed_lbf

W_LIST = [0.0, 0.1, 0.2]                     # the missing low-w cells
HORIZONS = [1_000_000, 4_000_000]
SEEDS = list(range(42, 62))
ALPHA, BETA, LAMBDA = 0.88, 0.88, 1.0       # same curvature condition as main
N_W, N_H, N_SEED = len(W_LIST), len(HORIZONS), len(SEEDS)
N_RUNS = N_W * N_H * N_SEED                  # 120

COLUMNS = ["run_idx", "w", "horizon", "alpha", "beta", "lambda_loss", "seed",
           "algo", "env", "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


def _append_row(row, out_path, columns):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=columns).to_csv(
            out_path, mode="a", header=write_header, index=False)


def map_run_idx(run_idx):
    """w (outer) x horizon (middle) x seed (inner) -- same layout as the main
    horizon driver so the analyzer's run_idx recovery is consistent if needed."""
    block = N_H * N_SEED
    w_idx = run_idx // block
    rem = run_idx % block
    h_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], HORIZONS[h_idx], SEEDS[seed_idx]


def run_one(run_idx, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, horizon, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, _inner = build_mappo_model(w, LAMBDA, ALPHA, BETA, seed)
    cb = MetricsCallback()
    _inner.reset(seed=int(seed))
    model.learn(total_timesteps=horizon, callback=cb)

    eval_env = _build_mappo_env(w, LAMBDA, ALPHA, BETA)
    mean_reward = evaluate_lbf(model, eval_env, num_episodes=20, base_seed=seed * 10)
    collapsed = is_collapsed_lbf(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "horizon": horizon, "alpha": ALPHA,
           "beta": BETA, "lambda_loss": LAMBDA, "seed": seed, "algo": "MAPPO",
           "env": "LBF_5x5_2p_2f", "mean_reward": mean_reward, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path, COLUMNS)
    print(f"[MAPPO-LBF-horizon-loww] run_idx={run_idx} w={w} T={horizon} seed={seed} "
          f"-> team_return={mean_reward:.3f} collapsed={collapsed}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--out", default="docs/expMAPPO_lbf_horizon_figures/mappo_lbf_horizon_loww_raw_data.csv")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); print(f"  (LBF horizon low-w: {N_RUNS} runs, w in {W_LIST})"); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.out)


if __name__ == "__main__":
    main()
