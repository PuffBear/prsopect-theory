"""
Exp 1 - matched-condition 600k null run (closes the metastability confound).

Produces a 600k row for the (w, T) heatmap under the SAME condition as the
150k/300k rows (alpha=beta=1, lambda=1) so the horizon comparison is controlled at every
row. The existing 600k data (Exp T) used alpha=beta=0.88, conflating curvature with
horizon; this run removes that.

Substrate / eval / collapse label are reused verbatim from expA_interior_w
(or-gym NetworkManagement-v1, ScaledBaseStockWrapper Sin[0,500], collapse from
commit 03c3fc6). `_build_env(w, 1.0, seed)` already sets alpha=beta=1, lambda=1.

Grid: w in {0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80} (transition zone, shifted
left of the 300k transition), 20 seeds -> 160 runs. SLURM/PBS array: one task
per run_idx 0..159.
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
from expA_interior_w import _build_env, evaluate_and_extract_metrics, MetricsCallback

# Grid (spec): transition zone for the 600k horizon
W_LIST = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SEEDS = list(range(42, 62))  # 20
N_RUNS = len(W_LIST) * len(SEEDS)  # 160

LAMBDA_LOSS = 1.0          # null: no loss aversion
# alpha=beta=1 are baked into expA._build_env, matching the 150k/300k heatmap rows.
S_FLOOR, PROFIT_CEIL = 10.0, -127.1   # collapse label (commit 03c3fc6)

COLUMNS = ["run_idx", "w", "seed", "mean_S", "profit", "lost_sales", "collapsed",
           "final_value_loss", "final_explained_variance"]


def map_run_idx(run_idx):
    return W_LIST[run_idx // len(SEEDS)], SEEDS[run_idx % len(SEEDS)]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    train_env = _build_env(w, LAMBDA_LOSS, seed)   # alpha=beta=1, lambda=1
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, LAMBDA_LOSS, seed)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)

    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "seed": seed, "mean_S": mean_S,
           "profit": profit, "lost_sales": lost_sales, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)
    print(f"[Exp1-600k] run_idx={run_idx} w={w} seed={seed} -> "
          f"mean_S={mean_S:.1f} profit={profit:.1f} collapsed={collapsed}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help="0..159")
    ap.add_argument("--horizon", "--timesteps", dest="horizon", type=int, default=600_000)
    ap.add_argument("--out", default="docs/exp0b_extended/exp0b_600k_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.horizon, args.out)


if __name__ == "__main__":
    main()
