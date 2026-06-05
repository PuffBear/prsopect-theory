"""
Exp N — null model (no Prospect-Theory shaping).

Null condition: CPTRewardWrapper with lambda_loss=1.0, alpha=1.0, beta=1.0, which
makes v(x)=x (linear; no loss aversion, no curvature). Sweeps reward centralization
w on the base-stock substrate and labels collapse, so the collapse transition can be
measured WITHOUT any behavioral shaping and compared against the existing PT(λ=1)
300k data (docs/expA300k_figures/).

Designed as a SLURM array job (one task per run_idx 0..179). Each task trains one
PPO run and appends a single row to the shared CSV under a filelock.

NOTE ON SUBSTRATE RANGE: the reference script expA_interior_w.py (and the comparison
data expA300k) use ScaledBaseStockWrapper range [0,500]. This script reuses that
range (via expA_interior_w._build_env) on purpose:
  * the collapse label is (mean_S < 10); a [200,700] range forces mean_S >= 200 so
    collapse could never fire;
  * the comparison to expA300k requires the SAME substrate.
The [200,700] mentioned in the generic project description is the ScaledBaseStock
default, not the substrate of the collapse study.
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
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # experiments/ for the reference import
# Reuse the EXACT substrate / eval / callback from the reference script so the null
# condition differs from the PT data only in (here, nothing — lambda is already 1).
from expA_interior_w import _build_env, evaluate_and_extract_metrics, MetricsCallback

# Frozen grid (spec)
W_LIST = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEED_LIST = list(range(42, 62))  # 20 seeds
N_RUNS = len(W_LIST) * len(SEED_LIST)  # 9 * 20 = 180

# Null PT parameters: v(x) = x
LAMBDA_LOSS, ALPHA, BETA = 1.0, 1.0, 1.0

# Collapse label (spec): collapsed := (mean_S < 10) AND (profit <= -127.1)
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

COLUMNS = ["run_idx", "w", "seed", "mean_S", "profit", "lost_sales", "collapsed",
           "final_value_loss", "final_explained_variance"]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")

    w = W_LIST[run_idx // len(SEED_LIST)]
    seed = SEED_LIST[run_idx % len(SEED_LIST)]
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    # _build_env(w, lam, seed) builds the [0,500] scaled base-stock substrate with
    # CPTRewardWrapper(lambda=lam, alpha=1, beta=1, global_reward_weight=w).
    # lam=LAMBDA_LOSS=1.0 -> v(x)=x (the null condition).
    train_env = _build_env(w, LAMBDA_LOSS, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, LAMBDA_LOSS, seed)
    mean_S, _mean_order, _mean_inv, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)

    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "seed": seed,
           "mean_S": mean_S, "profit": profit, "lost_sales": lost_sales,
           "collapsed": collapsed, "final_value_loss": fvl,
           "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} w={w} seed={seed} -> mean_S={mean_S:.1f} "
          f"profit={profit:.1f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    """Process-safe single-row append using filelock (one writer at a time)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lock = FileLock(out_path + ".lock")
    with lock:
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help="0..179")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="docs/expN_figures/expN_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
