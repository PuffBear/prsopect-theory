"""
MAPPO-MPE collapse-threshold calibration (centralized critic, occupancy-based).

Trains the endpoints (w=0 herding-collapse vs w=1 coordination, NULL condition)
with the EXACT main-sweep config (shared train_mappo_model: centralized critic +
agent-ID + 50-step horizon), and logs mean per-step landmark occupancy.

Grid: w in {0.0, 1.0} x seeds 42-51 -> 20 runs. PBS array 0..19.

After the array finishes:  python experiments/analyze_mappo_mpe_calibrate.py
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
from exp_mappo_mpe import train_mappo_model, _build_env
from exp_ppo_mpe import evaluate_mpe, HERDING_SCALE

W_CAL  = [0.0, 1.0]
SEEDS  = list(range(42, 52))
N_RUNS = len(W_CAL) * len(SEEDS)        # 20
ALPHA, BETA, LAMBDA = 1.0, 1.0, 1.0     # null condition

COLUMNS = ["run_idx", "w", "seed", "mean_reward", "mean_occupancy",
           "final_value_loss", "final_explained_variance"]


def map_run_idx(run_idx):
    return W_CAL[run_idx // len(SEEDS)], SEEDS[run_idx % len(SEEDS)]


def run_one(run_idx, timesteps, out_path, herding_scale=HERDING_SCALE):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, cb = train_mappo_model(w, ALPHA, BETA, LAMBDA, seed, timesteps,
                                  herding_scale=herding_scale)
    eval_env = _build_env(w, LAMBDA, ALPHA, BETA, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe(model, eval_env, num_episodes=10,
                                         base_seed=seed * 10)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "seed": seed,
           "mean_reward": mean_reward, "mean_occupancy": mean_occ,
           "final_value_loss": fvl, "final_explained_variance": fev}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)
    print(f"[MAPPO-MPE-calib] run_idx={run_idx} w={w} seed={seed} "
          f"-> r_global={mean_reward:.1f} occ={mean_occ:.3f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx",   type=int, required=True, help="0..19")
    ap.add_argument("--timesteps", type=int, default=4_000_000)
    ap.add_argument("--out", default="docs/exp_mpe/mappo_mpe_calibration_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
