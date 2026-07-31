"""
MAgent collapse-threshold calibration (gated, mirrors exp_lbf_calibrate.py).

Trains the NULL-condition endpoints (w=0 selfish vs w=1 team-reward) with the
exact main-sweep config and logs mean red-team TRUE reward per eval episode. The
analyzer sets theta = midpoint of the COLLAPSED regime (w=0, selfish -> team gets
picked apart) and the COORDINATED regime (w=1, focus-fire) -- the same
collapsed-vs-coordinated midpoint used for LBF/MPE.

Grid: w in {0.0, 1.0} x seeds 42-51 -> 20 runs. PBS array 0..19.
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
from exp_mappo_magent import build_mappo_model, evaluate_magent
from exp_ppo_lbf import MetricsCallback

W_CAL  = [0.0, 1.0]
SEEDS  = list(range(42, 52))
N_RUNS = len(W_CAL) * len(SEEDS)
ALPHA, BETA, LAMBDA = 1.0, 1.0, 1.0   # null condition

COLUMNS = ["run_idx", "w", "seed", "mean_reward",
           "final_value_loss", "final_explained_variance"]


def map_run_idx(run_idx):
    return W_CAL[run_idx // len(SEEDS)], SEEDS[run_idx % len(SEEDS)]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model = build_mappo_model(w, LAMBDA, ALPHA, BETA, seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    mean_reward = evaluate_magent(model, w, LAMBDA, ALPHA, BETA,
                                  num_episodes=10, base_seed=seed * 10)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "seed": seed,
           "mean_reward": mean_reward,
           "final_value_loss": fvl, "final_explained_variance": fev}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)
    print(f"[MAgent-calib] run_idx={run_idx} w={w} seed={seed} "
          f"-> team_reward={mean_reward:.3f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx",   type=int, required=True, help="0..19")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--out", default="docs/exp_magent/magent_calibration_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
