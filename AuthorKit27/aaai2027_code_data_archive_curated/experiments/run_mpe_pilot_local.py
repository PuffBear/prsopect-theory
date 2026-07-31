"""
Local MPE pilot runner for MacBook validation.

Runs two stages before submitting the 1020-run HPC sweep:

  STAGE 1 — Calibration (4 runs, ~8 min on M3 Pro with 4 workers):
    w=0.0  × seeds 42, 43  (herding dominant -> collapsed)
    w=1.0  × seeds 42, 43  (global dominant  -> coordinated)
    Reports w=0 vs w=1 mean true_reward, prints the suggested theta.

  STAGE 2 — Pilot sweep (12 runs, ~25 min on M3 Pro with 8 workers):
    w in {0.10, 0.25, 0.40, 0.55, 0.62, 0.70, 0.80, 0.90}  (8 w values)
    1 seed each; lambda=1 only.
    Goal: verify P(collapse) actually transitions near w*≈0.62.

Total: ~30-40 min.  A clear sigmoid confirms HPC sweep is worth running.

Usage:
  cd /path/to/prospect-theory
  python experiments/run_mpe_pilot_local.py
  python experiments/run_mpe_pilot_local.py --workers 4 --timesteps 200000

If the transition is not visible, check:
  - herding_scale (default 4.0 should be fine)
  - W_PILOT list (should straddle w*≈0.62)
  - true_reward values (large negatives = collapse; near zero = coordination)
"""
import os
import sys
import argparse
import warnings
import multiprocessing as mp
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import supersuit as ss
from stable_baselines3 import PPO

from exp_ppo_mpe import (
    _build_env, evaluate_mpe, MetricsCallback,
    ALPHA, BETA, HERDING_SCALE,
)
from env.mpe_coop_nav import load_collapse_threshold, THRESHOLD_IS_CALIBRATED

# --- Stage 1: calibration w values (just extremes) ---
W_CAL     = [0.0, 1.0]
SEEDS_CAL = [42, 43]        # 2 seeds per extreme -> 4 runs

# --- Stage 2: pilot sweep across the transition region ---
W_PILOT   = [0.10, 0.25, 0.40, 0.55, 0.62, 0.70, 0.80, 0.90]
SEED_PILOT = 42             # 1 seed per w (8 runs)
LAMBDA     = 1              # null CPT for both stages


def _train_eval_one(args):
    """Worker function: train + eval one (w, seed, timesteps, herding_scale) combo."""
    w, seed, timesteps, herding_scale = args
    warnings.filterwarnings("ignore")
    np.random.seed(seed)

    train_env = _build_env(w, LAMBDA, ALPHA, BETA, herding_scale=herding_scale)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0,
                n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=timesteps)

    eval_env = _build_env(w, LAMBDA, ALPHA, BETA, herding_scale=herding_scale)
    mean_r = evaluate_mpe(model, eval_env, num_episodes=10, base_seed=seed * 10)
    print(f"  [pilot] w={w:.2f} seed={seed} -> true_reward={mean_r:.1f}")
    return {"w": w, "seed": seed, "mean_reward": mean_r}


def run_pool(tasks, workers):
    """Run tasks in parallel with a multiprocessing pool."""
    # 'spawn' avoids PyTorch / OpenMP fork issues on macOS
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        return pool.map(_train_eval_one, tasks)


def main():
    ap = argparse.ArgumentParser(
        description="Local MPE pilot: calibration + sweep to validate phase transition"
    )
    ap.add_argument("--workers",      type=int,   default=8,
                    help="Parallel workers (default 8 for M3 Pro; use 4 if OOM)")
    ap.add_argument("--timesteps",    type=int,   default=300_000,
                    help="Timesteps per run (300k recommended; use 150k for quick check)")
    ap.add_argument("--herding_scale",type=float, default=HERDING_SCALE)
    ap.add_argument("--out_dir",      default="docs/exp_mpe", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MPE Pilot Runner  |  herding_scale={args.herding_scale}")
    print(f"Timesteps/run: {args.timesteps:,}  |  Workers: {args.workers}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ Stage 1
    print("STAGE 1: Calibration (w=0 vs w=1)")
    print("-" * 40)
    cal_tasks = [
        (w, seed, args.timesteps, args.herding_scale)
        for w in W_CAL for seed in SEEDS_CAL
    ]
    cal_results = run_pool(cal_tasks, min(args.workers, len(cal_tasks)))

    w0_rewards = [r["mean_reward"] for r in cal_results if r["w"] == 0.0]
    w1_rewards = [r["mean_reward"] for r in cal_results if r["w"] == 1.0]
    mean_w0 = float(np.mean(w0_rewards))
    mean_w1 = float(np.mean(w1_rewards))
    theta    = (mean_w0 + mean_w1) / 2.0

    print(f"\n  w=0.0 (herding): mean true_reward = {mean_w0:.1f}")
    print(f"  w=1.0 (global):  mean true_reward = {mean_w1:.1f}")
    print(f"  Suggested theta (midpoint)         = {theta:.1f}")

    gap = abs(mean_w1 - mean_w0)
    if gap < 30:
        print("\n  *** WARNING: Gap between w=0 and w=1 is small (<30).")
        print("  ***   - Check local_ratio=0.0 in mpe_coop_nav.py")
        print("  ***   - Consider increasing timesteps (--timesteps 600000)")
    else:
        print(f"\n  Gap = {gap:.1f}  ✓  (>30 — calibration looks healthy)")

    # Save calibrated threshold
    threshold_path = os.path.join(args.out_dir, "mpe_collapse_threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({
            "threshold": theta,
            "mean_w0": mean_w0,
            "mean_w1": mean_w1,
            "gap": gap,
            "timesteps": args.timesteps,
            "herding_scale": args.herding_scale,
            "note": "pilot calibration (2 seeds); re-run exp_mpe_calibrate.py (20 seeds) for final value"
        }, f, indent=2)
    print(f"  Saved: {threshold_path}")

    # ------------------------------------------------------------------ Stage 2
    print(f"\nSTAGE 2: Pilot sweep  (w = {W_PILOT})")
    print("-" * 40)
    sweep_tasks = [
        (w, SEED_PILOT, args.timesteps, args.herding_scale)
        for w in W_PILOT
    ]
    sweep_results = run_pool(sweep_tasks, min(args.workers, len(sweep_tasks)))

    print(f"\n{'w':>6}  {'true_reward':>12}  {'collapsed':>10}")
    print("-" * 34)
    n_collapsed = 0
    for r in sorted(sweep_results, key=lambda x: x["w"]):
        collapsed = r["mean_reward"] <= theta
        n_collapsed += int(collapsed)
        flag = "COLLAPSE" if collapsed else "coord"
        print(f"  {r['w']:.2f}  {r['mean_reward']:>12.1f}  {flag:>10}")

    # Simple sigmoid sanity check
    print(f"\n  Fraction collapsed (w<={W_PILOT[len(W_PILOT)//2-1]}): "
          f"{n_collapsed}/{len(W_PILOT)}")
    if n_collapsed == 0:
        print("  *** No collapse detected.  Check herding_scale or reduce w_pilot range.")
    elif n_collapsed == len(W_PILOT):
        print("  *** All collapsed.  Shift W_PILOT up, or increase timesteps.")
    else:
        print("  *** Phase transition detected. ✓  Looks good for HPC sweep.")

    # ------------------------------------------------------------------ Summary
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"  1. Inspect theta={theta:.1f}  (should be between mean_w0 and mean_w1)")
    print(f"  2. If transition is visible above: qsub docs/slurm/run_mpe_calibrate.pbs")
    print(f"       (20 seeds, 600k steps -> final theta via analyze_mpe_calibrate.py)")
    print(f"  3. Then: qsub docs/slurm/run_ppo_mpe.pbs  (1020-run sweep)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
