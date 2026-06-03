"""
Experiment 0 — convergence / horizon study (BLOCKING; gates A's claim and all of B/D).

The horizon check (w=0.9: 4/20 collapsed @150k -> 0/20 @400k) showed transition-
collapse is metastable: w_crit may be a property of the 150k TRAINING BUDGET, not of
the system's equilibria. This experiment settles it.

Design: w in {0.7, 0.8, 0.9} (straddling w_crit=0.82), 8 seeds each, on the SCALED
[0,500] substrate (same as Exp A; standard collapse label applies — no offset-bug
risk). Each run trained to 600k with collapse logged at checkpoints 150k/300k/600k
via segmented learn() with reset_num_timesteps=False.

Readout: collapse fraction per (w, checkpoint).
  - If collapse fraction at the transition cells STABILIZES across checkpoints
    -> w_crit is a property of the equilibria -> real phase transition.
  - If it keeps falling toward 0 with more training -> "collapse" is just
    "undertrained" -> the collapse framing is a horizon artifact.

    python experiments/exp0_convergence.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import warnings

import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # experiments/ for expA_interior_w
from expA_interior_w import _build_env, evaluate_and_extract_metrics, is_collapsed

CHECKPOINTS = [150000, 300000, 600000]


def _run_single(args):
    w, seed = args
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    train_env = _build_env(w, 1.0, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None
    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)

    eval_env = _build_env(w, 1.0, seed)
    rows = []
    prev = 0
    for ckpt in CHECKPOINTS:
        model.learn(total_timesteps=ckpt - prev, reset_num_timesteps=(prev == 0))
        prev = ckpt
        mean_S, mean_order, mean_inv, mean_ls, profit, bw = evaluate_and_extract_metrics(
            model, eval_env, num_episodes=10, base_seed=seed * 10)
        rows.append({"w": w, "seed": seed, "checkpoint": ckpt,
                     "mean_S": mean_S, "profit": profit, "lost_sales": mean_ls,
                     "bullwhip": bw, "collapsed": is_collapsed(mean_S, profit)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="0.7,0.8,0.9")
    ap.add_argument("--seeds", default="42-49")  # 8 seeds
    ap.add_argument("--procs", type=int, default=min(mp.cpu_count(), 10))
    ap.add_argument("--out", default="docs/exp0_figures/exp0_convergence_data.csv")
    args = ap.parse_args()

    ws = [float(x) for x in args.ws.split(",")]
    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    jobs = [(w, s) for w in ws for s in seeds]
    print(f"Exp 0 convergence: {len(ws)} w x {len(seeds)} seeds = {len(jobs)} runs to 600k "
          f"(checkpoints {CHECKPOINTS}), procs={args.procs}")
    os.makedirs("docs/exp0_figures", exist_ok=True)

    all_rows = []
    with mp.Pool(processes=args.procs) as pool:
        for i, rows in enumerate(pool.imap_unordered(_run_single, jobs), 1):
            all_rows.extend(rows)
            r600 = [r for r in rows if r["checkpoint"] == 600000][0]
            print(f"[{i:2d}/{len(jobs)}] w={rows[0]['w']:.2f} seed={rows[0]['seed']:2d} -> "
                  f"collapsed@[150k,300k,600k]={[r['collapsed'] for r in rows]} profit600k={r600['profit']:.0f}")
            pd.DataFrame(all_rows).to_csv(args.out, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print("\n=== Collapse fraction by (w, checkpoint) ===")
    print(df.pivot_table(index="w", columns="checkpoint", values="collapsed", aggfunc="mean").round(3).to_string())
    print("\nStable across checkpoints -> real transition. Falling toward 0 -> horizon artifact.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
