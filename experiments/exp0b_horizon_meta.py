"""
Exp 0b — horizon lock (Q1) + metastability origin (Q2). See docs/exp0b_preregistration.md.

Piece 1: w in {0.6,0.7} x seeds 42-61, single learn() to 600k.
Piece 2: w in {0.3,0.6,0.7,0.8} x seeds 42-49 x {single(300k), segmented(150k+150k)}.

Label: is_collapsed(mean_S_singlereset, profit) — same as the 300k re-estimation, and
the scaled mean_S mapping is correct for the scaled substrate. converged_S (last-10k
logged S) recorded alongside as a cross-check.
"""
import os
# thread-limit BEFORE numpy/torch import to avoid the load-46 oversubscription thrash
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import warnings

import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from expA_interior_w import (_build_env, evaluate_and_extract_metrics, is_collapsed,
                             MetricsCallback)


def _converged_S_from_log(log_file):
    if not os.path.exists(log_file):
        return np.nan
    d = pd.read_csv(log_file)
    d2 = d[d["agent"] == "node_2"].sort_values("step")
    return d2.tail(10000)["S"].mean() if len(d2) else np.nan


def _run_single(args):
    w, seed, arm, horizon = args
    np.random.seed(seed)
    warnings.filterwarnings("ignore")
    log_file = f"/tmp/exp0b_{w}_{seed}_{arm}_{horizon}.csv"
    if os.path.exists(log_file):
        os.remove(log_file)

    # train env with S logging for converged_S
    raw = _build_env_with_log(w, seed, log_file)
    vec = ss.pad_action_space_v0(raw)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None
    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()

    if arm == "segmented":
        # reproduce Exp 0's path: learn(150k) -> (boundary) -> learn(rest)
        model.learn(total_timesteps=150000, reset_num_timesteps=True, callback=cb)
        model.learn(total_timesteps=horizon - 150000, reset_num_timesteps=False, callback=cb)
    else:  # single
        model.learn(total_timesteps=horizon, reset_num_timesteps=True, callback=cb)

    raw.close()
    converged_S = _converged_S_from_log(log_file)

    eval_env = _build_env(w, 1.0, seed)
    mean_S, _, _, _, profit, _ = evaluate_and_extract_metrics(model, eval_env, num_episodes=10, base_seed=seed * 10)
    if os.path.exists(log_file):
        os.remove(log_file)

    return {"w": w, "seed": seed, "arm": arm, "horizon": horizon,
            "collapsed": is_collapsed(mean_S, profit),
            "mean_S": mean_S, "converged_S": converged_S, "profit": profit,
            "final_value_loss": np.mean(cb.value_losses) if cb.value_losses else np.nan,
            "explained_variance": np.mean(cb.explained_variances) if cb.explained_variances else np.nan}


def _build_env_with_log(w, seed, log_file):
    """Same as expA _build_env but with S logging enabled on the scaled wrapper."""
    from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
    from env.diagnostic_wrapper import DiagnosticWrapper
    from agents.cpt_wrapper import CPTRewardWrapper
    from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=0.0, high_bound=500.0, log_file=log_file)


def build_jobs():
    jobs = []
    # Piece 1: horizon lock
    for w in (0.6, 0.7):
        for s in range(42, 62):
            jobs.append((w, s, "single", 600000))
    # Piece 2: perturbation origin (both arms), shared seeds 42-49
    for w in (0.3, 0.6, 0.7, 0.8):
        for s in range(42, 50):
            jobs.append((w, s, "single", 300000))
            jobs.append((w, s, "segmented", 300000))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--out", default="docs/exp0b_figures/exp0b_raw_data.csv")
    args = ap.parse_args()
    jobs = build_jobs()
    print(f"Exp 0b: {len(jobs)} runs (Piece1 40 @600k single; Piece2 64 @300k single+segmented), procs={args.procs}")
    os.makedirs("docs/exp0b_figures", exist_ok=True)

    rows = []
    with mp.Pool(processes=args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_single, jobs), 1):
            rows.append(r)
            print(f"[{i:3d}/{len(jobs)}] w={r['w']:.2f} seed={r['seed']:2d} {r['arm']:9s} "
                  f"{r['horizon']//1000}k -> collapsed={r['collapsed']} convS={r['converged_S']:.1f}")
            pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
