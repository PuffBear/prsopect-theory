"""
Reviewer Point 3 -- Heterogeneous cognitive profiles.

Research question: in a mixed population of risk-neutral (lambda=1) and highly
loss-averse (lambda=7) nodes, does a single irrational node propagate
instability?  Does a downstream loss-averse node cascade upstream and shift
w_crit for the whole system?

Design: sweep w x {heterogeneity config} x seed.  Heterogeneity configs assign
lambda=7 to specific nodes and lambda=1 to the rest, isolating (a) the number
of loss-averse nodes and (b) their position (downstream vs upstream).  The
per-agent CPT params are already supported by CPTRewardWrapper.

Collapse fraction vs w gives w_crit per config (same logistic estimator as the
homogeneous experiments), directly comparable to the homogeneous lambda=1 (0.586)
and lambda=7 (0.851) boundaries.
"""
import os, sys, argparse, warnings
import numpy as np, pandas as pd
from filelock import FileLock

import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from expA_interior_w import (evaluate_and_extract_metrics, is_collapsed,
                             MetricsCallback, LOW_BOUND, HIGH_BOUND)
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

W_LIST = [0.5, 0.65, 0.75, 0.85, 0.95]
# Heterogeneity configs: which nodes are loss-averse (lambda=7); rest are lambda=1.
# node ordering downstream->upstream: 1(retail,scripted) 2,3(distrib) 4,5,6(mfg)
CONFIGS = {
    "homog_LA":      ["node_2", "node_3", "node_4", "node_5", "node_6"],  # all LA (ref)
    "single_down":   ["node_2"],                  # one downstream loss-averse node
    "single_up":     ["node_6"],                  # one upstream loss-averse node
    "half_down":     ["node_2", "node_3"],        # downstream half loss-averse
}
CONFIG_NAMES = list(CONFIGS.keys())
SEEDS = list(range(42, 62))  # 20
ALPHA, BETA, LAM_LA, LAM_NEU = 0.88, 0.88, 7.0, 1.0
N_W, N_CFG, N_SEED = len(W_LIST), len(CONFIG_NAMES), len(SEEDS)
N_RUNS = N_W * N_CFG * N_SEED  # 400
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

COLUMNS = ["run_idx", "config", "w", "seed", "mean_S", "profit", "lost_sales",
           "collapsed", "final_value_loss", "final_explained_variance"]


def _build_env(config_name, w, seed):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    la_nodes = set(CONFIGS[config_name])
    params = {a: {"lambda": (LAM_LA if a in la_nodes else LAM_NEU),
                  "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def map_run_idx(run_idx):
    cfg_idx = run_idx // (N_W * N_SEED)
    rem = run_idx % (N_W * N_SEED)
    w_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return CONFIG_NAMES[cfg_idx], W_LIST[w_idx], SEEDS[seed_idx]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}]")
    cfg, w, seed = map_run_idx(run_idx)
    np.random.seed(seed); warnings.filterwarnings("ignore")

    train_env = _build_env(cfg, w, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(cfg, w, seed)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "config": cfg, "w": w, "seed": seed, "mean_S": mean_S,
           "profit": profit, "lost_sales": lost_sales, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} config={cfg} w={w} seed={seed} -> "
          f"mean_S={mean_S:.1f} profit={profit:.1f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help=f"0..{N_RUNS-1}")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="docs/expHet_figures/expHet_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
