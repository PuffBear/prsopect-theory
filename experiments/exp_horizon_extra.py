"""
Reviewer Point 2 -- Additional matched horizons for the metastable slide.

Research question: the T->infinity asymptote and the (approx) -ln T trend of
w_crit currently rest on two clean points (150k -> 0.822, 300k -> 0.666). Adding
450k and 900k at MATCHED alpha=beta=1 (risk-neutral, no curvature confound)
converts the asserted trend into a fitted one.

Design: structural condition only (alpha=beta=1, lambda=1), interior-w grid,
20 seeds, at two new horizons {450k, 900k}. Same substrate, optimizer, collapse
label, and estimator as Experiment A, so the new w_crit values are directly
comparable to the existing 150k/300k boundaries and feed the same Kramers /
-ln T fit.

Output schema mirrors expA so analyze_* and the logistic/bootstrap estimator
ingest it unchanged. Array layout: horizon x w x seed.
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

# --- Design -----------------------------------------------------------------
# Structural condition (matched alpha=beta=1, lambda=1) at two new horizons.
HORIZONS = [450_000, 900_000]
# Interior-w grid focused where the 150k/300k boundaries sit (~0.55-0.85),
# so the crossing is well resolved at the longer horizons (where w_crit drops).
W_LIST = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SEEDS = list(range(42, 62))  # 20
ALPHA, BETA, LAM = 1.0, 1.0, 1.0

N_H, N_W, N_SEED = len(HORIZONS), len(W_LIST), len(SEEDS)
N_RUNS = N_H * N_W * N_SEED  # 2 x 8 x 20 = 320
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

COLUMNS = ["run_idx", "horizon", "w", "lambda_loss", "alpha", "beta", "seed",
           "mean_S", "profit", "lost_sales", "collapsed",
           "final_value_loss", "final_explained_variance"]


def _build_env(w, seed):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": LAM, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def map_run_idx(run_idx):
    h_idx = run_idx // (N_W * N_SEED)
    rem = run_idx % (N_W * N_SEED)
    w_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return HORIZONS[h_idx], W_LIST[w_idx], SEEDS[seed_idx]


def run_one(run_idx, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    horizon, w, seed = map_run_idx(run_idx)
    np.random.seed(seed); warnings.filterwarnings("ignore")

    train_env = _build_env(w, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=horizon, callback=cb)

    eval_env = _build_env(w, seed)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "horizon": horizon, "w": w, "lambda_loss": LAM,
           "alpha": ALPHA, "beta": BETA, "seed": seed, "mean_S": mean_S,
           "profit": profit, "lost_sales": lost_sales, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} horizon={horizon} w={w} seed={seed} -> "
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
    ap.add_argument("--out", default="docs/expHorizon_extra_figures/horizon_extra_raw_data.csv")
    ap.add_argument("--smoke", action="store_true",
                    help="20k-step sanity run (overrides the mapped horizon)")
    args = ap.parse_args()
    if args.smoke:
        # Cheap wiring check: one run at 20k regardless of mapped horizon.
        global HORIZONS
        h, w, seed = map_run_idx(args.run_idx)
        print(f"[SMOKE] would train horizon={h} at w={w}, seed={seed}; running 20k instead")
        HORIZONS = [20_000, 20_000]
    run_one(args.run_idx, args.out)


if __name__ == "__main__":
    main()
