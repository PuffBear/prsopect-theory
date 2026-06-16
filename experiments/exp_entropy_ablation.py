"""
Reviewer Point 4 -- Exploration noise / representational dynamics ablation.

Research question: is the location of w_crit driven by the behavioral bias
(lambda), or primarily by exploration noise overwhelming the diluted reward
signal at low centralization?  The mean-field model (Sec. Mean-Field Analysis)
predicts the boundary depends on the drift-to-noise ratio, where the noise
scales with the PPO entropy/exploration term -- so varying ent_coef should move
w_crit even at fixed lambda.

Design: sweep w x ent_coef x seed at fixed lambda.  Estimating w_crit(ent_coef)
tests the prediction directly:  if w_crit rises with ent_coef, the boundary is
(partly) an exploration-noise phenomenon, not purely a behavioral one.
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

W_LIST = [0.5, 0.6, 0.7, 0.8, 0.9]
ENT_COEFS = [0.0, 0.001, 0.01, 0.05]   # PPO entropy coefficient (exploration noise)
SEEDS = list(range(42, 62))            # 20
ALPHA, BETA, LAM = 1.0, 1.0, 1.0       # structural condition (no PT) to isolate noise
N_W, N_ENT, N_SEED = len(W_LIST), len(ENT_COEFS), len(SEEDS)
N_RUNS = N_W * N_ENT * N_SEED          # 400
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

COLUMNS = ["run_idx", "ent_coef", "w", "seed", "mean_S", "profit", "lost_sales",
           "collapsed", "final_value_loss", "final_explained_variance"]


def _build_env(w, seed):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": LAM, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def map_run_idx(run_idx):
    ent_idx = run_idx // (N_W * N_SEED)
    rem = run_idx % (N_W * N_SEED)
    w_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return ENT_COEFS[ent_idx], W_LIST[w_idx], SEEDS[seed_idx]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}]")
    ent, w, seed = map_run_idx(run_idx)
    np.random.seed(seed); warnings.filterwarnings("ignore")

    train_env = _build_env(w, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256,
                ent_coef=ent, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, seed)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "ent_coef": ent, "w": w, "seed": seed, "mean_S": mean_S,
           "profit": profit, "lost_sales": lost_sales, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} ent_coef={ent} w={w} seed={seed} -> "
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
    ap.add_argument("--out", default="docs/expEnt_figures/expEnt_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
