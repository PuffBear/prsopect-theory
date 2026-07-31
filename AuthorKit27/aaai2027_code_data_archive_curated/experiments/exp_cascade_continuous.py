"""
Continuous-lambda and node-position cascade.

The headline cascade result uses binary lambda in {1,7} on four configurations
(single up / single down / two down / all). This experiment extends it on two
axes WITHOUT a new substrate:

  (A) NODE-POSITION sweep at fixed lambda=7. Place a single loss-averse node at
      every position along the directed chain (downstream -> upstream), to trace
      w_crit as a continuous function of network position and locate where the
      cascade transitions from "sets the whole boundary" to "barely moves it".

  (B) CONTINUOUS-lambda sweep at a fixed DOWNSTREAM node. Vary the single
      downstream node's lambda over {1,2,3,4,5,7}, the rest risk-neutral, to show
      how much bias at the critical position is needed to drag the system boundary
      (the cascade as a function of bias strength, not just its presence).

Node ordering downstream->upstream: node_1 (retail, scripted) | node_2, node_3
(distribution) | node_4, node_5, node_6 (manufacturing). node_1 is scripted, so
the trainable positions are node_2 ... node_6.

Same substrate, optimizer, CPT machinery (per-agent lambda via CPTRewardWrapper),
collapse label, grid, and 20 seeds as exp_heterogeneous. Output schema extends
expHet with explicit 'la_node' and 'la_lambda' columns so position and bias are
first-class. Array layout: arm (A|B) cells x w x seed.
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
W_LIST = [0.5, 0.65, 0.75, 0.85, 0.95]
SEEDS = list(range(42, 62))  # 20
ALPHA, BETA, LAM_NEU = 0.88, 0.88, 1.0

# Trainable positions, ordered downstream (near retail) -> upstream (manufacturing).
NODES_DOWN_TO_UP = ["node_2", "node_3", "node_4", "node_5", "node_6"]

# Arm A: single loss-averse (lambda=7) node at each position. (la_node, la_lambda)
ARM_A = [(n, 7.0) for n in NODES_DOWN_TO_UP]                       # 5 cells
# Arm B: fixed downstream node (node_2), continuous lambda.
ARM_B = [("node_2", lam) for lam in (1.0, 2.0, 3.0, 4.0, 5.0, 7.0)]  # 6 cells
# (node_2 @ lambda 1 and 7 overlap with existing single_down points -- kept for a
#  self-contained, internally consistent curve.)

CELLS = ARM_A + ARM_B                       # 11 distinct (node, lambda) cells
CELL_TAGS = ([f"posA_{n}" for n, _ in ARM_A] +
             [f"lamB_{n}_L{int(l)}" for n, l in ARM_B])

N_CELL, N_W, N_SEED = len(CELLS), len(W_LIST), len(SEEDS)
N_RUNS = N_CELL * N_W * N_SEED              # 11 x 5 x 20 = 1100
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

COLUMNS = ["run_idx", "cell_tag", "la_node", "la_lambda", "w", "seed",
           "mean_S", "profit", "lost_sales", "collapsed",
           "final_value_loss", "final_explained_variance"]


def _build_env(la_node, la_lambda, w, seed):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": (la_lambda if a == la_node else LAM_NEU),
                  "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def map_run_idx(run_idx):
    cell_idx = run_idx // (N_W * N_SEED)
    rem = run_idx % (N_W * N_SEED)
    w_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    (la_node, la_lambda) = CELLS[cell_idx]
    return CELL_TAGS[cell_idx], la_node, la_lambda, W_LIST[w_idx], SEEDS[seed_idx]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    tag, la_node, la_lambda, w, seed = map_run_idx(run_idx)
    np.random.seed(seed); warnings.filterwarnings("ignore")

    train_env = _build_env(la_node, la_lambda, w, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(la_node, la_lambda, w, seed)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "cell_tag": tag, "la_node": la_node,
           "la_lambda": la_lambda, "w": w, "seed": seed, "mean_S": mean_S,
           "profit": profit, "lost_sales": lost_sales, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} cell={tag} node={la_node} L={la_lambda} w={w} "
          f"seed={seed} -> mean_S={mean_S:.1f} profit={profit:.1f} collapsed={collapsed}")
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
    ap.add_argument("--out",
                    default="docs/expCascade_continuous_figures/cascade_continuous_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
