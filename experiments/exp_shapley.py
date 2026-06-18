"""
Reviewer Point 2 -- Structural credit assignment arm.

Research question: does Shapley / difference-reward credit assignment make the
system robust to loss aversion?  If an agent receives credit that isolates its
marginal contribution to system profit, does the collapse boundary vanish, or
does loss aversion (lambda=7) still force a lazy-agent state?

Design: replaces the scalar-w CPT blend with ShapleyRewardWrapper.  We cross
{credit mode} x {lambda} x {seed} and measure the collapse fraction.  Because
the credit method has no w, a vanishing collapse fraction (vs the scalar-w
baseline's collapse below w_crit) is the positive result.

SLURM array job: one task per run_idx, appends one row under a filelock.
Mirrors exp_b_reduced.py exactly (same eval / collapse label / substrate).
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
from env.shapley_reward_wrapper import ShapleyRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

MODES = ["difference", "shapley"]      # structural credit methods
LAMBDA_LIST = [1, 7]                    # no-PT vs strong loss aversion
SEEDS = list(range(42, 62))            # 20
ALPHA, BETA = 0.88, 0.88
N_MODE, N_LAM, N_SEED = len(MODES), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_MODE * N_LAM * N_SEED        # 80
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

COLUMNS = ["run_idx", "credit_mode", "lambda_loss", "seed", "mean_S", "profit",
           "lost_sales", "collapsed", "final_value_loss", "final_explained_variance"]


def _build_env(mode, lam, seed):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    cpt_params = {a: {"lambda": lam, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    shap = ShapleyRewardWrapper(interv, mode=mode, n_perm=8, default_order=0.0,
                                cpt_params=cpt_params, reward_scale=1.0, seed=seed,
                                baseline="mean")
    return ScaledBaseStockWrapper(shap, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def map_run_idx(run_idx):
    mode_idx = run_idx // (N_LAM * N_SEED)
    rem = run_idx % (N_LAM * N_SEED)
    lam_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return MODES[mode_idx], LAMBDA_LIST[lam_idx], SEEDS[seed_idx]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}]")
    mode, lam, seed = map_run_idx(run_idx)
    np.random.seed(seed); warnings.filterwarnings("ignore")

    train_env = _build_env(mode, lam, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None
    # Shapley/difference credit has high variance and large scale (counterfactual
    # stockout cascades early in training), which blows up the PPO critic
    # (explained_variance ~ -40, universal collapse). Normalise + clip the TRAINING
    # reward so the value head can learn. Eval below uses the raw env, so the
    # collapse label is still computed on true profit, unaffected by this.
    from stable_baselines3.common.vec_env import VecNormalize
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0,
                       gamma=0.99)

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(mode, lam, seed)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "credit_mode": mode, "lambda_loss": lam, "seed": seed,
           "mean_S": mean_S, "profit": profit, "lost_sales": lost_sales,
           "collapsed": collapsed, "final_value_loss": fvl,
           "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} mode={mode} lambda={lam} seed={seed} -> "
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
    ap.add_argument("--out", default="docs/expShapley_figures/expShapley_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
