"""
Experiment A — Interior-w sweep + logistic collapse model (experiment_spec_v1 sec.A).

Sweeps reward centralization w = global_reward_weight across its interior on the
learnable base-stock substrate, lambda fixed at 1.0 (no loss aversion). One row
per run, with the PRE-REGISTERED collapse label (docs/collapse_definition.md).

Mean_S is computed identically to phase4_pt_replication.py so the collapse label
is consistent with the Day-0 calibration histogram.

CLI:
    python experiments/expA_interior_w.py --ws 0,0.1,...,1.0 --seeds 42-61 \
        --timesteps 150000 --procs 10 --out docs/expA_figures/expA_raw_data.csv
A smoke test:
    python experiments/expA_interior_w.py --ws 0.0,1.0 --seeds 42-42 \
        --timesteps 5000 --procs 2 --out /tmp/expA_smoke.csv
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
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

# --- Frozen collapse definition (docs/collapse_definition.md) ---
S_FLOOR_THRESHOLD = 10.0
PROFIT_FLOOR = -128.1
EPSILON = 1.0

LOW_BOUND, HIGH_BOUND = 0.0, 500.0


def is_collapsed(mean_S, profit):
    return int((mean_S < S_FLOOR_THRESHOLD) and (profit <= PROFIT_FLOOR + EPSILON))


class MetricsCallback(BaseCallback):
    """Records PPO value loss and explained variance per rollout (from phase3_9)."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.value_losses = []
        self.explained_variances = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is not None:
            nv = self.logger.name_to_value
            if "train/value_loss" in nv:
                self.value_losses.append(nv["train/value_loss"])
            if "train/explained_variance" in nv:
                self.explained_variances.append(nv["train/explained_variance"])


def evaluate_and_extract_metrics(model, eval_env, num_episodes=10, base_seed=42):
    """Mirrors phase4_pt_replication.evaluate_and_extract_metrics (mean_S included)."""
    total_order_qty, total_inv, total_lost_sales, total_reward, bw_ratios = [], [], [], [], []

    for ep in range(num_episodes):
        obs, _ = eval_env.reset(seed=base_seed + ep)
        ep_rew = 0.0
        ep_orders, ep_demand = [], []

        while eval_env.agents:
            actions = {}
            for agent in eval_env.agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    real_shape = eval_env.action_space(agent).shape[0]
                    actions[agent] = action[:real_shape]
            obs, rewards, term, trunc, info = eval_env.step(actions)

            curr_env = eval_env
            while hasattr(curr_env, 'env') or hasattr(curr_env, 'unwrapped_env'):
                if hasattr(curr_env, 'unwrapped_env'):
                    base_env = curr_env.unwrapped_env
                    break
                curr_env = curr_env.env

            t = base_env.period - 1
            step_ppo_orders = []
            for agent in eval_env.possible_agents:
                node_id = eval_env.unwrapped.agent_name_to_id[agent]
                for link in base_env.reorder_links:
                    if link[1] == node_id:
                        step_ppo_orders.append(base_env.R.loc[t, link])
            ep_orders.append(np.mean(step_ppo_orders))
            ep_demand.append(base_env.D.loc[t, :].sum())
            for agent in rewards:
                ep_rew += info[agent]["true_reward"]
            total_order_qty.extend(step_ppo_orders)

        total_inv.append(base_env.X.mean().mean())
        total_D = base_env.D.sum().sum()
        total_U = base_env.U.sum().sum()
        total_lost_sales.append(total_U / total_D if total_D > 0 else 0)
        total_reward.append(ep_rew)
        bw_ratios.append(np.var(ep_orders) / np.var(ep_demand) if np.var(ep_demand) > 0 else 0)

    # Mean_S the phase4 way: single deterministic action at reset, mapped to [0,500]
    obs, _ = eval_env.reset(seed=base_seed)
    s_vals = []
    for agent in eval_env.agents:
        if agent in obs:
            action, _ = model.predict(obs[agent], deterministic=True)
            S = LOW_BOUND + 0.5 * (action[0] + 1.0) * (HIGH_BOUND - LOW_BOUND)
            s_vals.append(S)

    mean_order = np.mean(total_order_qty) if total_order_qty else 0.0
    return (np.mean(s_vals), mean_order, np.mean(total_inv),
            np.mean(total_lost_sales), np.mean(total_reward), np.mean(bw_ratios))


def _build_env(w, lam, seed):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": lam, "alpha": 1.0, "beta": 1.0} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def _run_single(args):
    w, lam, seed, timesteps = args
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    train_env = _build_env(w, lam, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, seed)
    mean_S, mean_order, mean_inv, mean_ls, profit, bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)

    return {
        "w": w, "lambda": lam, "seed": seed,
        "mean_S": mean_S, "profit": profit, "lost_sales": mean_ls,
        "bullwhip": bw, "mean_order": mean_order, "mean_inventory": mean_inv,
        "collapsed": is_collapsed(mean_S, profit),
        "final_value_loss": np.mean(cb.value_losses) if cb.value_losses else np.nan,
        "final_explained_variance": np.mean(cb.explained_variances) if cb.explained_variances else np.nan,
        "train_steps": timesteps,
    }


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", default="42-61")  # 20 seeds
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--procs", type=int, default=min(mp.cpu_count(), 10))
    ap.add_argument("--out", default="docs/expA_figures/expA_raw_data.csv")
    args = ap.parse_args()

    ws = [float(x) for x in args.ws.split(",")]
    seeds = parse_seeds(args.seeds)
    jobs = [(w, args.lam, s, args.timesteps) for w in ws for s in seeds]

    print(f"Experiment A: {len(ws)} w-levels x {len(seeds)} seeds = {len(jobs)} runs "
          f"@ {args.timesteps} steps, procs={args.procs}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    with mp.Pool(processes=args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_single, jobs), 1):
            rows.append(r)
            print(f"[{i:3d}/{len(jobs)}] w={r['w']:.2f} seed={r['seed']:2d} -> "
                  f"S={r['mean_S']:6.1f} profit={r['profit']:7.1f} collapsed={r['collapsed']}")
            pd.DataFrame(rows).to_csv(args.out, index=False)  # incremental save

    df = pd.DataFrame(rows).sort_values(["w", "seed"])
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")
    print("\nCollapse fraction by w:")
    print(df.groupby("w")["collapsed"].agg(["mean", "sum", "count"]).to_string())


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
