"""
Discretized-PPO control on or-gym supply chain (VALUE_BASED_EXPERIMENT_PLAN sec 7,
the Tier-3-optional control).

WHY THIS EXISTS
---------------
QMIX and IQL on the discretized supply chain BOTH produced a w-INDEPENDENT
collapse rate (no boundary) -- and identically, so the cause is shared, not the
mixer. Hypothesis: discretizing the continuous order-quantity action + taking a
discrete argmax erases the loss-aversion boundary, which is a property of the
CONTINUOUS policy-gradient reward surface (the w-blend moves the optimum's VALUE,
not WHICH of the K coarse bins is optimal).

This control isolates paradigm vs action-space: run PPO -- the SAME learner that
DOES show the boundary with a continuous action -- over the EXACT SAME Discrete(K)
shell QMIX/IQL used. If discretized-PPO ALSO goes flat, the cause is provably the
ACTION-SPACE DISCRETIZATION, not the value-based paradigm. If discretized-PPO
keeps the boundary, the discretization is innocent and the value-based paradigm
is what erases it.

Held identical to exp_qmix_orgym: same DiscreteBaseStockWrapper(K) over the same
CPT + ScaledBaseStock stack, same w/lambda grid, seeds, collapse label, and the
same discrete-greedy eval (index -> mean_S). The ONLY change vs QMIX/IQL is the
learner: SB3 PPO with a categorical actor over Discrete(K) (no mixer, no replay,
on-policy) -- i.e. policy gradient on the discretized action.

    python experiments/exp_ppo_disc_orgym.py --wiring_check
    python experiments/exp_ppo_disc_orgym.py --run_idx 0 --timesteps 20000 --out /tmp/s.csv
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from filelock import FileLock

import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from expA_interior_w import LOW_BOUND, HIGH_BOUND, MetricsCallback
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper
from env.discrete_action_wrapper import DiscreteBaseStockWrapper

# Same grid / conditions / label as exp_qmix_orgym (so the control is comparable).
from exp_qmix_orgym import (W_LIST, LAMBDA_LIST, SEEDS, N_W, N_LAM, N_SEED,
                            N_RUNS, N_RUNS_NULL, ALPHA, BETA, S_FLOOR, PROFIT_CEIL,
                            DEFAULT_K, map_sweep_idx, map_null_idx)

PPO_KWARGS = dict(n_steps=128, batch_size=256)  # same as the continuous PPO runs

COLUMNS = ["run_idx", "w", "lambda_loss", "alpha", "beta", "seed", "algo",
           "condition", "mean_S", "profit", "lost_sales", "collapsed",
           "final_value_loss", "K"]


def _build_env(w, lam, alpha, beta, K=DEFAULT_K):
    """SAME discrete stack as exp_qmix_orgym MINUS GlobalStateWrapper (PPO here is
    the independent policy-gradient control; no centralized critic / joint state)."""
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in interv.possible_agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    scaled = ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)
    return DiscreteBaseStockWrapper(scaled, K=K)


def _build_vec(w, lam, alpha, beta, K, seed):
    env = _build_env(w, lam, alpha, beta, K=K)
    vec = ss.pad_action_space_v0(env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None
    return vec


def evaluate_ppo_disc(model, eval_env, K, num_episodes=10, base_seed=42):
    """Discrete-greedy eval identical in spirit to exp_qmix_orgym.evaluate_supply:
    map the model's categorical argmax (a bin index) -> mean_S and extract
    profit / lost-sales from the base env."""
    norm = eval_env._norm  # DiscreteBaseStockWrapper: index -> [-1,1]
    total_reward, total_lost = [], []
    for ep in range(num_episodes):
        obs, _ = eval_env.reset(seed=base_seed + ep)
        ep_rew = 0.0
        while eval_env.agents:
            actions = {}
            for a in eval_env.agents:
                if a in obs:
                    act, _ = model.predict(obs[a], deterministic=True)
                    actions[a] = int(np.asarray(act).item())
            obs, rewards, term, trunc, info = eval_env.step(actions)
            for a in rewards:
                ep_rew += info[a]["true_reward"]
        base_env = eval_env.unwrapped.unwrapped_env
        total_D = base_env.D.sum().sum(); total_U = base_env.U.sum().sum()
        total_lost.append(total_U / total_D if total_D > 0 else 0.0)
        total_reward.append(ep_rew)

    obs, _ = eval_env.reset(seed=base_seed)
    s_vals = []
    for a in eval_env.agents:
        if a in obs:
            act, _ = model.predict(obs[a], deterministic=True)
            k = int(np.asarray(act).item())
            S = LOW_BOUND + 0.5 * (norm[k] + 1.0) * (HIGH_BOUND - LOW_BOUND)
            s_vals.append(S)
    return float(np.mean(s_vals)), float(np.mean(total_reward)), float(np.mean(total_lost))


def run_one(run_idx, null, timesteps, K, out_path):
    if null:
        if not (0 <= run_idx < N_RUNS_NULL):
            raise ValueError(f"null run_idx must be in [0,{N_RUNS_NULL-1}]")
        w, seed = map_null_idx(run_idx)
        lam, alpha, beta, cond = 1.0, 1.0, 1.0, "null"
    else:
        if not (0 <= run_idx < N_RUNS):
            raise ValueError(f"sweep run_idx must be in [0,{N_RUNS-1}]")
        w, lam, seed = map_sweep_idx(run_idx)
        alpha, beta = ALPHA, BETA
        cond = "curvature" if lam == 1 else "loss_aversion"

    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    vec = _build_vec(w, lam, alpha, beta, K, seed)
    model = PPO("MlpPolicy", vec, verbose=0, seed=seed, **PPO_KWARGS)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, alpha, beta, K=K)
    mean_S, profit, lost = evaluate_ppo_disc(model, eval_env, K,
                                             num_episodes=10, base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "alpha": alpha,
           "beta": beta, "seed": seed, "algo": "PPO_DISC", "condition": cond,
           "mean_S": mean_S, "profit": profit, "lost_sales": lost,
           "collapsed": collapsed, "final_value_loss": fvl, "K": K}
    _append_row(row, out_path)
    print(f"[PPO_DISC-OG{'-null' if null else ''}] run_idx={run_idx} w={w} "
          f"lambda={lam} seed={seed} cond={cond} -> mean_S={mean_S:.1f} "
          f"profit={profit:.1f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def wiring_check(K=DEFAULT_K):
    """Assert PPO sees Discrete(K) and the categorical actor is built over it."""
    vec = _build_vec(0.75, 7, ALPHA, BETA, K, seed=42)
    sp = vec.action_space
    assert sp.__class__.__name__ == "Discrete" and sp.n == K, \
        f"vec action space is {sp}, expected Discrete({K})"
    model = PPO("MlpPolicy", vec, seed=42, **PPO_KWARGS)
    dist = model.policy.action_dist.__class__.__name__
    assert "Categorical" in dist, f"actor dist is {dist}, expected Categorical (discrete)"
    print(f"  action space PPO sees = Discrete({K}); actor dist = {dist}")
    print("  WIRING CHECK PASS: PPO is policy-gradient over the SAME Discrete(K) shell.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.wiring_check:
        wiring_check(K=args.K); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    out = args.out or "docs/expVB_ppodisc_orgym_figures/ppodisc_orgym_raw_data.csv"
    run_one(args.run_idx, args.null, args.timesteps, args.K, out)


if __name__ == "__main__":
    main()
