"""
Exp QMIX/IQL-MPE -- value-based MARL (QMIX | IQL) on MPE Cooperative Navigation
(simple_spread, natively Discrete(5)) with CPT reward shaping. The cross-env
value-based point of the 2x2 (VALUE_BASED_EXPERIMENT_PLAN.md, Tiers 2/3).

    --algo qmix : centralized value-based (monotonic mixer)
    --algo iql  : independent value-based  (per-agent Q, no mixer)

Wrapper stack is IDENTICAL to exp_mappo_mpe except continuous_actions=False (so
the action space is the native Discrete(5)) -- no custom discretization:

    simple_spread (continuous_actions=False, local_ratio=0.0)
      -> MPELocalRewardWrapper(w, herding_scale)
        -> CPTRewardWrapper(alpha, beta, lambda; global_reward_weight=0.0)
          -> GlobalStateWrapper                (joint state for the mixer)

Grid uses the MPE-MAPPO recalibrated axis (plan section 1): the 11-level w-grid
and seeds inherited from exp_ppo_mpe, full lambda sweep {1..7}, so QMIX/IQL-MPE
land on the same lambda axis as MAPPO-MPE. Conditions: null (a=b=1,l=1),
curvature (a=b=0.88,l=1 = lambda=1 point of the sweep), loss aversion
(a=b=0.88,l=2..7). Collapse = occupancy < theta (is_collapsed_mpe_occupancy).

Budget: 4M env steps/run (match exp_mappo_mpe_highlambda).
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from filelock import FileLock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env.mpe_coop_nav import (make_mpe_coop_nav, MPELocalRewardWrapper,
                              is_collapsed_mpe_occupancy, HERDING_SCALE)
from env.global_state_wrapper import GlobalStateWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from agents.qmix_rollout import make_learner, train, greedy_action_fn, _split

# Inherit the MPE w-grid and seeds so QMIX/IQL-MPE share MAPPO-MPE's axis.
from exp_ppo_mpe import W_LIST  # 11 w-levels (shared MPE axis)
# Value-based MPE cells stay at 10 seeds: they are NOT the headline MPE result
# (the AAAI 20-seed request targets MAPPO/PPO-MPE), and pinning here keeps the
# already-running QMIX/IQL-MPE cluster arrays (0-769 / 0-109) consistent. Defined
# locally so the PPO-MPE 20-seed bump does NOT propagate into these arrays.
SEEDS = list(range(42, 52))  # 10 seeds (value-based MPE; intentionally not 20)

LAMBDA_LIST = [1, 2, 3, 4, 5, 6, 7]
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED          # sweep array
N_RUNS_NULL = N_W * N_SEED            # null array

ALPHA, BETA = 0.88, 0.88

COLUMNS = ["run_idx", "w", "lambda_loss", "alpha", "beta", "seed", "algo",
           "condition", "env", "mean_reward", "mean_occupancy", "collapsed",
           "final_value_loss"]


def _build_env(w, lam, alpha, beta, herding_scale=HERDING_SCALE):
    raw = make_mpe_coop_nav(n_agents=3, max_cycles=50, continuous_actions=False)
    local_w = MPELocalRewardWrapper(raw, w=w, herding_scale=herding_scale)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in local_w.possible_agents}
    cpt = CPTRewardWrapper(local_w, params, reward_scale=1.0,
                           global_reward_weight=0.0)
    return GlobalStateWrapper(cpt)


def evaluate_mpe_discrete(learner, eval_env, num_episodes=10, base_seed=42):
    """Greedy rollout -> (mean true r_global per episode, mean occupancy)."""
    local_dim = learner.local_dim
    act_fn = greedy_action_fn(learner, local_dim)
    ep_rewards, ep_occ = [], []
    for ep in range(num_episodes):
        obs, _ = eval_env.reset(seed=base_seed + ep)
        r_global_sum = 0.0
        occ_steps = []
        while eval_env.agents:
            actions = {a: act_fn(obs[a]) for a in eval_env.agents if a in obs}
            obs, rewards, term, trunc, info = eval_env.step(actions)
            any_agent = next(iter(rewards))
            r_global_sum += info[any_agent]["r_global"]
            occ_steps.append(info[any_agent]["occupancy"])
        ep_rewards.append(r_global_sum)
        ep_occ.append(np.nanmean(occ_steps) if occ_steps else float("nan"))
    return float(np.mean(ep_rewards)), float(np.nanmean(ep_occ))


def map_sweep_idx(run_idx):
    lam_idx = run_idx // (N_W * N_SEED)
    rem = run_idx % (N_W * N_SEED)
    w_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], LAMBDA_LIST[lam_idx], SEEDS[seed_idx]


def map_null_idx(run_idx):
    w_idx = run_idx // N_SEED
    seed_idx = run_idx % N_SEED
    return W_LIST[w_idx], SEEDS[seed_idx]


def run_one(run_idx, algo, null, timesteps, out_path,
            herding_scale=HERDING_SCALE, verbose=False):
    use_mixer = (algo == "qmix")
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

    env = _build_env(w, lam, alpha, beta, herding_scale=herding_scale)
    learner = make_learner(env, use_mixer=use_mixer, seed=seed)
    train(env, learner, total_env_steps=timesteps, seed=seed, verbose=verbose)

    eval_env = _build_env(w, lam, alpha, beta, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe_discrete(learner, eval_env,
                                                  num_episodes=10, base_seed=seed * 10)
    collapsed = is_collapsed_mpe_occupancy(mean_occ)

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "alpha": alpha,
           "beta": beta, "seed": seed, "algo": algo.upper(), "condition": cond,
           "env": "MPE_CoopNav", "mean_reward": mean_reward,
           "mean_occupancy": mean_occ, "collapsed": collapsed,
           "final_value_loss": learner.last_value_loss}
    _append_row(row, out_path)
    print(f"[{algo.upper()}-MPE{'-null' if null else ''}] run_idx={run_idx} w={w} "
          f"lambda={lam} seed={seed} cond={cond} -> r_global={mean_reward:.1f} "
          f"occ={mean_occ:.2f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def wiring_check(algo):
    use_mixer = (algo == "qmix")
    env = _build_env(0.5, 7, ALPHA, BETA)
    agents = list(env.possible_agents)
    for a in agents:
        sp = env.action_space(a)
        assert sp.__class__.__name__ == "Discrete" and sp.n == 5, \
            f"action space for {a} is {sp}, expected Discrete(5)"
    print(f"  [1] action space = Discrete(5) per agent for all {len(agents)} agents  OK")

    obs, _ = env.reset(seed=0)
    act = {a: 0 for a in env.agents}
    _o, rewards, _t, _tr, info = env.step(act)
    a0 = agents[0]
    blended = info[a0]["blended_reward"]; cpt = info[a0]["cpt_reward"]
    expect = blended ** ALPHA if blended >= 0 else -7 * ((-blended) ** BETA)
    # note: global_reward_weight=0.0 here, so blended == r_blend from MPELocalReward
    assert abs(rewards[a0] - cpt) < 1e-9, "buffer reward != cpt_reward"
    print(f"  [2] one transition: blended={blended:.4f} cpt={cpt:.4f} "
          f"(reward written == cpt)  OK")

    learner = make_learner(env, use_mixer=use_mixer, seed=42)
    local_dim = env.local_dim
    aug_dim = int(np.prod(env.observation_space(a0).shape))
    state_dim = aug_dim - local_dim
    if algo == "qmix":
        assert learner.mixer is not None, "QMIX has NO mixer (silently IQL)!"
        assert learner.mixer.hyper_w1.in_features == state_dim, "mixer != global slice"
        qs = torch.zeros(1, learner.n_agents, requires_grad=True)
        g, = torch.autograd.grad(learner.mixer(qs, torch.randn(1, state_dim)).sum(), qs)
        assert (g >= -1e-6).all(), f"mixer not monotone: {g.tolist()}"
        print(f"  [3] QMIX mixer reads global state (dim={state_dim}); monotone  OK")
    else:
        assert learner.mixer is None, "IQL has a mixer!"
        print(f"  [3] IQL has NO mixer; Q_tot = sum_i Q_i  OK")
    print("  WIRING CHECK PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["qmix", "iql"], required=True)
    ap.add_argument("--run_idx", type=int, default=None)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--timesteps", type=int, default=4_000_000)
    ap.add_argument("--herding_scale", type=float, default=HERDING_SCALE)
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.wiring_check:
        wiring_check(args.algo); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")

    out = args.out or (f"docs/expVB_{args.algo}_mpe_figures/"
                       f"{args.algo}_mpe_raw_data.csv")
    run_one(args.run_idx, args.algo, args.null, args.timesteps, out,
            herding_scale=args.herding_scale, verbose=args.verbose)


if __name__ == "__main__":
    main()
