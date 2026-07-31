"""
Exp QMIX/IQL-OG -- value-based MARL (QMIX | IQL) on or-gym NetworkManagement-v1
with CPT reward shaping. The VALUE-BASED half of the credit-assignment x
learning-paradigm 2x2 (see VALUE_BASED_EXPERIMENT_PLAN.md).

    --algo qmix  : centralized value-based (monotonic mixing network)
    --algo iql   : independent value-based  (per-agent Q, no mixer)

Everything except the learner is held byte-identical to exp_mappo_orgym /
exp_b_reduced: same CPT wrapper + order, same w/lambda grid, same seeds, same
frozen collapse label. The ONLY new pieces are (a) the Discrete(K) action shell
(value-based learners need a discrete action space) and (b) the learner itself.

WRAPPER STACK (OUTER -> INNER), plan section 3:
    GlobalStateWrapper                       (joint state for the QMIX mixer)
      -> DiscreteBaseStockWrapper(K)         index -> normalized [-1,1] base-stock
        -> CPTRewardWrapper(blend w, CPT)
          -> ScaledBaseStockWrapper          [-1,1] -> S in [0,500] -> order qty
            -> DiagnosticWrapper(node_1)
              -> MultiAgentNetInvMgmt (NetworkManagement-v1)

CONDITIONS (plan section 2): each cell carries its own re-anchored baseline.
  - Null         : alpha=beta=1, lambda=1            ->  5 w x 20 seeds = 100  (null array)
  - Curvature    : alpha=beta=0.88, lambda=1         ->  == lambda=1 point of the sweep
  - Loss aversion: alpha=beta=0.88, lambda=2..7      ->  5 w x 7 lambda x 20 = 700 (sweep array)
The sweep array (0..699) CONTAINS the curvature point (lambda=1); the null array
(0..99) re-anchors the learner honestly. Per-cell total = 800 runs.

>>> BEFORE CLUSTER TIME run, in order:
      python experiments/exp_qmix_orgym.py --algo qmix --wiring_check
      python experiments/run_qmix_toy_sanity.py            # known-good QMIX check
      python experiments/run_qmix_smoke_local.py --algo qmix --env supply
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from filelock import FileLock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from expA_interior_w import LOW_BOUND, HIGH_BOUND
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper
from env.discrete_action_wrapper import DiscreteBaseStockWrapper
from env.global_state_wrapper import GlobalStateWrapper
from agents.qmix_rollout import make_learner, train, greedy_action_fn, _split

# --- Grid (identical to exp_mappo_orgym / exp_b_reduced) ---------------------
W_LIST = [0.5, 0.65, 0.75, 0.85, 0.95]
LAMBDA_LIST = [1, 2, 3, 4, 5, 6, 7]      # sweep (curvature = lambda=1 point)
SEEDS = list(range(42, 62))              # 20
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED            # 700 (sweep array)
N_RUNS_NULL = N_W * N_SEED              # 100 (null array)

ALPHA, BETA = 0.88, 0.88
S_FLOOR, PROFIT_CEIL = 10.0, -127.1     # frozen collapse label (env-outcome)
DEFAULT_K = 11

COLUMNS = ["run_idx", "w", "lambda_loss", "alpha", "beta", "seed", "algo",
           "condition", "mean_S", "profit", "lost_sales", "collapsed",
           "final_value_loss", "K"]


# --- Env builder (GlobalStateWrapper OUTERMOST so the mixer sees joint state) -
def _build_env(w, lam, alpha, beta, K=DEFAULT_K):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in interv.possible_agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    scaled = ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)
    disc = DiscreteBaseStockWrapper(scaled, K=K)
    return GlobalStateWrapper(disc)


# --- Discrete eval mirroring evaluate_and_extract_metrics (mean_S, profit) ---
def evaluate_supply(learner, eval_env, num_episodes=10, base_seed=42):
    """Greedy rollout; extract mean_S (mapped base-stock), profit (true reward),
    lost-sales ratio -- the same metrics exp_mappo_orgym logs, computed for a
    discrete learner."""
    local_dim = learner.local_dim
    act_fn = greedy_action_fn(learner, local_dim)
    levels = eval_env.unwrapped  # walk to DiscreteBaseStockWrapper for S mapping
    # find the discrete wrapper to map index -> S
    disc = eval_env
    while disc is not None and not isinstance(disc, DiscreteBaseStockWrapper):
        disc = getattr(disc, "env", None)
    norm = disc._norm  # [-1,1] per index

    total_reward, total_inv, total_lost = [], [], []
    s_record = []
    for ep in range(num_episodes):
        obs, _ = eval_env.reset(seed=base_seed + ep)
        ep_rew = 0.0
        while eval_env.agents:
            actions = {a: act_fn(obs[a]) for a in eval_env.agents if a in obs}
            obs, rewards, term, trunc, info = eval_env.step(actions)
            for a in rewards:
                ep_rew += info[a]["true_reward"]
        # base env metrics
        base_env = None
        cur = eval_env
        while cur is not None:
            if hasattr(cur, "unwrapped_env"):
                base_env = cur.unwrapped_env; break
            cur = getattr(cur, "env", None)
        total_inv.append(base_env.X.mean().mean())
        total_D = base_env.D.sum().sum(); total_U = base_env.U.sum().sum()
        total_lost.append(total_U / total_D if total_D > 0 else 0.0)
        total_reward.append(ep_rew)

    # mean_S: greedy action at reset mapped to [0,500] via the same path as PPO
    obs, _ = eval_env.reset(seed=base_seed)
    for a in eval_env.agents:
        if a in obs:
            k = act_fn(obs[a])
            S = LOW_BOUND + 0.5 * (norm[k] + 1.0) * (HIGH_BOUND - LOW_BOUND)
            s_record.append(S)
    return (float(np.mean(s_record)), float(np.mean(total_reward)),
            float(np.mean(total_lost)))


# --- run_idx mapping (sweep: w x lambda x seed; null: w x seed) --------------
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


def run_one(run_idx, algo, null, timesteps, K, out_path, verbose=False):
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

    env = _build_env(w, lam, alpha, beta, K=K)
    learner = make_learner(env, use_mixer=use_mixer, seed=seed)
    train(env, learner, total_env_steps=timesteps, seed=seed, verbose=verbose)

    eval_env = _build_env(w, lam, alpha, beta, K=K)
    mean_S, profit, lost = evaluate_supply(learner, eval_env, num_episodes=10,
                                           base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "alpha": alpha,
           "beta": beta, "seed": seed, "algo": algo.upper(), "condition": cond,
           "mean_S": mean_S, "profit": profit, "lost_sales": lost,
           "collapsed": collapsed, "final_value_loss": learner.last_value_loss,
           "K": K}
    _append_row(row, out_path)
    print(f"[{algo.upper()}-OG{'-null' if null else ''}] run_idx={run_idx} w={w} "
          f"lambda={lam} seed={seed} cond={cond} -> mean_S={mean_S:.1f} "
          f"profit={profit:.1f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


# --- Wiring check (plan section 3.3): mixer sees joint state; IQL does not ----
def wiring_check(algo, K=DEFAULT_K):
    import torch
    use_mixer = (algo == "qmix")
    env = _build_env(0.75, 7, ALPHA, BETA, K=K)
    agents = list(env.possible_agents)

    # (1) action space the learner sees is Discrete(K) per agent
    for a in agents:
        sp = env.action_space(a)
        assert sp.__class__.__name__ == "Discrete" and sp.n == K, \
            f"action space for {a} is {sp}, expected Discrete({K})"
    print(f"  [1] action space = Discrete({K}) per agent for all {len(agents)} agents  OK")

    # (2) reward written to the buffer equals CPT(blend(raw, w)); log one transition
    obs, _ = env.reset(seed=0)
    act = {a: 0 for a in env.agents}
    _o, rewards, _t, _tr, info = env.step(act)
    a0 = agents[0]
    raw = info[a0]["raw_local_reward"]; blended = info[a0]["blended_reward"]
    cpt = info[a0]["cpt_reward"]
    # reconstruct CPT(blended) independently
    lam, al, be = 7, ALPHA, BETA
    expect = blended ** al if blended >= 0 else -lam * ((-blended) ** be)
    assert abs(cpt - expect) < 1e-6 and abs(rewards[a0] - cpt) < 1e-9, \
        f"reward transform mismatch: raw={raw} blended={blended} cpt={cpt} expect={expect}"
    print(f"  [2] one transition: raw={raw:.4f} blended={blended:.4f} "
          f"cpt={cpt:.4f} == CPT(blend)  OK")

    # (3) QMIX mixer receives the global state; IQL has no mixer
    learner = make_learner(env, use_mixer=use_mixer, seed=42)
    local_dim = env.local_dim
    aug_dim = int(np.prod(env.observation_space(a0).shape))
    state_dim = aug_dim - local_dim
    if algo == "qmix":
        assert learner.mixer is not None, "QMIX has NO mixer (silently IQL)!"
        first = learner.mixer.hyper_w1
        assert first.in_features == state_dim, \
            f"mixer hypernet in_features {first.in_features} != global slice {state_dim}"
        # monotonicity: dQ_tot/dQ_i >= 0 for the abs()'d weights
        qs = torch.zeros(1, learner.n_agents, requires_grad=True)
        st = torch.randn(1, state_dim)
        qtot = learner.mixer(qs, st)
        g, = torch.autograd.grad(qtot.sum(), qs)
        assert (g >= -1e-6).all(), f"mixer is NOT monotone: dQ_tot/dQ_i = {g.tolist()}"
        print(f"  [3] QMIX mixer reads global state (dim={state_dim}); "
              f"monotone dQ_tot/dQ_i>=0  OK")
    else:
        assert learner.mixer is None, "IQL has a mixer (should be independent)!"
        print(f"  [3] IQL has NO mixer; Q_tot = sum_i Q_i (independent)  OK")

    print("  WIRING CHECK PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["qmix", "iql"], required=True)
    ap.add_argument("--run_idx", type=int, default=None)
    ap.add_argument("--null", action="store_true",
                    help="run the null-baseline array (alpha=beta=1, lambda=1)")
    ap.add_argument("--timesteps", type=int, default=300_000)  # plan: 300k env steps
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.wiring_check:
        wiring_check(args.algo, K=args.K); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")

    out = args.out or (f"docs/expVB_{args.algo}_orgym_figures/"
                       f"{args.algo}_orgym_raw_data.csv")
    run_one(args.run_idx, args.algo, args.null, args.timesteps, args.K, out,
            verbose=args.verbose)


if __name__ == "__main__":
    main()
