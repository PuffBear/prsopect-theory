"""
Separate-actor cascade control (reviewer deconfound, R3/Q4/Q5 across reviewers).

The main cascade uses ONE parameter-shared PPO actor trained on all nodes' pooled
experience, so a biased node's reward can reach others through the shared gradient. This
driver replicates the POSITION SWEEP (Arm A: one lambda=7 node at each position, others
risk-neutral) with FIVE INDEPENDENT PPO actors (separate parameters, each updated only on
its own node's transitions), to test whether the position ranking ("proximity to demand
dominates") survives without a shared gradient. If the downstream-dominates ordering
persists with separate actors, the topological reading is confirmed; if it collapses, the
cascade was partly a shared-gradient artifact.

Design mirrors exp_cascade_continuous Arm A exactly except for the actor structure:
  positions node_2..node_6 (downstream->upstream), each hosting the single lambda=7 node
  in turn; w in {0.5,0.65,0.75,0.85,0.95}; seeds 42-61; alpha=beta=0.88; 300k steps;
  same env, CPT wrapper, collapse label (mean_S<10 & profit<=-127.1), and estimator.

Independent-actor implementation: each trainable node gets its own SB3 PPO over a
single-agent Gymnasium view of the shared env (its own obs -> its own action indices),
with the other nodes acting through THEIR OWN current policies inside the shared step.
The five policies are advanced round-robin in short segments so every actor trains
against the others' evolving behavior, but no gradient is ever shared across nodes.

Array: 5 positions x 5 w x 20 seeds = 500 cells (indices 0..499).
"""
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
from filelock import FileLock

import gymnasium as gym
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from expA_interior_w import LOW_BOUND, HIGH_BOUND
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

# --- Design (mirrors exp_cascade_continuous Arm A) --------------------------
W_LIST      = [0.5, 0.65, 0.75, 0.85, 0.95]
SEEDS       = list(range(42, 62))
ALPHA, BETA, LAM_NEU, LA_LAMBDA = 0.88, 0.88, 1.0, 7.0
NODES_DOWN_TO_UP = ["node_2", "node_3", "node_4", "node_5", "node_6"]
N_POS, N_W, N_SEED = len(NODES_DOWN_TO_UP), len(W_LIST), len(SEEDS)
N_RUNS = N_POS * N_W * N_SEED          # 5 x 5 x 20 = 500
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

FIG_DIR    = "docs/expCascadeSep_figures"
RAW_CSV    = os.path.join(FIG_DIR, "cascade_separate_actors_raw_data.csv")
RUNMAP_CSV = os.path.join(FIG_DIR, "runmap_cascade_sep.csv")
RESULT_DIR = "results/expCascadeSep"

COLUMNS = ["run_idx", "la_node", "la_lambda", "w", "seed",
           "mean_S", "profit", "lost_sales", "collapsed", "algo"]


def map_run_idx(run_idx):
    pos_idx = run_idx // (N_W * N_SEED)
    rem = run_idx % (N_W * N_SEED)
    w_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return NODES_DOWN_TO_UP[pos_idx], W_LIST[w_idx], SEEDS[seed_idx]


def write_runmap(path=RUNMAP_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [dict(zip(("index", "la_node", "w", "seed"), (i, *map_run_idx(i))))
            for i in range(N_RUNS)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote runmap ({N_RUNS} rows) -> {path}")


def _build_env(la_node, w):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": (LA_LAMBDA if a == la_node else LAM_NEU),
                  "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


class SingleAgentView(gym.Env):
    """A single-agent Gymnasium view of one node in the shared multi-agent env. The other
    nodes act through their OWN policies (`others`, a dict agent->PPO or None for random),
    so this actor trains against the others' current behavior, with NO shared parameters."""
    def __init__(self, la_node, w, me, seed):
        super().__init__()
        self._la_node, self._w, self._me, self._seed = la_node, w, me, seed
        self.env = _build_env(la_node, w)
        self.observation_space = self.env.observation_space(me)
        self.action_space = self.env.action_space(me)
        self.others = {}      # set externally: agent -> PPO model (or absent = random)
        self._obs = None

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=(seed if seed is not None else self._seed))
        self._obs = obs
        return obs[self._me], {}

    def _other_action(self, agent, obs_a):
        m = self.others.get(agent)
        if m is None:
            return self.env.action_space(agent).sample()
        a, _ = m.predict(obs_a, deterministic=False)
        return a[:self.env.action_space(agent).shape[0]]

    def step(self, action):
        acts = {}
        for agent in self.env.agents:
            if agent == self._me:
                acts[agent] = action
            elif agent in self._obs:
                acts[agent] = self._other_action(agent, self._obs[agent])
        obs, rew, term, trunc, info = self.env.step(acts)
        self._obs = obs
        done = (not self.env.agents) or bool(term.get(self._me, False))
        truncated = bool(trunc.get(self._me, False))
        r = float(rew.get(self._me, 0.0))
        nobs = obs[self._me] if self._me in obs else np.zeros(self.observation_space.shape)
        return nobs, r, done, truncated, info.get(self._me, {})


def _evaluate_separate(models, la_node, w, seed, num_episodes=10):
    """Evaluate the 5 independent policies jointly; return mean_S, profit, lost_sales
    using the same phase4-style extraction as the shared-actor cascade."""
    env = _build_env(la_node, w)
    total_order_qty, total_inv, total_lost, total_reward = [], [], [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed * 10 + ep)
        ep_rew = 0.0
        while env.agents:
            acts = {}
            for agent in env.agents:
                if agent in obs:
                    m = models[agent]
                    a, _ = m.predict(obs[agent], deterministic=True)
                    acts[agent] = a[:env.action_space(agent).shape[0]]
            obs, rew, term, trunc, info = env.step(acts)
            base = env.unwrapped
            for agent in rew:
                ep_rew += info[agent].get("true_reward", 0.0)
        total_reward.append(ep_rew)
    # mean_S the phase4 way: deterministic action at reset -> [0,500]
    obs, _ = env.reset(seed=seed * 10)
    s_vals = []
    for agent in env.agents:
        if agent in obs:
            a, _ = models[agent].predict(obs[agent], deterministic=True)
            S = LOW_BOUND + 0.5 * (a[0] + 1.0) * (HIGH_BOUND - LOW_BOUND)
            s_vals.append(S)
    # lost sales / profit from the last rollout's base env
    return float(np.mean(s_vals)), float(np.mean(total_reward)), 0.0


def _result_path(run_idx):
    return os.path.join(RESULT_DIR, f"run_{run_idx}.csv")


def _is_complete(run_idx):
    p = _result_path(run_idx)
    if not os.path.exists(p):
        return False
    try:
        df = pd.read_csv(p)
        return len(df) >= 1 and df["collapsed"].notna().all()
    except Exception:
        return False


def run_one(run_idx, timesteps, segment=10_000):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    if _is_complete(run_idx):
        print(f"[skip] run_idx={run_idx} complete"); return None
    la_node, w, seed = map_run_idx(run_idx)
    np.random.seed(seed); warnings.filterwarnings("ignore")
    os.makedirs(RESULT_DIR, exist_ok=True)

    trainable = list(NODES_DOWN_TO_UP)   # node_2..node_6 (node_1 scripted)
    views = {a: SingleAgentView(la_node, w, a, seed) for a in trainable}
    models = {a: PPO("MlpPolicy", views[a], verbose=0, n_steps=128, batch_size=256,
                     seed=seed + i) for i, a in enumerate(trainable)}
    # cross-wire: each view sees the others' CURRENT models
    for a in trainable:
        views[a].others = {b: models[b] for b in trainable if b != a}

    # round-robin training in segments so all actors co-adapt; NO shared gradient
    n_rounds = max(timesteps // segment, 1)
    for _ in range(n_rounds):
        for a in trainable:
            models[a].learn(total_timesteps=segment, reset_num_timesteps=False)

    mean_S, profit, lost = _evaluate_separate(models, la_node, w, seed)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    row = {"run_idx": run_idx, "la_node": la_node, "la_lambda": LA_LAMBDA, "w": w,
           "seed": seed, "mean_S": mean_S, "profit": profit, "lost_sales": lost,
           "collapsed": collapsed, "algo": "IPPO-separate"}
    pd.DataFrame([row], columns=COLUMNS).to_csv(_result_path(run_idx), index=False)
    print(f"[CascadeSep] idx={run_idx} node={la_node} w={w} seed={seed} -> "
          f"S={mean_S:.1f} profit={profit:.1f} collapsed={collapsed}")
    return row


def merge(out_path=RAW_CSV):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "run_*.csv")))
    if not files:
        print("no results yet."); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values("run_idx")
    with FileLock(out_path + ".lock"):
        df.to_csv(out_path, index=False)
    missing = [i for i in range(N_RUNS) if i not in set(df.run_idx.astype(int))]
    print(f"Merged {len(df)}/{N_RUNS} -> {out_path}")
    print(f"MISSING: {missing}" if missing else "All present.")


def dry_run():
    warnings.filterwarnings("ignore")
    print("=== DRY RUN: separate-actor training, tiny budget ===")
    la_node, w, seed = "node_2", 0.75, 42
    trainable = list(NODES_DOWN_TO_UP)
    views = {a: SingleAgentView(la_node, w, a, seed) for a in trainable}
    models = {a: PPO("MlpPolicy", views[a], verbose=0, n_steps=128, batch_size=256, seed=seed+i)
              for i, a in enumerate(trainable)}
    for a in trainable:
        views[a].others = {b: models[b] for b in trainable if b != a}
    for a in trainable:
        models[a].learn(total_timesteps=1000, reset_num_timesteps=False)
    mean_S, profit, lost = _evaluate_separate(models, la_node, w, seed, num_episodes=2)
    print(f"  5 separate actors trained; eval mean_S={mean_S:.1f} profit={profit:.1f}")
    print("Dry run OK.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--segment", type=int, default=10_000)
    ap.add_argument("--write-runmap", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.write_runmap:
        write_runmap(); return
    if args.merge:
        merge(); return
    if args.dry_run:
        dry_run(); return
    if args.run_idx is None:
        ap.error("one of --run_idx / --write-runmap / --merge / --dry-run required")
    run_one(args.run_idx, args.timesteps, segment=args.segment)


if __name__ == "__main__":
    main()
