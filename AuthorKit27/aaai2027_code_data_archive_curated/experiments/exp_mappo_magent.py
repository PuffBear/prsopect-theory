"""
Exp MAPPO-MAgent -- centralized-critic (CNN) MAPPO on MAgent2 battle_v4 with CPT
reward shaping. FOURTH ENVIRONMENT (breadth result), a structurally distinct
spatial-combat domain.

Why: the boundary replicates on supply chain (PPO+MAPPO), MPE (MAPPO, curvature),
and LBF (MAPPO, full loss-aversion sweep). MAgent adds a 4th, very different
substrate -- large-scale spatial combat with team focus-fire coordination -- to
push breadth from accept toward strong-accept. Gated viable by
run_magent_smoke_local.py (CNN-PPO @ w=1: red team reward -16.3 -> -2.0).

REAL CNN MAPPO: GlobalStateImageWrapper stacks agents' images on the channel axis;
CentralizedCriticImagePolicy routes the actor CNN to the own-image channel slice
and the critic CNN to the joint-image slice. This is the spatial analogue of the
validated flat MAPPO machinery (exp_mappo_lbf.py / mappo_policy.py).

    >>> BEFORE COMMITTING CLUSTER TIME:
          python experiments/exp_mappo_magent.py --wiring_check

Sweep (mirrors LBF -- the env that answered the breadth reject):
  w          in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}              (6)
  condition  null (a=b=1, lam=1) + lambda-sweep a=b=0.88, lam in {1..7}  (8)
  seeds      42-61                                          (20)
  Total: 6 x 8 x 20 = 960 runs                              (PBS array 0..959)
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

from env.magent_wrapper import make_magent_env, is_collapsed_magent, MAP_SIZE, MAX_CYCLES
from agents.mappo_image_policy import CentralizedCriticImagePolicy
from exp_ppo_lbf import MetricsCallback   # generic value_loss/expvar logger

# --- Grid --------------------------------------------------------------------
W_LIST = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CONDITIONS = [("null", 1.00, 1.00, 1.0)] + [
    (f"lambda{L}", 0.88, 0.88, float(L)) for L in range(1, 8)
]
SEEDS = list(range(42, 62))
N_W, N_COND, N_SEED = len(W_LIST), len(CONDITIONS), len(SEEDS)
N_RUNS = N_W * N_COND * N_SEED  # 960

PPO_KWARGS = dict(n_steps=256, batch_size=512, verbose=0)
N_ENV_COPIES = 1

COLUMNS = ["run_idx", "w", "condition", "alpha", "beta", "lambda_loss",
           "seed", "algo", "env",
           "mean_reward", "collapsed",
           "final_value_loss", "final_explained_variance"]


def _set_threads():
    """Honor the PBS thread budget for torch intra-op parallelism (the CNN is
    conv-heavy; without this torch may stay single-threaded and run ~4x slower)."""
    import torch
    n = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if n > 0:
        torch.set_num_threads(n)


def build_mappo_model(w, lam, alpha, beta, seed):
    _set_threads()
    env = make_magent_env(w, lam, alpha, beta, with_global_state=True)
    local_channels = env.local_channels
    vec = ss.black_death_v3(env)                # pad dead agents -> fixed agent set
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                base_class='stable_baselines3')
    vec.seed = lambda s: None
    policy_kwargs = dict(local_channels=local_channels,
                         net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO(CentralizedCriticImagePolicy, vec, seed=seed,
                policy_kwargs=policy_kwargs, **PPO_KWARGS)
    return model


def evaluate_magent(model, w, lam, alpha, beta, num_episodes=10, base_seed=0):
    """Mean red-team TRUE reward per episode (uses the global-state stack so the
    obs layout matches training; the critic channels are ignored by predict)."""
    env = make_magent_env(w, lam, alpha, beta, with_global_state=True)
    tot = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_r = 0.0
        while env.agents:
            acts = {a: int(model.predict(np.asarray(obs[a]), deterministic=True)[0])
                    for a in env.agents}
            obs, rew, term, trunc, info = env.step(acts)
            ep_r += sum(i.get("true_reward", 0.0) for i in info.values())
        tot.append(ep_r)
    return float(np.mean(tot))


def map_run_idx(run_idx):
    block = N_COND * N_SEED
    w_idx    = run_idx // block
    rem      = run_idx % block
    c_idx    = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], CONDITIONS[c_idx], SEEDS[seed_idx]


def run_one(run_idx, timesteps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, (cond, alpha, beta, lam), seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model = build_mappo_model(w, lam, alpha, beta, seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    mean_reward = evaluate_magent(model, w, lam, alpha, beta,
                                  num_episodes=10, base_seed=seed * 10)
    collapsed = is_collapsed_magent(mean_reward)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {
        "run_idx": run_idx, "w": w, "condition": cond,
        "alpha": alpha, "beta": beta, "lambda_loss": lam, "seed": seed,
        "algo": "MAPPO", "env": "MAgent_battle",
        "mean_reward": mean_reward, "collapsed": collapsed,
        "final_value_loss": fvl, "final_explained_variance": fev,
    }
    _append_row(row, out_path)
    print(f"[MAPPO-MAgent] run_idx={run_idx} w={w} cond={cond} lam={lam} seed={seed} "
          f"-> team_reward={mean_reward:.3f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            out_path, mode="a", header=write_header, index=False)


def wiring_check():
    """Assert the CNN centralized critic is built, trainable, and reads the joint
    (global) image channel slice -- not silently IPPO."""
    from agents.mappo_image_policy import _CentralCNNCriticExtractor
    model = build_mappo_model(0.6, 7.0, 0.88, 0.88, seed=42)
    pol = model.policy
    total_c = model.observation_space.shape[0]
    lc = pol._local_channels
    assert isinstance(pol.vf_features_extractor, _CentralCNNCriticExtractor), \
        "vf extractor is NOT the CNN centralized critic (silently IPPO!)"
    opt_params = {id(p) for g in pol.optimizer.param_groups for p in g["params"]}
    crit = list(pol.vf_features_extractor.parameters())
    assert crit and all(id(p) in opt_params for p in crit), \
        "centralized critic params are NOT in the optimizer!"
    assert pol.vf_features_extractor.global_channels == total_c - lc, \
        "critic input channels != global-image slice"
    print(f"  local_channels={lc} total_channels={total_c} "
          f"global_slice={total_c - lc}")
    print(f"  grid: {N_W} w x {N_COND} cond x {N_SEED} seeds = {N_RUNS} runs")
    print("  WIRING CHECK PASS: CNN centralized critic wired, trainable, reads joint image.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--out", default="docs/exp_magent/mappo_magent_raw_data.csv")
    ap.add_argument("--wiring_check", action="store_true")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.timesteps, args.out)


if __name__ == "__main__":
    main()
