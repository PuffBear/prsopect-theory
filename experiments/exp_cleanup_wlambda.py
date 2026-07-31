"""
Exp CLEANUP-W-LAMBDA -- MAPPO on the in-repo Cleanup social dilemma (env/cleanup_env.py)
with the paper's w-dial + CPT machinery, full w x lambda x 20-seed sweep.

WHY THIS SUBSTRATE (and not the real DeepMind Melting Pot): the real `meltingpot`
depends on the compiled dmlab2d engine, whose Linux wheels require glibc >= 2.31. The
cluster is CentOS 7 / glibc 2.17 with no container runtime, so dmlab2d cannot load and
cannot be pip-built without a multi-day toolchain effort (the same binary wall this
project's notes record for SUMO/SMAC). The in-repo Cleanup is a pure-NumPy
implementation of the SAME Leibo-et-al-2017 tragedy-of-the-commons structure
(harvest-vs-clean; local cost / global benefit), runs on glibc 2.17 today, and carries
the identical reward-blend + CPT contract. RESULTS_B.md states this substitution openly:
this is the lightweight Cleanup, NOT the DeepMind benchmark.

PURPOSE (paper Experiment B): transfer the DIRECTION result -- loss aversion (lambda)
raises the coordination-collapse boundary w_crit -- to a social-dilemma substrate whose
public-good structure matches the paper's local-cost/global-benefit mechanism.

Reward blend + CPT layered exactly as every other substrate (blend then transform):
    r_i^PT = v( (1-w) r_i^local + w r^global ),   v as in the paper (alpha,beta,lambda)
with r_i^local = agent i's apple harvest, r^global = mean per-agent harvest (the w-blend
is done by CleanupLocalRewardWrapper inside make_cleanup_env_with_w; CPT wrapper uses
global_reward_weight=0.0 so the blend is not double-applied). Cleaning is unrewarded.

=== SWEEP GRID ===
    w      in {0.0,0.1,0.2,0.3,0.5,0.6,0.7,0.75,0.80,0.85,0.90,0.95,1.0}  (13 levels)
    lambda in {1,2,3,4,5,6,7}                                             (7 levels)
    seeds  42-61                                                          (20 seeds)
    alpha=beta=0.88. Total: 13 x 7 x 20 = 1820 cells (indices 0..1819).

=== PRE-REGISTRATION ===
    Collapse label = mean collective return < THRESHOLD, frozen in
    docs/expCleanup_wlambda_figures/cleanup_wlambda_collapse_threshold.json and COMMITTED
    BEFORE the main analysis. Set from `--pilot`. FROZEN-LABEL COMMIT HASH: <FILL>.

=== ARRAY / RESUME CONTRACT (PBS array, one element = one cell) ===
    - array index -> (w, lambda, seed) via committed runmap_cleanup_wlambda.csv
    - unique per-element file results/expCleanupWL/run_${idx}.csv; idempotent skip
    - checkpoint checkpoints/expCleanupWL/run_${idx}.zip; resume on requeue
    - fail loud (nonzero exit on exception); separate --merge concatenates.

References: Leibo et al. (2017) AAMAS; Yu et al. (2022) MAPPO (NeurIPS).
"""
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
from filelock import FileLock

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env.cleanup_env import make_cleanup_env_with_w
from env.cleanup_wlambda_collapse import is_collapsed_cleanup_wlambda
from agents.cpt_wrapper import CPTRewardWrapper
from exp_ppo_cleanup import evaluate_cleanup   # reuse the existing evaluator verbatim

# --- Sweep grid (matches the meltingpot driver so the two are comparable) ----
W_LIST      = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
LAMBDA_LIST = [1, 2, 3, 4, 5, 6, 7]
SEEDS       = list(range(42, 62))
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED                 # 13 x 7 x 20 = 1820

ALPHA, BETA = 0.88, 0.88
N_AGENTS    = 3
MAX_CYCLES  = 200

FIG_DIR    = "docs/expCleanup_wlambda_figures"
RAW_CSV    = os.path.join(FIG_DIR, "cleanup_wlambda_raw_data.csv")
RUNMAP_CSV = os.path.join(FIG_DIR, "runmap_cleanup_wlambda.csv")
RESULT_DIR = "results/expCleanupWL"
CKPT_DIR   = "checkpoints/expCleanupWL"

COLUMNS = ["run_idx", "w", "lambda_loss", "seed", "algo", "env",
           "collective_return", "own_return", "collapsed",
           "final_value_loss", "final_explained_variance"]


# --- MAPPO centralised critic (wider vf, matches exp_mappo_cleanup) ----------

class MAPPOCriticExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 128), nn.ReLU(),
            nn.Linear(128, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def make_policy_kwargs():
    return dict(
        features_extractor_class=MAPPOCriticExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[128, 128]),
    )


# --- Index mapping -----------------------------------------------------------

def map_run_idx(run_idx):
    lam_seed_block = N_LAM * N_SEED
    w_idx    = run_idx // lam_seed_block
    rem      = run_idx % lam_seed_block
    lam_idx  = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], LAMBDA_LIST[lam_idx], SEEDS[seed_idx]


def write_runmap(path=RUNMAP_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [dict(zip(("index", "w", "lambda_loss", "seed"), (i, *map_run_idx(i))))
            for i in range(N_RUNS)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote runmap ({N_RUNS} rows) -> {path}")


# --- Env builder (blend then transform) -------------------------------------

def _build_env(w, lam, alpha, beta):
    local_w = make_cleanup_env_with_w(w, n_agents=N_AGENTS, max_cycles=MAX_CYCLES)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in local_w.possible_agents}
    return CPTRewardWrapper(local_w, params, reward_scale=1.0, global_reward_weight=0.0)


def _vectorise(env):
    # Some supersuit versions read `render_mode` off the unwrapped env; the custom
    # CleanupEnv doesn't set it. Provide it defensively (no behavioural change).
    if not hasattr(env.unwrapped, "render_mode"):
        try:
            env.unwrapped.render_mode = None
        except Exception:
            pass
    vec = ss.pad_action_space_v0(env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class="stable_baselines3")
    vec.seed = lambda s: None
    return vec


# --- Callback ----------------------------------------------------------------

class MetricsCallback(BaseCallback):
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


# --- Single run --------------------------------------------------------------

def _result_path(run_idx):
    return os.path.join(RESULT_DIR, f"run_{run_idx}.csv")


def _ckpt_zip(run_idx):
    return os.path.join(CKPT_DIR, f"run_{run_idx}.zip")


def _is_complete(run_idx):
    p = _result_path(run_idx)
    if not os.path.exists(p):
        return False
    try:
        df = pd.read_csv(p)
        return len(df) >= 1 and "collapsed" in df.columns and df["collapsed"].notna().all()
    except Exception:
        return False


def run_one(run_idx, timesteps, ckpt_every=50_000):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    if _is_complete(run_idx):
        print(f"[skip] run_idx={run_idx} already complete at {_result_path(run_idx)}")
        return None

    w, lam, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    vec = _vectorise(_build_env(w, lam, ALPHA, BETA))
    resume = _ckpt_zip(run_idx)
    if os.path.exists(resume):
        print(f"[resume] loading {resume}")
        model = PPO.load(resume, env=vec)
        remaining = max(timesteps - model.num_timesteps, 0)
    else:
        model = PPO("MlpPolicy", vec, verbose=0, n_steps=256, batch_size=512,
                    seed=seed, policy_kwargs=make_policy_kwargs())
        remaining = timesteps

    cb = MetricsCallback()
    ckpt_cb = CheckpointCallback(save_freq=max(ckpt_every, 1), save_path=CKPT_DIR,
                                 name_prefix=f"run_{run_idx}")
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=[cb, ckpt_cb],
                    reset_num_timesteps=False)
    model.save(resume)

    eval_env = _build_env(w, lam, ALPHA, BETA)
    collective = evaluate_cleanup(model, eval_env, num_episodes=10, base_seed=seed * 10)
    own = collective / N_AGENTS
    collapsed = is_collapsed_cleanup_wlambda(collective)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "seed": seed,
           "algo": "MAPPO", "env": "Cleanup_custom",
           "collective_return": collective, "own_return": own, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    pd.DataFrame([row], columns=COLUMNS).to_csv(_result_path(run_idx), index=False)
    print(f"[CleanupWL] run_idx={run_idx} w={w} lam={lam} seed={seed} -> "
          f"collective={collective:.2f} collapsed={collapsed}")
    return row


# --- Merge + verify ----------------------------------------------------------

def merge(out_path=RAW_CSV):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "run_*.csv")))
    if not files:
        print(f"No per-element results in {RESULT_DIR}/ -- nothing to merge.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values("run_idx")
    with FileLock(out_path + ".lock"):
        df.to_csv(out_path, index=False)
    present = set(df["run_idx"].astype(int))
    missing = [i for i in range(N_RUNS) if i not in present]
    print(f"Merged {len(df)}/{N_RUNS} rows -> {out_path}")
    print(f"MISSING {len(missing)} indices (requeue): {missing}" if missing
          else "All runmap indices present.")


# --- Pilot (freeze the threshold) -------------------------------------------

def pilot(timesteps, seeds=(42, 43), ws=(0.0, 0.1, 0.95)):
    warnings.filterwarnings("ignore")
    print(f"[pilot] ws={ws} seeds={seeds} timesteps={timesteps} (lambda=1)")
    scale = {}
    for w in ws:
        vals = []
        for seed in seeds:
            np.random.seed(seed)
            model = PPO("MlpPolicy", _vectorise(_build_env(w, 1, ALPHA, BETA)),
                        verbose=0, n_steps=256, batch_size=512, seed=seed,
                        policy_kwargs=make_policy_kwargs())
            model.learn(total_timesteps=timesteps, callback=MetricsCallback())
            coll = evaluate_cleanup(model, _build_env(w, 1, ALPHA, BETA),
                                    num_episodes=5, base_seed=seed * 10)
            vals.append(coll)
            print(f"  w={w} seed={seed} -> collective_return={coll:.2f}")
        scale[w] = float(np.mean(vals))
    lo, hi = scale[min(ws)], scale[max(ws)]
    suggested = round((lo + hi) / 2.0, 3)
    print(f"\n[pilot] low-w({min(ws)})={lo:.2f}  high-w({max(ws)})={hi:.2f}")
    print(f"[pilot] SUGGESTED frozen threshold (midpoint) = {suggested}")
    print(f"[pilot] To FREEZE: echo '{{\"threshold\": {suggested}}}' > "
          f"{FIG_DIR}/cleanup_wlambda_collapse_threshold.json ; git commit BEFORE analysis.")
    return suggested


# --- Dry run -----------------------------------------------------------------

def dry_run():
    warnings.filterwarnings("ignore")
    print("=== DRY RUN: reward blend + CPT check ===")
    for w in (0.0, 0.95):
        for lam in (1, 7):
            base = make_cleanup_env_with_w(w, n_agents=N_AGENTS, max_cycles=MAX_CYCLES)
            params = {a: {"lambda": lam, "alpha": ALPHA, "beta": BETA}
                      for a in base.possible_agents}
            cpt = CPTRewardWrapper(base, params, reward_scale=1.0, global_reward_weight=0.0)
            cpt.reset(seed=42)
            acts = {a: cpt.action_space(a).sample() for a in cpt.agents}
            _o, _r, _t, _tr, info = cpt.step(acts)
            a0 = cpt.possible_agents[0]
            print(f"w={w} lam={lam} {a0}: r_local={info[a0]['r_local']:.3f} "
                  f"r_global={info[a0]['r_global']:.3f} r_blend={info[a0]['blended_reward']:.3f} "
                  f"r_cpt={info[a0]['cpt_reward']:.3f}")
    print("=== one tiny cell (2k steps) ===")
    vec = _vectorise(_build_env(0.5, 1, ALPHA, BETA))
    m = PPO("MlpPolicy", vec, verbose=0, n_steps=256, batch_size=512, seed=42,
            policy_kwargs=make_policy_kwargs())
    m.learn(total_timesteps=2000, callback=MetricsCallback())
    coll = evaluate_cleanup(m, _build_env(0.5, 1, ALPHA, BETA), num_episodes=2, base_seed=420)
    print(f"  cell w=0.5 lam=1: collective={coll:.2f}")
    print("Dry run OK.")


# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, help=f"0..{N_RUNS-1} (one PBS array element)")
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--ckpt_every", type=int, default=50_000)
    ap.add_argument("--write-runmap", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot_timesteps", type=int, default=200_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.write_runmap:
        write_runmap(); return
    if args.merge:
        merge(); return
    if args.pilot:
        pilot(args.pilot_timesteps); return
    if args.dry_run:
        dry_run(); return
    if args.run_idx is None:
        ap.error("one of --run_idx / --write-runmap / --merge / --pilot / --dry-run required")
    run_one(args.run_idx, args.timesteps, ckpt_every=args.ckpt_every)


if __name__ == "__main__":
    main()
