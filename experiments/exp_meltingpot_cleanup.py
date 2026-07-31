"""
Exp MELTINGPOT-CLEANUP -- CNN-MAPPO on the *real* DeepMind Melting Pot `clean_up`
substrate with the paper's w-dial + CPT reward machinery.

PURPOSE (paper Experiment B): transfer the DIRECTION result -- loss aversion (lambda)
raises the coordination-collapse boundary w_crit -- to a recognized standard
social-dilemma benchmark whose local-cost/global-benefit structure matches the paper's
mechanism. Magnitude / functional form are expected to be substrate-specific (that is
already the paper's stance); the claim under test is the SIGN of dw_crit/dlambda.

This uses google-deepmind/meltingpot (import REQUIRED, no fallback) via
`env/meltingpot_cleanup_wrapper.py`. Reward blend + CPT are layered exactly as on every
other substrate:

    r_i^PT = v( (1-w) r_i^local + w r^global ),   v as in the paper (alpha,beta,lambda)

with r_i^local = agent i's raw apple harvest, r^global = mean per-agent harvest.
Cleaning is unrewarded -- w carries the incentive to maintain the commons.

=== SWEEP GRID (comparable to the other transfer substrates) ===
    w      in {0.0,0.1,0.2,0.3,0.5,0.6,0.7,0.75,0.80,0.85,0.90,0.95,1.0}  (13 levels)
           -- spans the full range: the standard supply-chain grid floors at 0.5, but
           the boundary can sit LOW on non-supply-chain substrates (the custom Cleanup
           uses w>=0.10; LBF needed a w in {0,0.1,0.2} extension to locate w_crit), so
           we cover 0.0..1.0 to keep the lambda=1 anchor informative.
    lambda in {1,2,3,4,5,6,7}                             (null CPT + six loss-aversion)
    seeds  42-61                                          (20 seeds)
    CPT curvature alpha=beta=0.88 (same as the supply-chain / Cleanup drivers).
    Total: 13 x 7 x 20 = 1820 cells.

=== PRE-REGISTRATION ===
    Collapse label = mean collective return < THRESHOLD, THRESHOLD frozen in
    docs/expMeltingpot_cleanup_figures/meltingpot_collapse_threshold.json and COMMITTED
    BEFORE the main analysis. Set the threshold from `--pilot` (reads low-w vs high-w
    scale), commit it, record the hash in the header below and in RESULTS_B.md.

    FROZEN LABEL COMMIT HASH: <FILL AFTER `--pilot` + git commit of the threshold JSON>

=== ARRAY / RESUME CONTRACT (PBS array, one element = one cell) ===
    - The array index maps to (w, lambda, seed) via the committed runmap
      (docs/expMeltingpot_cleanup_figures/runmap_B.csv), written by `--write-runmap`.
    - Each element writes a UNIQUE per-element file: results/expMeltingpot/run_${idx}.csv.
    - Idempotent: if that file already exists and is complete, the element SKIPS.
    - Checkpointing: the SB3 model is saved to checkpoints/expMeltingpot/run_${idx}.zip
      periodically and on exit; a requeued element resumes from it. Essential for long
      GPU jobs (pixel CNN + N=7 + long episodes = hours/run).
    - Fail loud: any exception -> nonzero exit (the PBS element shows as failed).
    - A separate `--merge` step concatenates all run_*.csv into the final raw_data CSV.

References:
    Leibo et al. (2017) AAMAS; Agapiou et al. (2022) Melting Pot 2.0 (arXiv:2211.13746);
    Yu et al. (2022) MAPPO (NeurIPS).
"""
import os
import sys
import glob
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from filelock import FileLock

import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import NatureCNN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from agents.cpt_wrapper import CPTRewardWrapper

# --- Sweep grid --------------------------------------------------------------
# Full-range w-grid. The standard supply-chain grid floors at 0.5, but on the
# non-supply-chain substrates (custom Cleanup w in {0.10..0.90}, and the MPE/LBF
# transfers) the boundary can sit LOW and slide below a 0.5 floor -- LBF needed a
# w in {0,0.1,0.2} extension to locate w_crit at all. So we span 0.0..1.0 to keep the
# lambda=1 anchor informative and avoid an off-grid boundary, while retaining the
# standard upper points {0.75,0.85,0.95} + finer {0.80,0.90} for comparability.
W_LIST      = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
LAMBDA_LIST = [1, 2, 3, 4, 5, 6, 7]
SEEDS       = list(range(42, 62))                         # 20 seeds
N_W, N_LAM, N_SEED = len(W_LIST), len(LAMBDA_LIST), len(SEEDS)
N_RUNS = N_W * N_LAM * N_SEED                             # 13 x 7 x 20 = 1820

ALPHA, BETA = 0.88, 0.88
SUBSTRATE   = "clean_up"

# Output locations (docs/exp*_figures convention).
FIG_DIR      = "docs/expMeltingpot_cleanup_figures"
RAW_CSV      = os.path.join(FIG_DIR, "meltingpot_cleanup_raw_data.csv")
RUNMAP_CSV   = os.path.join(FIG_DIR, "runmap_B.csv")
RESULT_DIR   = "results/expMeltingpot"
CKPT_DIR     = "checkpoints/expMeltingpot"

COLUMNS = ["run_idx", "w", "lambda_loss", "seed", "algo", "env",
           "collective_return", "own_return", "sustainability", "collapsed",
           "final_value_loss", "final_explained_variance"]


# --- Index mapping (w-major, then lambda, then seed) -------------------------

def map_run_idx(run_idx):
    lam_seed_block = N_LAM * N_SEED
    w_idx    = run_idx // lam_seed_block
    rem      = run_idx % lam_seed_block
    lam_idx  = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], LAMBDA_LIST[lam_idx], SEEDS[seed_idx]


def write_runmap(path=RUNMAP_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for idx in range(N_RUNS):
        w, lam, seed = map_run_idx(idx)
        rows.append({"index": idx, "w": w, "lambda_loss": lam, "seed": seed})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote runmap ({N_RUNS} rows) -> {path}")


# --- CNN-MAPPO ---------------------------------------------------------------

def make_policy_kwargs():
    """CNN-MAPPO: NatureCNN encoder (pixels) + a wider centralised critic head
    (Yu et al. 2022 MAPPO uses a larger value network than the policy)."""
    return dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[256, 256]),
    )


# --- Env builder (blend then transform, CPT weight 0 to avoid double-blend) --

def _build_env(w, lam, alpha, beta, max_cycles=None):
    from env.meltingpot_cleanup_wrapper import make_meltingpot_cleanup_with_w
    local_w = make_meltingpot_cleanup_with_w(w, substrate_name=SUBSTRATE,
                                             max_cycles=max_cycles)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in local_w.possible_agents}
    return CPTRewardWrapper(local_w, params, reward_scale=1.0, global_reward_weight=0.0)


def _vectorise(env):
    vec = ss.pettingzoo_env_to_vec_env_v1(env)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class="stable_baselines3")
    vec.seed = lambda s: None
    return vec


# --- Callbacks ---------------------------------------------------------------

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


# --- Evaluation --------------------------------------------------------------

def evaluate_meltingpot(model, env, num_episodes=10, base_seed=42):
    """Roll out `num_episodes` and return three episode-mean summaries:

        collective_return : mean over episodes of SUM over the team of raw apple harvest
                            (the TRUE collective return -- the collapse statistic).
        own_return        : mean per-agent raw harvest per episode.
        sustainability    : mean per-step team harvest rate (a coarse apple-supply
                            proxy: sustained supply => higher; collapse => ~0).
    """
    coll, own, sust = [], [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_true = 0.0
        n_steps = 0
        n_agents_seen = max(len(env.agents), 1)
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    act, _ = model.predict(obs[agent], deterministic=True)
                    actions[agent] = int(act)
            obs, rewards, term, trunc, info = env.step(actions)
            step_true = 0.0
            for agent in rewards:
                if agent in info and "true_reward" in info[agent]:
                    step_true += info[agent]["true_reward"]
                else:
                    step_true += rewards[agent]
            ep_true += step_true
            n_steps += 1
        coll.append(ep_true)
        own.append(ep_true / n_agents_seen)
        sust.append(ep_true / max(n_steps, 1))
    return float(np.mean(coll)), float(np.mean(own)), float(np.mean(sust))


# --- Single run --------------------------------------------------------------

def _result_path(run_idx):
    return os.path.join(RESULT_DIR, f"run_{run_idx}.csv")


def _ckpt_prefix(run_idx):
    return os.path.join(CKPT_DIR, f"run_{run_idx}")


def _is_complete(run_idx):
    """A result file counts as complete if it exists and has one well-formed row."""
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

    train_env = _build_env(w, lam, ALPHA, BETA)
    vec = _vectorise(train_env)

    # Resume from checkpoint if one exists (killed/preempted elements pick up here).
    resume_zip = _ckpt_prefix(run_idx) + ".zip"
    if os.path.exists(resume_zip):
        print(f"[resume] loading checkpoint {resume_zip}")
        model = PPO.load(resume_zip, env=vec)
        remaining = max(timesteps - model.num_timesteps, 0)
    else:
        model = PPO("CnnPolicy", vec, verbose=0,
                    n_steps=256, batch_size=512, seed=seed,
                    policy_kwargs=make_policy_kwargs())
        remaining = timesteps

    cb = MetricsCallback()
    ckpt_cb = CheckpointCallback(save_freq=max(ckpt_every, 1), save_path=CKPT_DIR,
                                 name_prefix=f"run_{run_idx}")
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=[cb, ckpt_cb],
                    reset_num_timesteps=False)
    model.save(resume_zip)   # final checkpoint

    eval_env = _build_env(w, lam, ALPHA, BETA)
    collective, own, sust = evaluate_meltingpot(model, eval_env, num_episodes=10,
                                                base_seed=seed * 10)

    from env.meltingpot_collapse import is_collapsed_meltingpot
    collapsed = is_collapsed_meltingpot(collective)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "lambda_loss": lam, "seed": seed,
           "algo": "CNN-MAPPO", "env": "MeltingPot_clean_up",
           "collective_return": collective, "own_return": own,
           "sustainability": sust, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    pd.DataFrame([row], columns=COLUMNS).to_csv(_result_path(run_idx), index=False)
    print(f"[MP-Cleanup] run_idx={run_idx} w={w} lam={lam} seed={seed} -> "
          f"collective={collective:.2f} own={own:.2f} collapsed={collapsed}")
    return row


# --- Merge + verify ----------------------------------------------------------

def merge(out_path=RAW_CSV):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "run_*.csv")))
    if not files:
        print(f"No per-element results in {RESULT_DIR}/ -- nothing to merge.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("run_idx").reset_index(drop=True)
    with FileLock(out_path + ".lock"):
        df.to_csv(out_path, index=False)
    present = set(df["run_idx"].astype(int))
    missing = [i for i in range(N_RUNS) if i not in present]
    print(f"Merged {len(df)}/{N_RUNS} rows -> {out_path}")
    if missing:
        print(f"MISSING {len(missing)} indices (requeue these): {missing}")
    else:
        print("All runmap indices present.")


# --- Pilot (for freezing the collapse threshold) ----------------------------

def pilot(timesteps, seeds=(42, 43), ws=(0.0, 0.1, 0.95)):
    """Small pilot to read the collective-return SCALE under obvious free-riding
    (low w) vs coordination (high w), lambda=1. Probes the low end (w=0.0, 0.1) so we
    also sanity-check that the boundary is not below the grid floor. Prints a suggested
    threshold from the min-w vs max-w scale. Do NOT tune it to make a boundary -- set it
    once, commit, freeze."""
    warnings.filterwarnings("ignore")
    print(f"[pilot] ws={ws} seeds={seeds} timesteps={timesteps} (lambda=1)")
    scale = {}
    for w in ws:
        vals = []
        for seed in seeds:
            np.random.seed(seed)
            env = _build_env(w, 1, ALPHA, BETA)
            vec = _vectorise(env)
            model = PPO("CnnPolicy", vec, verbose=0, n_steps=256, batch_size=512,
                        seed=seed, policy_kwargs=make_policy_kwargs())
            model.learn(total_timesteps=timesteps, callback=MetricsCallback())
            coll, _own, _s = evaluate_meltingpot(model, _build_env(w, 1, ALPHA, BETA),
                                                 num_episodes=5, base_seed=seed * 10)
            vals.append(coll)
            print(f"  w={w} seed={seed} -> collective_return={coll:.2f}")
        scale[w] = float(np.mean(vals))
    lo, hi = scale[min(ws)], scale[max(ws)]
    suggested = round((lo + hi) / 2.0, 3)   # midpoint of collapse vs coordination scale
    print(f"\n[pilot] mean collective_return: low-w({min(ws)})={lo:.2f}  "
          f"high-w({max(ws)})={hi:.2f}")
    print(f"[pilot] SUGGESTED frozen threshold (midpoint) = {suggested}")
    print(f"[pilot] To FREEZE: write {{\"threshold\": {suggested}}} to "
          f"{FIG_DIR}/meltingpot_collapse_threshold.json, then git commit BEFORE the "
          "main analysis and record the hash in RESULTS_B.md.")
    return suggested


# --- Dry run (mandatory local sanity check, brief section 5) -----------------

def dry_run():
    """1 seed x 2 w x lambda in {1,7}, tiny steps. Confirms a well-formed row AND logs
    the blended + CPT reward for one step so the reward math can be eyeballed."""
    warnings.filterwarnings("ignore")
    print("=== DRY RUN: reward blend + CPT check ===")
    for w in (0.5, 0.95):
        for lam in (1, 7):
            local_w_env = None
            from env.meltingpot_cleanup_wrapper import make_meltingpot_cleanup_with_w
            base = make_meltingpot_cleanup_with_w(w, substrate_name=SUBSTRATE)
            params = {a: {"lambda": lam, "alpha": ALPHA, "beta": BETA}
                      for a in base.possible_agents}
            cpt = CPTRewardWrapper(base, params, reward_scale=1.0, global_reward_weight=0.0)
            obs, _ = cpt.reset(seed=42)
            acts = {a: cpt.action_space(a).sample() for a in cpt.agents}
            obs, rew, term, trunc, info = cpt.step(acts)
            a0 = cpt.possible_agents[0]
            print(f"w={w} lam={lam} agent={a0}: "
                  f"r_local={info[a0]['r_local']:.3f} r_global={info[a0]['r_global']:.3f} "
                  f"r_blend={info[a0]['blended_reward']:.3f} "
                  f"r_cpt={info[a0]['cpt_reward']:.3f}")
    print("=== running one tiny full cell (2k steps) ===")
    _dry_one_cell(w=0.5, lam=1, timesteps=2000)
    _dry_one_cell(w=0.95, lam=7, timesteps=2000)
    print("Dry run OK.")


def _dry_one_cell(w, lam, timesteps):
    env = _build_env(w, lam, ALPHA, BETA)
    vec = _vectorise(env)
    model = PPO("CnnPolicy", vec, verbose=0, n_steps=256, batch_size=512, seed=42,
                policy_kwargs=make_policy_kwargs())
    model.learn(total_timesteps=timesteps, callback=MetricsCallback())
    coll, own, sust = evaluate_meltingpot(model, _build_env(w, lam, ALPHA, BETA),
                                          num_episodes=2, base_seed=420)
    print(f"  cell w={w} lam={lam}: collective={coll:.2f} own={own:.2f} sust={sust:.3f}")


# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, help=f"0..{N_RUNS-1} (one PBS array element)")  # 0..1819
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--ckpt_every", type=int, default=50_000)
    ap.add_argument("--write-runmap", action="store_true",
                    help="write the committed runmap and exit")
    ap.add_argument("--merge", action="store_true",
                    help="merge results/expMeltingpot/run_*.csv into the raw_data CSV")
    ap.add_argument("--pilot", action="store_true",
                    help="run the label-freezing pilot (low-w vs high-w scale)")
    ap.add_argument("--pilot_timesteps", type=int, default=300_000)
    ap.add_argument("--dry-run", action="store_true",
                    help="local reward-blend + CPT sanity check (brief section 5)")
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
        ap.error("one of --run_idx / --write-runmap / --merge / --pilot / --dry-run is required")
    run_one(args.run_idx, args.timesteps, ckpt_every=args.ckpt_every)


if __name__ == "__main__":
    main()
