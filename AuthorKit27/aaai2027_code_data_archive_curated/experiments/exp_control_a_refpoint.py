"""
Control A -- Reference-point sweep (highest-priority robustness control).

Question: the CPT transform currently treats ZERO as the gain-loss reference point
(v(x), kink at x=0). Does the loss-aversion rightward shift in w_crit survive plausible
ALTERNATIVE reference points z0, i.e. v(x - z0)?

Design (deliberately reduced -- this tests robustness, not a re-run of the paper):
  - Reference points z0 (4 arms):
      z0=0             : the current paper condition (kink at 0).
      z0=mean          : empirical MEAN of the blended per-step reward (lambda=1 pilot).
      z0=median        : empirical MEDIAN of the blended per-step reward (lambda=1 pilot).
      z0=-0.5sigma     : a fixed principled offset, -0.5 * std(blended reward) (lambda=1 pilot).
  - lambda in {1, 3, 7}      (NOT the full sweep -- robustness only)
  - w in {0.5, 0.65, 0.75, 0.85, 0.95}   (IDENTICAL to the paper's grid)
  - seeds 42-61 (20)         (IDENTICAL seeds), alpha=beta=0.88, PPO, 300k steps.
  Everything except z0 (and lambda/w/seed) is held identical to exp_ppo_lambda_sweep.

z0 is measured ONCE from a lambda=1 pilot that logs the blended per-step reward, then
FROZEN in docs/expControlA_refpoint_figures/refpoint_z0.json and committed BEFORE the
main analysis (same pre-registration discipline as the collapse label). z0 is a single
fixed number PER REFERENCE-POINT ARM (not per-w, not per-lambda) -- a reference point the
    agent plausibly holds, applied uniformly. The frozen z0 record is stored with the outputs.

Per-condition diagnostics logged from
the evaluation rollout of every cell:
  frac_gain / frac_loss : fraction of per-step transformed rewards on the gain/loss side
                          of z0 (i.e. sign of (blended - z0)).
  mean_cpt_reward       : mean transformed (CPT) per-step reward.
  reward_std            : std of the per-step blended reward.
  return_std            : std of per-episode TRUE return across eval episodes.

Collapse label: the supply chain's frozen label (mean_S<10 and profit<=-127.1),
unchanged. Output schema extends the cascade CSV with z0_arm / z0_value + the diagnostics.

Array: 4 arms x 3 lambda x 5 w x 20 seeds = 1200 cells (indices 0..1199).
"""
import os
import sys
import json
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
from filelock import FileLock

import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from expA_interior_w import (evaluate_and_extract_metrics, MetricsCallback,
                             LOW_BOUND, HIGH_BOUND)
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

# --- Design ------------------------------------------------------------------
W_LIST      = [0.5, 0.65, 0.75, 0.85, 0.95]      # IDENTICAL to the paper's grid
LAMBDA_LIST = [1, 3, 7]                           # reduced (robustness only)
SEEDS       = list(range(42, 62))                 # IDENTICAL 20 seeds
ALPHA, BETA = 0.88, 0.88
Z0_ARMS     = ["zero", "mean", "median", "neg_half_sigma"]  # 4 reference-point arms
N_ARM, N_LAM, N_W, N_SEED = len(Z0_ARMS), len(LAMBDA_LIST), len(W_LIST), len(SEEDS)
N_RUNS = N_ARM * N_LAM * N_W * N_SEED             # 4 x 3 x 5 x 20 = 1200

S_FLOOR, PROFIT_CEIL = 10.0, -127.1               # frozen supply-chain collapse label

FIG_DIR    = "docs/expControlA_refpoint_figures"
RAW_CSV    = os.path.join(FIG_DIR, "control_a_refpoint_raw_data.csv")
RUNMAP_CSV = os.path.join(FIG_DIR, "runmap_control_a.csv")
Z0_JSON    = os.path.join(FIG_DIR, "refpoint_z0.json")
RESULT_DIR = "results/expControlA"

COLUMNS = ["run_idx", "z0_arm", "z0_value", "lambda_loss", "w", "seed",
           "mean_S", "profit", "lost_sales", "collapsed",
           "frac_gain", "frac_loss", "mean_cpt_reward", "reward_std", "return_std",
           "final_value_loss", "final_explained_variance"]


# --- z0 (frozen; None arms except 'zero' until the pilot writes the JSON) -----

def load_z0():
    """Return {arm: z0_value}. 'zero' is always 0.0; the empirical arms come from the
    frozen JSON. Raises for empirical arms if the JSON is missing (pre-registration)."""
    z0 = {"zero": 0.0}
    try:
        with open(Z0_JSON) as f:
            d = json.load(f)
        for arm in ("mean", "median", "neg_half_sigma"):
            z0[arm] = float(d[arm])
    except (FileNotFoundError, KeyError, ValueError, OSError):
        pass
    return z0


def _require_z0(arm, z0map):
    if arm not in z0map:
        raise RuntimeError(
            f"No frozen z0 for arm '{arm}' at {Z0_JSON}. Run --pilot, commit the JSON "
            "BEFORE the main analysis (pre-registration), then rerun.")
    return z0map[arm]


# --- Index mapping (arm -> lambda -> w -> seed) ------------------------------

def map_run_idx(run_idx):
    per_arm = N_LAM * N_W * N_SEED
    arm_idx = run_idx // per_arm
    rem     = run_idx % per_arm
    per_lam = N_W * N_SEED
    lam_idx = rem // per_lam
    rem2    = rem % per_lam
    w_idx   = rem2 // N_SEED
    seed_idx = rem2 % N_SEED
    return Z0_ARMS[arm_idx], LAMBDA_LIST[lam_idx], W_LIST[w_idx], SEEDS[seed_idx]


def write_runmap(path=RUNMAP_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [dict(zip(("index", "z0_arm", "lambda_loss", "w", "seed"),
                     (i, *map_run_idx(i)))) for i in range(N_RUNS)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote runmap ({N_RUNS} rows) -> {path}")


# --- Env builder (only z0 differs from exp_ppo_lambda_sweep) -----------------

def _build_env(w, lam, z0):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": lam, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w,
                           reference_point=z0)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def _vectorise(env, seed):
    vec = ss.pad_action_space_v0(env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class="stable_baselines3")
    vec.seed = lambda s: None
    return vec


# --- Diagnostics from an eval rollout (free, requested) ----------------------

def collect_reward_diagnostics(env, model, z0, num_episodes=5, base_seed=0):
    """Roll out and collect gain/loss fractions, mean CPT reward, reward std, and
    per-episode true-return std -- all relative to the reference point z0."""
    step_blended, step_cpt, gains, losses = [], [], 0, 0
    ep_true_returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_true = 0.0
        while env.agents:
            actions = {}
            for a in env.agents:
                if a in obs:
                    act, _ = model.predict(obs[a], deterministic=True)
                    real = env.action_space(a).shape[0]
                    actions[a] = act[:real]
            obs, rewards, term, trunc, info = env.step(actions)
            for a in rewards:
                bl = info[a].get("blended_reward")
                cp = info[a].get("cpt_reward")
                if bl is not None:
                    step_blended.append(bl)
                    if (bl - z0) >= 0:
                        gains += 1
                    else:
                        losses += 1
                if cp is not None:
                    step_cpt.append(cp)
                ep_true += info[a].get("true_reward", 0.0)
        ep_true_returns.append(ep_true)
    n = max(gains + losses, 1)
    return dict(
        frac_gain=gains / n, frac_loss=losses / n,
        mean_cpt_reward=float(np.mean(step_cpt)) if step_cpt else float("nan"),
        reward_std=float(np.std(step_blended)) if step_blended else float("nan"),
        return_std=float(np.std(ep_true_returns)) if len(ep_true_returns) > 1 else float("nan"),
    )


# --- Single run --------------------------------------------------------------

def _result_path(run_idx):
    return os.path.join(RESULT_DIR, f"run_{run_idx}.csv")


def _is_complete(run_idx):
    p = _result_path(run_idx)
    if not os.path.exists(p):
        return False
    try:
        df = pd.read_csv(p)
        return len(df) >= 1 and "collapsed" in df.columns and df["collapsed"].notna().all()
    except Exception:
        return False


def run_one(run_idx, timesteps):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    if _is_complete(run_idx):
        print(f"[skip] run_idx={run_idx} already complete"); return None

    arm, lam, w, seed = map_run_idx(run_idx)
    z0 = _require_z0(arm, load_z0())
    np.random.seed(seed); warnings.filterwarnings("ignore")
    os.makedirs(RESULT_DIR, exist_ok=True)

    vec = _vectorise(_build_env(w, lam, z0), seed)
    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, z0)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    diag = collect_reward_diagnostics(_build_env(w, lam, z0), model, z0,
                                      num_episodes=5, base_seed=seed * 100)

    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "z0_arm": arm, "z0_value": z0, "lambda_loss": lam,
           "w": w, "seed": seed, "mean_S": mean_S, "profit": profit,
           "lost_sales": lost_sales, "collapsed": collapsed,
           "frac_gain": diag["frac_gain"], "frac_loss": diag["frac_loss"],
           "mean_cpt_reward": diag["mean_cpt_reward"], "reward_std": diag["reward_std"],
           "return_std": diag["return_std"], "final_value_loss": fvl,
           "final_explained_variance": fev}
    pd.DataFrame([row], columns=COLUMNS).to_csv(_result_path(run_idx), index=False)
    print(f"[ControlA] idx={run_idx} arm={arm} z0={z0:.3f} lam={lam} w={w} seed={seed} "
          f"-> S={mean_S:.1f} profit={profit:.1f} collapsed={collapsed} "
          f"frac_loss={diag['frac_loss']:.2f}")
    return row


# --- Merge -------------------------------------------------------------------

def merge(out_path=RAW_CSV):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "run_*.csv")))
    if not files:
        print(f"No per-element results in {RESULT_DIR}/."); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values("run_idx")
    with FileLock(out_path + ".lock"):
        df.to_csv(out_path, index=False)
    present = set(df["run_idx"].astype(int))
    missing = [i for i in range(N_RUNS) if i not in present]
    print(f"Merged {len(df)}/{N_RUNS} -> {out_path}")
    print(f"MISSING {len(missing)}: {missing}" if missing else "All indices present.")


# --- Pilot: measure + FREEZE z0 from the lambda=1 blended reward distribution ----

def pilot(timesteps, seeds=(42, 43, 44), ws=W_LIST, freeze=False):
    """Train lambda=1 (z0=0) across the w-grid, collect the BLENDED per-step reward, and
    report mean / median / -0.5*std. These become the frozen z0 for the empirical arms.
    z0 is a single fixed value per arm (pooled across w), i.e. a reference point the agent
    holds independent of the current mixing coefficient."""
    warnings.filterwarnings("ignore")
    print(f"[pilot] lambda=1, ws={ws}, seeds={seeds}, timesteps={timesteps}")
    blended = []
    for w in ws:
        for seed in seeds:
            np.random.seed(seed)
            vec = _vectorise(_build_env(w, 1, 0.0), seed)
            model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
            model.learn(total_timesteps=timesteps, callback=MetricsCallback())
            env = _build_env(w, 1, 0.0)
            obs, _ = env.reset(seed=seed * 7)
            steps = 0
            while env.agents and steps < 400:
                acts = {}
                for a in env.agents:
                    if a in obs:
                        act, _ = model.predict(obs[a], deterministic=True)
                        acts[a] = act[:env.action_space(a).shape[0]]
                obs, rew, term, trunc, info = env.step(acts)
                for a in rew:
                    if info[a].get("blended_reward") is not None:
                        blended.append(info[a]["blended_reward"])
                steps += 1
        print(f"  w={w} done, pooled n={len(blended)}")
    arr = np.array(blended)
    z0 = {"mean": float(np.mean(arr)), "median": float(np.median(arr)),
          "neg_half_sigma": float(-0.5 * np.std(arr))}
    print(f"\n[pilot] blended reward: mean={z0['mean']:.3f} median={z0['median']:.3f} "
          f"std={np.std(arr):.3f} -> -0.5sigma={z0['neg_half_sigma']:.3f}")
    if freeze:
        os.makedirs(os.path.dirname(Z0_JSON), exist_ok=True)
        with open(Z0_JSON, "w") as f:
            json.dump(z0, f, indent=2)
        print(f"[pilot] WROTE {Z0_JSON}. git commit it BEFORE the main analysis "
              "(pre-registration), then record the hash in the driver header + RESULTS.")
    else:
        print(f"[pilot] To FREEZE: write these to {Z0_JSON} and git commit BEFORE analysis:")
        print("        " + json.dumps(z0))
    return z0


# --- Dry run -----------------------------------------------------------------

def dry_run():
    warnings.filterwarnings("ignore")
    print("=== DRY RUN: reference-point shifts the gain/loss kink ===")
    # Same blended reward, different z0 -> different sign classification + CPT value.
    for z0 in (0.0, -2.0, 2.0):
        params = {"node_2": {"lambda": 7.0, "alpha": ALPHA, "beta": BETA}}
        # hand-check the transform math directly on the wrapper
        raw = MultiAgentNetInvMgmt()
        interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
        p = {a: {"lambda": 7.0, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
        cpt = CPTRewardWrapper(interv, p, reward_scale=1.0, global_reward_weight=0.75,
                               reference_point=z0)
        cpt.reset(seed=42)
        acts = {a: cpt.action_space(a).sample() for a in cpt.agents}
        _o, _r, _t, _tr, info = cpt.step(acts)
        a0 = list(info)[0]
        bl = info[a0]["blended_reward"]
        print(f"z0={z0:+.1f}: blended={bl:.3f} scaled=(bl-z0)={bl - z0:+.3f} "
              f"cpt={info[a0]['cpt_reward']:.3f} "
              f"side={'GAIN' if (bl - z0) >= 0 else 'LOSS'}")
    print("=== one tiny cell (2k steps, arm=zero) ===")
    vec = _vectorise(_build_env(0.75, 1, 0.0), 42)
    m = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=42)
    m.learn(total_timesteps=2000, callback=MetricsCallback())
    diag = collect_reward_diagnostics(_build_env(0.75, 1, 0.0), m, 0.0, num_episodes=2, base_seed=1)
    print(f"  diagnostics: {diag}")
    print("Dry run OK.")


# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, help=f"0..{N_RUNS-1}")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--write-runmap", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot_timesteps", type=int, default=150_000)
    ap.add_argument("--pilot_ws", default=None,
                    help="comma list of w for the pilot (default: full w-grid)")
    ap.add_argument("--pilot_seeds", default=None,
                    help="comma list of seeds for the pilot (default: 42,43,44)")
    ap.add_argument("--freeze_z0", action="store_true",
                    help="with --pilot: write the measured z0 to the JSON (still commit manually)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.write_runmap:
        write_runmap(); return
    if args.merge:
        merge(); return
    if args.pilot:
        ws = ([float(x) for x in args.pilot_ws.split(",")] if args.pilot_ws else W_LIST)
        seeds = ([int(x) for x in args.pilot_seeds.split(",")] if args.pilot_seeds
                 else (42, 43, 44))
        pilot(args.pilot_timesteps, seeds=seeds, ws=ws, freeze=args.freeze_z0); return
    if args.dry_run:
        dry_run(); return
    if args.run_idx is None:
        ap.error("one of --run_idx / --write-runmap / --merge / --pilot / --dry-run required")
    run_one(args.run_idx, args.timesteps)


if __name__ == "__main__":
    main()
