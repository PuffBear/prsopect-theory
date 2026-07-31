"""
Control B -- Scale-matched control. Does lambda merely change PPO's effective reward
MAGNITUDE (i.e. act as reward amplification), or does the w_crit shift survive after the
generic reward-scale difference across lambda is removed?

Manipulation (distinct from A/C): renormalize the TRANSFORMED (CPT) reward by a
per-lambda constant so the transformed-reward RMS matches across lambda. If the
lambda=1->7 rightward w_crit shift SURVIVES this magnitude match, lambda is NOT merely
amplification -- the loss ASYMMETRY (not the amplitude) drives the boundary.

Two arms:
  raw        : output_scale = 1                (baseline, the paper condition)
  stdmatched : output_scale = rms(lambda) / rms(lambda=1)
               so every lambda's CPT reward has the SAME RMS as lambda=1 (the reference).
               (lambda=1's own factor is 1, i.e. the reference arm is unchanged at lambda=1.)

rms(lambda) is measured ONCE from a pilot (per-lambda RMS of the raw CPT reward over the
w-grid) and FROZEN in docs/expControlB_scalematch_figures/rms_output_scale.json, committed
BEFORE the main analysis (same pre-registration discipline as A's z0). FROZEN RMS COMMIT
HASH: <FILL>.

Design (reduced): 2 arms x lambda{1,3,7} x w{0.5,0.65,0.75,0.85,0.95} x seeds 42-61,
alpha=beta=0.88, PPO 300k. Everything except output_scale is identical to Control A.

Reward-side diagnostics (reward std, frac pos/neg, return std) are the SAME columns
Control A/C log; this driver logs them too so B is self-contained, and the analysis can
additionally pull A/C for the "report reward variance / sign / return variance" ask
without re-running.

Array: 2 arms x 3 lambda x 5 w x 20 seeds = 600 cells (indices 0..599).
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
from exp_control_a_refpoint import collect_reward_diagnostics
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

# --- Design ------------------------------------------------------------------
W_LIST      = [0.5, 0.65, 0.75, 0.85, 0.95]
LAMBDA_LIST = [1, 3, 7]
SEEDS       = list(range(42, 62))
ALPHA, BETA = 0.88, 0.88
ARM_NAMES   = ["raw", "stdmatched"]
N_ARM, N_LAM, N_W, N_SEED = len(ARM_NAMES), len(LAMBDA_LIST), len(W_LIST), len(SEEDS)
N_RUNS = N_ARM * N_LAM * N_W * N_SEED             # 2 x 3 x 5 x 20 = 600

S_FLOOR, PROFIT_CEIL = 10.0, -127.1

FIG_DIR    = "docs/expControlB_scalematch_figures"
RAW_CSV    = os.path.join(FIG_DIR, "control_b_scalematch_raw_data.csv")
RUNMAP_CSV = os.path.join(FIG_DIR, "runmap_control_b.csv")
RMS_JSON   = os.path.join(FIG_DIR, "rms_output_scale.json")
RESULT_DIR = "results/expControlB"

COLUMNS = ["run_idx", "match_arm", "output_scale", "lambda_loss", "w", "seed",
           "mean_S", "profit", "lost_sales", "collapsed",
           "frac_gain", "frac_loss", "mean_cpt_reward", "reward_std", "return_std",
           "final_value_loss", "final_explained_variance"]


def load_rms_scales():
    """{lambda: rms(lambda)/rms(1)} for the stdmatched arm. lambda=1 => 1.0 always.
    Empirical entries come from the frozen JSON; missing => raises for stdmatched."""
    scales = {1: 1.0}
    try:
        with open(RMS_JSON) as f:
            d = json.load(f)
        for lam in LAMBDA_LIST:
            if str(lam) in d:
                scales[lam] = float(d[str(lam)])
    except (FileNotFoundError, ValueError, OSError):
        pass
    return scales


def _output_scale(arm, lam):
    if arm == "raw":
        return 1.0
    scales = load_rms_scales()
    if lam not in scales:
        raise RuntimeError(
            f"No frozen RMS scale for lambda={lam} at {RMS_JSON}. Run --pilot, commit the "
            "JSON BEFORE the main analysis (pre-registration), then rerun.")
    return scales[lam]


def map_run_idx(run_idx):
    per_arm = N_LAM * N_W * N_SEED
    arm_idx = run_idx // per_arm
    rem     = run_idx % per_arm
    per_lam = N_W * N_SEED
    lam_idx = rem // per_lam
    rem2    = rem % per_lam
    w_idx   = rem2 // N_SEED
    seed_idx = rem2 % N_SEED
    return ARM_NAMES[arm_idx], LAMBDA_LIST[lam_idx], W_LIST[w_idx], SEEDS[seed_idx]


def write_runmap(path=RUNMAP_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [dict(zip(("index", "match_arm", "lambda_loss", "w", "seed"),
                     (i, *map_run_idx(i)))) for i in range(N_RUNS)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote runmap ({N_RUNS} rows) -> {path}")


def _build_env(w, lam, out_scale):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": lam, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w,
                           output_scale=out_scale)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def _vectorise(env):
    vec = ss.pad_action_space_v0(env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class="stable_baselines3")
    vec.seed = lambda s: None
    return vec


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
    out_scale = _output_scale(arm, lam)
    np.random.seed(seed); warnings.filterwarnings("ignore")
    os.makedirs(RESULT_DIR, exist_ok=True)

    vec = _vectorise(_build_env(w, lam, out_scale))
    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, out_scale)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    diag = collect_reward_diagnostics(_build_env(w, lam, out_scale), model, 0.0,
                                      num_episodes=5, base_seed=seed * 100)

    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "match_arm": arm, "output_scale": out_scale,
           "lambda_loss": lam, "w": w, "seed": seed, "mean_S": mean_S, "profit": profit,
           "lost_sales": lost_sales, "collapsed": collapsed,
           "frac_gain": diag["frac_gain"], "frac_loss": diag["frac_loss"],
           "mean_cpt_reward": diag["mean_cpt_reward"], "reward_std": diag["reward_std"],
           "return_std": diag["return_std"], "final_value_loss": fvl,
           "final_explained_variance": fev}
    pd.DataFrame([row], columns=COLUMNS).to_csv(_result_path(run_idx), index=False)
    print(f"[ControlB] idx={run_idx} arm={arm} out_scale={out_scale:.3f} lam={lam} "
          f"w={w} seed={seed} -> S={mean_S:.1f} profit={profit:.1f} collapsed={collapsed}")
    return row


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


def pilot(timesteps, seeds=(42, 43, 44), ws=W_LIST, freeze=False):
    """Measure per-lambda RMS of the RAW CPT reward (output_scale=1) over the w-grid, at
    lambda in {1,3,7}. The stdmatched arm's output_scale(lambda) = rms(lambda)/rms(1), so
    every lambda's transformed reward has lambda=1's RMS. Freeze + commit before analysis."""
    warnings.filterwarnings("ignore")
    print(f"[pilot] measuring CPT-reward RMS per lambda, ws={ws}, seeds={seeds}")
    rms = {}
    for lam in LAMBDA_LIST:
        vals = []
        for w in ws:
            for seed in seeds:
                np.random.seed(seed)
                vec = _vectorise(_build_env(w, lam, 1.0))
                model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
                model.learn(total_timesteps=timesteps, callback=MetricsCallback())
                env = _build_env(w, lam, 1.0)
                obs, _ = env.reset(seed=seed * 7)
                steps = 0
                while env.agents and steps < 400:
                    acts = {}
                    for a in env.agents:
                        if a in obs:
                            act, _ = model.predict(obs[a], deterministic=True)
                            acts[a] = act[:env.action_space(a).shape[0]]
                    obs, r, term, trunc, info = env.step(acts)
                    for a in r:
                        if info[a].get("cpt_reward") is not None:
                            vals.append(info[a]["cpt_reward"])
                    steps += 1
        rms[lam] = float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
        print(f"  lambda={lam}: rms(CPT reward) = {rms[lam]:.3f}  (n={len(vals)})")
    ref = rms[1]
    out_scale = {str(lam): rms[lam] / ref for lam in LAMBDA_LIST}   # lambda=1 -> 1.0
    print(f"\n[pilot] rms per lambda: {rms}")
    if freeze:
        os.makedirs(os.path.dirname(RMS_JSON), exist_ok=True)
        with open(RMS_JSON, "w") as f:
            json.dump(out_scale, f, indent=2)
        print(f"[pilot] WROTE {RMS_JSON}. Record it as the pre-registration anchor BEFORE "
              "the main analysis.")
    else:
        print(f"[pilot] output_scale(lambda)=rms(lambda)/rms(1) to FREEZE at {RMS_JSON}:")
        print("        " + json.dumps(out_scale))
    return out_scale


def dry_run():
    warnings.filterwarnings("ignore")
    print("=== DRY RUN: output_scale RMS-matches the transformed reward ===")
    for out_scale in (1.0, 2.0, 3.0):
        raw = MultiAgentNetInvMgmt()
        interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
        p = {a: {"lambda": 7.0, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
        cpt = CPTRewardWrapper(interv, p, reward_scale=1.0, global_reward_weight=0.75,
                               output_scale=out_scale)
        cpt.reset(seed=42)
        acts = {a: cpt.action_space(a).sample() for a in cpt.agents}
        _o, _r, _t, _tr, info = cpt.step(acts)
        a0 = list(info)[0]
        print(f"out_scale={out_scale}: blended={info[a0]['blended_reward']:.2f} "
              f"cpt={info[a0]['cpt_reward']:.3f}")
    print("=== one tiny cell (2k steps, arm=raw) ===")
    vec = _vectorise(_build_env(0.75, 1, 1.0))
    m = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=42)
    m.learn(total_timesteps=2000, callback=MetricsCallback())
    diag = collect_reward_diagnostics(_build_env(0.75, 1, 1.0), m, 0.0, num_episodes=2, base_seed=1)
    print(f"  diagnostics: {diag}")
    print("Dry run OK.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, help=f"0..{N_RUNS-1}")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--write-runmap", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot_timesteps", type=int, default=150_000)
    ap.add_argument("--pilot_ws", default=None, help="comma list of w (default: full grid)")
    ap.add_argument("--pilot_seeds", default=None, help="comma list of seeds (default 42,43,44)")
    ap.add_argument("--freeze_rms", action="store_true",
                    help="with --pilot: write the measured RMS output_scale JSON")
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
        pilot(args.pilot_timesteps, seeds=seeds, ws=ws, freeze=args.freeze_rms); return
    if args.dry_run:
        dry_run(); return
    if args.run_idx is None:
        ap.error("one of --run_idx / --write-runmap / --merge / --pilot / --dry-run required")
    run_one(args.run_idx, args.timesteps)


if __name__ == "__main__":
    main()
