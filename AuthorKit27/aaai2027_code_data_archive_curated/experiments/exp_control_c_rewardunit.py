"""
Control C -- Reward-unit sensitivity.

Question: is w a literal, scale-independent "fraction of centralization", or does the
numerical w_crit depend on how the local/global rewards are parameterized? We multiply
the LOCAL or the GLOBAL reward by a fixed factor (2x) BEFORE the (1-w)/w blend -- i.e.
we change the local-to-global incentive RATIO -- and see whether w_crit moves.

Expected (and fine): the numerical w_crit CHANGES. That does not destroy the paper; it
establishes that the boundary is CONDITIONAL on reward parameterization. The honest
consequence: stop calling w a scale-independent "fraction of centralization"; call it the
NOMINAL reward-mixture coefficient, and report a normalized local-to-global incentive
ratio alongside it. The qualitative loss-aversion effect (w_crit rises with lambda) is
expected to persist within each arm.

Design (reduced, robustness only -- everything except the scale arm is identical to
exp_ppo_lambda_sweep / Control A):
  - Reward-unit arms (3):
      baseline : local_scale=1, global_scale=1  (the paper condition)
      local2x  : local_scale=2, global_scale=1  (double local reward before blend)
      global2x : local_scale=1, global_scale=2  (double global reward before blend)
  - lambda in {1, 3, 7};  w in {0.5,0.65,0.75,0.85,0.95} (IDENTICAL grid);
    seeds 42-61 (IDENTICAL); alpha=beta=0.88; PPO 300k.

The blend becomes  (1-w)*(c_local * r_local) + w*(c_global * r_global), applied inside
CPTRewardWrapper (local_scale/global_scale params, both default 1.0). No pre-registration
pilot needed -- the scale factors are fixed constants, not measured.

Same FREE diagnostics as Control A (frac_gain/loss, mean_cpt_reward, reward_std,
return_std) so the reward-magnitude story is auditable. Collapse label unchanged
(mean_S<10 and profit<=-127.1).

Array: 3 arms x 3 lambda x 5 w x 20 seeds = 900 cells (indices 0..899).
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from expA_interior_w import (evaluate_and_extract_metrics, MetricsCallback,
                             LOW_BOUND, HIGH_BOUND)
from exp_control_a_refpoint import collect_reward_diagnostics   # reuse the diagnostics
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

# --- Design ------------------------------------------------------------------
W_LIST      = [0.5, 0.65, 0.75, 0.85, 0.95]
LAMBDA_LIST = [1, 3, 7]
SEEDS       = list(range(42, 62))
ALPHA, BETA = 0.88, 0.88
# arm -> (local_scale, global_scale)
ARMS = {"baseline": (1.0, 1.0), "local2x": (2.0, 1.0), "global2x": (1.0, 2.0)}
ARM_NAMES = list(ARMS)
N_ARM, N_LAM, N_W, N_SEED = len(ARM_NAMES), len(LAMBDA_LIST), len(W_LIST), len(SEEDS)
N_RUNS = N_ARM * N_LAM * N_W * N_SEED             # 3 x 3 x 5 x 20 = 900

S_FLOOR, PROFIT_CEIL = 10.0, -127.1

FIG_DIR    = "docs/expControlC_rewardunit_figures"
RAW_CSV    = os.path.join(FIG_DIR, "control_c_rewardunit_raw_data.csv")
RUNMAP_CSV = os.path.join(FIG_DIR, "runmap_control_c.csv")
RESULT_DIR = "results/expControlC"

COLUMNS = ["run_idx", "scale_arm", "local_scale", "global_scale", "lambda_loss", "w",
           "seed", "mean_S", "profit", "lost_sales", "collapsed",
           "frac_gain", "frac_loss", "mean_cpt_reward", "reward_std", "return_std",
           "final_value_loss", "final_explained_variance"]


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
    rows = [dict(zip(("index", "scale_arm", "lambda_loss", "w", "seed"),
                     (i, *map_run_idx(i)))) for i in range(N_RUNS)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote runmap ({N_RUNS} rows) -> {path}")


def _build_env(w, lam, c_local, c_global):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": lam, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w,
                           local_scale=c_local, global_scale=c_global)
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
    c_local, c_global = ARMS[arm]
    np.random.seed(seed); warnings.filterwarnings("ignore")
    os.makedirs(RESULT_DIR, exist_ok=True)

    vec = _vectorise(_build_env(w, lam, c_local, c_global))
    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    eval_env = _build_env(w, lam, c_local, c_global)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    # diagnostics relative to reference point 0 (Control C does not move z0)
    diag = collect_reward_diagnostics(_build_env(w, lam, c_local, c_global), model, 0.0,
                                      num_episodes=5, base_seed=seed * 100)

    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "scale_arm": arm, "local_scale": c_local,
           "global_scale": c_global, "lambda_loss": lam, "w": w, "seed": seed,
           "mean_S": mean_S, "profit": profit, "lost_sales": lost_sales,
           "collapsed": collapsed, "frac_gain": diag["frac_gain"],
           "frac_loss": diag["frac_loss"], "mean_cpt_reward": diag["mean_cpt_reward"],
           "reward_std": diag["reward_std"], "return_std": diag["return_std"],
           "final_value_loss": fvl, "final_explained_variance": fev}
    pd.DataFrame([row], columns=COLUMNS).to_csv(_result_path(run_idx), index=False)
    print(f"[ControlC] idx={run_idx} arm={arm} (cl={c_local},cg={c_global}) lam={lam} "
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


def dry_run():
    warnings.filterwarnings("ignore")
    print("=== DRY RUN: reward-unit scaling changes the blend BEFORE transform ===")
    for arm, (cl, cg) in ARMS.items():
        raw = MultiAgentNetInvMgmt()
        interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
        p = {a: {"lambda": 7.0, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
        cpt = CPTRewardWrapper(interv, p, reward_scale=1.0, global_reward_weight=0.75,
                               local_scale=cl, global_scale=cg)
        cpt.reset(seed=42)
        acts = {a: cpt.action_space(a).sample() for a in cpt.agents}
        _o, _r, _t, _tr, info = cpt.step(acts)
        a0 = list(info)[0]
        print(f"{arm:>9} (cl={cl},cg={cg}): local={info[a0]['raw_local_reward']:.2f} "
              f"global={info[a0]['raw_global_reward']:.2f} blended={info[a0]['blended_reward']:.2f} "
              f"cpt={info[a0]['cpt_reward']:.2f}")
    print("=== one tiny cell (2k steps, baseline) ===")
    vec = _vectorise(_build_env(0.75, 1, 1.0, 1.0))
    m = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=42)
    m.learn(total_timesteps=2000, callback=MetricsCallback())
    diag = collect_reward_diagnostics(_build_env(0.75, 1, 1.0, 1.0), m, 0.0, num_episodes=2, base_seed=1)
    print(f"  diagnostics: {diag}")
    print("Dry run OK.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, help=f"0..{N_RUNS-1}")
    ap.add_argument("--timesteps", type=int, default=300_000)
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
    run_one(args.run_idx, args.timesteps)


if __name__ == "__main__":
    main()
