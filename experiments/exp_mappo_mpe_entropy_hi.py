"""
Exp MAPPO-MPE entropy ablation -- HIGH-entropy probe.

WHY (highest-effort follow-up on the MPE exploration-noise null):
The base entropy ablation (ent_coef in {0,0.001,0.01,0.05}) found NO w_crit shift
on MPE (+0.026, CIs overlap). A four-hypothesis diagnostic showed this is NOT a
theta artifact and NOT a ceiling artifact: the boundary is interior (room to move)
but the entropy knob bites only WEAKLY in 0..0.05 -- explained_variance falls just
0.846 -> 0.748 -- and the resulting occupancy shift at the transition (~0.03) is
SMALLER than the across-seed std (~0.06). So the effect, if real, sits below the
seed-noise floor at ent<=0.05.

This probe pushes the entropy term 2-3x harder, into the range where it should
bite hard enough to clear the noise floor, but stays BELOW the degenerate regime
(very high ent -> near-uniform categorical policy -> coordination collapses for
ALL w, which would measure 'entropy destroyed coordination', not a w_crit shift).
The grid {0.05, 0.08, 0.12} brackets that window; 0.05 overlaps the base sweep as
a calibration anchor.

DEGENERACY GUARD: if at a given ent_coef the w=1 (coordinated) occupancy collapses
to the w=0 level, that ent has randomized the policy -- the analyzer flags it and
excludes it from the w_crit trend (a degenerate point is not a boundary shift).

GRID
----
  w        in {0.0, 0.1, ..., 1.0}        (11)
  ent_coef in {0.05, 0.08, 0.12}          (3; 0.05 overlaps base sweep)
  seeds    42-61                          (20)
  Total: 11 x 3 x 20 = 660 runs   ->  PBS array 0..659

    python experiments/exp_mappo_mpe_entropy_hi.py --wiring_check
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

from exp_mappo_mpe_entropy import train_fixed_entropy   # fixed ent_coef, NO decay
from exp_mappo_mpe import _build_env
from exp_ppo_mpe import evaluate_mpe
from env.mpe_coop_nav import is_collapsed_mpe_occupancy, HERDING_SCALE

W_LIST = [round(0.1 * i, 1) for i in range(11)]
ENT_COEFS = [0.05, 0.08, 0.12]              # high probe (0.05 overlaps base anchor)
SEEDS = list(range(42, 62))
ALPHA, BETA, LAMBDA = 0.88, 0.88, 1.0
HORIZON = 4_000_000
N_W, N_ENT, N_SEED = len(W_LIST), len(ENT_COEFS), len(SEEDS)
N_RUNS = N_W * N_ENT * N_SEED               # 660

COLUMNS = ["run_idx", "w", "ent_coef", "alpha", "beta", "lambda_loss", "seed",
           "algo", "env", "mean_reward", "mean_occupancy", "collapsed",
           "final_value_loss", "final_explained_variance"]


def _append_row(row, out_path, columns):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        write_header = not os.path.exists(out_path)
        pd.DataFrame([row], columns=columns).to_csv(
            out_path, mode="a", header=write_header, index=False)


def map_run_idx(run_idx):
    """w (outer) x ent_coef (middle) x seed (inner)."""
    block = N_ENT * N_SEED
    w_idx = run_idx // block
    rem = run_idx % block
    e_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], ENT_COEFS[e_idx], SEEDS[seed_idx]


def run_one(run_idx, out_path, herding_scale=HERDING_SCALE):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}], got {run_idx}")
    w, ent, seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, cb = train_fixed_entropy(w, ALPHA, BETA, LAMBDA, seed, HORIZON, ent,
                                    herding_scale=herding_scale)
    eval_env = _build_env(w, LAMBDA, ALPHA, BETA, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe(model, eval_env, num_episodes=10,
                                         base_seed=seed * 10)
    collapsed = is_collapsed_mpe_occupancy(mean_occ)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "ent_coef": ent, "alpha": ALPHA,
           "beta": BETA, "lambda_loss": LAMBDA, "seed": seed, "algo": "MAPPO",
           "env": "MPE_CoopNav", "mean_reward": mean_reward,
           "mean_occupancy": mean_occ, "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path, COLUMNS)
    print(f"[MAPPO-MPE-entropy-hi] run_idx={run_idx} w={w} ent_coef={ent} seed={seed} "
          f"-> occ={mean_occ:.3f} collapsed={collapsed}", flush=True)
    return row


def wiring_check():
    from agents.mappo_policy import _CentralCriticExtractor
    model, _ = train_fixed_entropy(0.5, ALPHA, BETA, LAMBDA, seed=42,
                                   timesteps=0, ent_coef=0.12)
    assert isinstance(model.policy.vf_features_extractor, _CentralCriticExtractor), \
        "vf extractor is NOT the centralized critic!"
    assert abs(float(model.ent_coef) - 0.12) < 1e-9, \
        f"ent_coef not fixed at 0.12 (got {model.ent_coef}) -- decay may be active!"
    print(f"  centralized critic wired; ent_coef FIXED at {model.ent_coef} (no decay)")
    print(f"  WIRING CHECK PASS  (MPE entropy HI: {N_RUNS} runs, ent in {ENT_COEFS})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, default=None, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--wiring_check", action="store_true")
    ap.add_argument("--out", default="docs/expMAPPO_mpe_entropy_figures/mappo_mpe_entropy_hi_raw_data.csv")
    args = ap.parse_args()
    if args.wiring_check:
        wiring_check(); return
    if args.run_idx is None:
        ap.error("--run_idx is required (or use --wiring_check)")
    run_one(args.run_idx, args.out)


if __name__ == "__main__":
    main()
