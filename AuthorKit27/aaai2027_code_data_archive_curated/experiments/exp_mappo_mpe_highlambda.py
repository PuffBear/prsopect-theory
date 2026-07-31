"""
Exp MAPPO-MPE high-lambda -- extends the MPE decomposition to high loss aversion.

WHY THIS EXISTS
---------------
The existing MPE/MAPPO decomposition (exp_mappo_mpe.py) sweeps only
{null, curvature, lambda3}. It replicates the *curvature* (facilitation) effect
(excess ~= -0.059, correct sign) but does NOT establish the *loss-aversion*
(impediment) direction that is the supply-chain headline: at lambda=3 the MPE
excess is ~= -0.018 (indistinguishable from null). To test whether loss aversion
shifts w_crit RIGHT in a second environment, we need high lambda (5, 7) -- the
regime where the supply-chain effect is large (+0.166 at lambda=7).

This driver adds exactly the two missing conditions (lambda5, lambda7) with the
SAME wrapper stack, training config, env builder, eval, and collapse metric as
exp_mappo_mpe.py, so the new rows merge cleanly with the existing
mappo_mpe_raw_data.csv for a five-condition decomposition
{null, curvature, lambda3, lambda5, lambda7}.

GRID
----
    w          in {0.0, 0.1, ..., 1.0}      (11 levels, imported)
    condition  in {lambda5, lambda7}        (alpha=beta=0.88; NEW)
    seeds      42-61                         (20 seeds, imported; AAAI rev.)
    Total: 11 x 2 x 20 = 440 runs   ->  PBS array 0..439

NOTE ON TRAINING BUDGET: MPE's null boundary previously measured ~1.0, partly an
under-training artifact at the coordination end (w=1 reached only 0.26 occupancy
vs 0.64 ceiling at 2M steps; see exp_ppo_mpe.py v4 notes). Run this at the v4
budget (>=4M steps) so a rightward shift is detectable rather than masked by an
under-trained w=1 end. Default --timesteps is 4_000_000.
"""
import os
import sys
import argparse
import warnings
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env.mpe_coop_nav import is_collapsed_mpe_occupancy, HERDING_SCALE
# Reuse the EXACT machinery from the base MAPPO-MPE experiment so nothing drifts.
from exp_mappo_mpe import train_mappo_model, _build_env
from exp_ppo_mpe import W_LIST, SEEDS, COLUMNS, evaluate_mpe, _append_row  # noqa

# --- NEW conditions only (merge with existing null/curvature/lambda3 CSV) ----
#   name        alpha  beta  lambda
CONDITIONS_HL = [
    ("lambda5", 0.88, 0.88, 5.0),
    ("lambda7", 0.88, 0.88, 7.0),
]
N_W, N_COND, N_SEED = len(W_LIST), len(CONDITIONS_HL), len(SEEDS)
N_RUNS = N_W * N_COND * N_SEED  # 11 * 2 * 20 = 440  (20 seeds; AAAI rev.)


def map_run_idx(run_idx):
    """run_idx -> (w, condition_tuple, seed). Layout: w x condition x seed."""
    block = N_COND * N_SEED
    w_idx = run_idx // block
    rem = run_idx % block
    c_idx = rem // N_SEED
    seed_idx = rem % N_SEED
    return W_LIST[w_idx], CONDITIONS_HL[c_idx], SEEDS[seed_idx]


def run_one(run_idx, timesteps, out_path, herding_scale=HERDING_SCALE):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0, {N_RUNS-1}], got {run_idx}")
    w, (cond, alpha, beta, lam), seed = map_run_idx(run_idx)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    model, cb = train_mappo_model(w, alpha, beta, lam, seed, timesteps,
                                  herding_scale=herding_scale)
    eval_env = _build_env(w, lam, alpha, beta, herding_scale=herding_scale)
    mean_reward, mean_occ = evaluate_mpe(model, eval_env, num_episodes=10,
                                         base_seed=seed * 10)
    collapsed = is_collapsed_mpe_occupancy(mean_occ)
    fvl = float(np.mean(cb.value_losses)) if cb.value_losses else float("nan")
    fev = float(np.mean(cb.explained_variances)) if cb.explained_variances else float("nan")

    row = {"run_idx": run_idx, "w": w, "condition": cond,
           "alpha": alpha, "beta": beta, "lambda_loss": lam, "seed": seed,
           "algo": "MAPPO", "env": "MPE_CoopNav",
           "mean_reward": mean_reward, "mean_occupancy": mean_occ,
           "collapsed": collapsed,
           "final_value_loss": fvl, "final_explained_variance": fev}
    _append_row(row, out_path)
    print(f"[MAPPO-MPE-HL] run_idx={run_idx} w={w} cond={cond} lam={lam} "
          f"seed={seed} -> r_global={mean_reward:.1f} occ={mean_occ:.2f} "
          f"collapsed={collapsed}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help="0..%d" % (N_RUNS - 1))
    ap.add_argument("--timesteps", type=int, default=4_000_000)
    ap.add_argument("--herding_scale", type=float, default=HERDING_SCALE)
    ap.add_argument("--out", default="docs/expMAPPO_mpe_highlambda_figures/mappo_mpe_highlambda_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.out, herding_scale=args.herding_scale)


if __name__ == "__main__":
    main()
