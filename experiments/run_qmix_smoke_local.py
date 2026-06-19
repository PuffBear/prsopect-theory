"""
Local smoke test for the value-based (QMIX / IQL) cells -- run on your Mac BEFORE
rsync'ing anything to the HPC. Three gates, in increasing cost:

    PART 0  wiring check   : action space = Discrete(K)/(5); buffer reward ==
                             CPT(blend(w)); QMIX mixer reads the joint state and
                             is monotone; IQL has no mixer.   (~seconds)
    PART 1  toy sanity     : the learner solves a known-good 2-agent coordination
                             matrix (greedy joint action -> the unique optimum).
                             This is the "is the algorithm actually learning"
                             gate that the env wiring check cannot give.  (~seconds)
    PART 2  short train run : a real 20k-env-step run on the chosen (algo, env)
                             cell, asserting it produces a finite metric row and
                             a finite value loss (no NaNs / crashes in the real
                             wrapper stack + eval path).   (~1-3 min)

Usage:
    python experiments/run_qmix_smoke_local.py --algo qmix --env supply
    python experiments/run_qmix_smoke_local.py --algo iql  --env mpe
    python experiments/run_qmix_smoke_local.py --algo qmix --env supply --skip-train

A clean run ends with  ALL SMOKE CHECKS PASS.
"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["qmix", "iql"], default="qmix")
    ap.add_argument("--env", choices=["supply", "mpe"], default="supply")
    ap.add_argument("--timesteps", type=int, default=20_000)
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    import numpy as np

    print(f"=== PART 0: wiring check ({args.algo}, {args.env}) ===")
    if args.env == "supply":
        import exp_qmix_orgym as drv
        drv.wiring_check(args.algo, K=args.K)
    else:
        import exp_qmix_mpe as drv
        drv.wiring_check(args.algo)

    print("\n=== PART 1: toy sanity (known-good 2-agent coordination) ===")
    import run_qmix_toy_sanity as toy
    toy_ok = all([
        sum(toy.check(True, seed=s) for s in (0, 1, 2)) >= 2,
        sum(toy.check(False, seed=s) for s in (0, 1, 2)) >= 2,
    ])
    if not toy_ok:
        print("  TOY SANITY FAILED -- learner is not converging on a known-good "
              "problem. Do NOT submit to the cluster.")
        sys.exit(1)
    print("  toy sanity: PASS")

    if args.skip_train:
        print("\n(skipped PART 2; wiring + toy only)")
        print("\nWIRING + TOY PASS")
        return

    print(f"\n=== PART 2: short {args.timesteps}-step train run ({args.algo}, {args.env}) ===")
    from agents.qmix_rollout import make_learner, train
    use_mixer = (args.algo == "qmix")
    if args.env == "supply":
        env = drv._build_env(0.75, 7, 0.88, 0.88, K=args.K)
        learner = make_learner(env, use_mixer=use_mixer, seed=42)
        train(env, learner, total_env_steps=args.timesteps, seed=42, verbose=True)
        eval_env = drv._build_env(0.75, 7, 0.88, 0.88, K=args.K)
        mean_S, profit, lost = drv.evaluate_supply(learner, eval_env,
                                                   num_episodes=5, base_seed=420)
        print(f"  trained 20k: mean_S={mean_S:.1f} profit={profit:.1f} "
              f"lost_sales={lost:.3f} value_loss={learner.last_value_loss:.4f}")
        finite = all(np.isfinite([mean_S, profit, lost, learner.last_value_loss]))
    else:
        env = drv._build_env(0.5, 7, 0.88, 0.88)
        learner = make_learner(env, use_mixer=use_mixer, seed=42)
        train(env, learner, total_env_steps=args.timesteps, seed=42, verbose=True)
        eval_env = drv._build_env(0.5, 7, 0.88, 0.88)
        r, occ = drv.evaluate_mpe_discrete(learner, eval_env, num_episodes=5,
                                           base_seed=420)
        print(f"  trained 20k: r_global={r:.1f} occupancy={occ:.3f} "
              f"value_loss={learner.last_value_loss:.4f}")
        finite = all(np.isfinite([r, occ, learner.last_value_loss]))

    if not finite:
        print("  PART 2 FAILED: non-finite metric / value loss.")
        sys.exit(1)
    print("  PART 2: PASS (finite metrics, no crash)")
    print("\nALL SMOKE CHECKS PASS")


if __name__ == "__main__":
    main()
