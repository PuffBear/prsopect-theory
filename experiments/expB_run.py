"""
Experiment B runner - the 2-D (w, lambda) collapse surface (docs/expB_preregistration.md).

Reuses the validated _run_single from expA_interior_w (identical substrate, eval,
and frozen collapse label) over the outer product of the extended w-grid (amendment
A1) x the pre-registered lambda-grid x seeds. One labeled row per run, incremental save.

Default grid (pre-reg + amendment A1):
    w  in  {0,0.2,0.4,0.6,0.7,0.8,0.85,0.9,0.95,1.0}   (10 levels; 0.85/0.95 straddle w_crit=0.82)
    lambda  in  {1,2,3,5,7,10}                                (6 levels)
    seeds 42-56                                       (15 seeds)
  -> 10 x 6 x 15 = 900 runs @ 150k.

Analyze with: python experiments/analyze_expB.py --in docs/expB_figures/expB_raw_data.csv
"""
import os
import sys
import argparse
import pandas as pd
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from expA_interior_w import _run_single, parse_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="0.0,0.2,0.4,0.6,0.7,0.8,0.85,0.9,0.95,1.0")
    ap.add_argument("--lams", default="1,2,3,5,7,10")
    ap.add_argument("--seeds", default="42-56")  # 15 seeds
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--procs", type=int, default=min(mp.cpu_count(), 10))
    ap.add_argument("--out", default="docs/expB_figures/expB_raw_data.csv")
    args = ap.parse_args()

    ws = [float(x) for x in args.ws.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    seeds = parse_seeds(args.seeds)
    jobs = [(w, lam, s, args.timesteps) for w in ws for lam in lams for s in seeds]

    print(f"Experiment B: {len(ws)} w x {len(lams)} lambda x {len(seeds)} seeds = {len(jobs)} runs "
          f"@ {args.timesteps} steps, procs={args.procs}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    with mp.Pool(processes=args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_single, jobs), 1):
            rows.append(r)
            print(f"[{i:3d}/{len(jobs)}] w={r['w']:.2f} lambda={r['lambda']:.0f} seed={r['seed']:2d} -> "
                  f"S={r['mean_S']:6.1f} profit={r['profit']:7.1f} collapsed={r['collapsed']}")
            pd.DataFrame(rows).to_csv(args.out, index=False)

    pd.DataFrame(rows).sort_values(["w", "lambda", "seed"]).to_csv(args.out, index=False)
    print(f"\nSaved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
