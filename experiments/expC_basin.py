"""
Experiment C - basin-of-attraction / initialization control (experiment_spec_v1 sec.C).

Elevated by the Exp A review: the horizon check showed transition-collapse is
metastable convergence, and the critic diagnostic showed it is not a learning
failure. C asks the mechanism question directly: at a fixed 150k budget, does the
converged policy depend on where it is INITIALIZED (S_init), or only on w?

Uses OffsetBaseStockWrapper: S = clip(S_init + action*action_scale, 0, 500), so the
action mapping is centered on S_init. Converged S = mean of last 10k logged steps
(as in phase3_96). Collapse label = frozen definition on eval profit + converged S.

w-levels span the MEASURED transition (w_crit=0.82): {0.5 (deep collapse), 0.85
(~w_crit), 1.0 (recovery)}. lambda=1 (no loss aversion; C isolates initialization).

    python experiments/expC_basin.py            # default grid
Analyze/plot: converged_S vs S_init per w (see __main__ summary + figure).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings

import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.offset_base_stock_wrapper import OffsetBaseStockWrapper
from expA_interior_w import evaluate_and_extract_metrics, is_collapsed

ACTION_SCALE = 250.0


def _run_single(args):
    w, S_init, seed, timesteps = args
    np.random.seed(seed)
    warnings.filterwarnings("ignore")
    log_file = f"/tmp/expC_offset_{w}_{S_init}_{seed}.csv"
    if os.path.exists(log_file):
        os.remove(log_file)

    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    train_env = OffsetBaseStockWrapper(cpt, S_init=S_init, action_scale=ACTION_SCALE, log_file=log_file)

    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=timesteps)
    train_env.close()  # flush log buffer

    # Converged S = mean of last 10k logged steps for node_2 (phase3_96 convention)
    converged_S = np.nan
    if os.path.exists(log_file):
        d = pd.read_csv(log_file)
        d2 = d[d["agent"] == "node_2"].sort_values("step")
        if len(d2):
            converged_S = d2.tail(10000)["S"].mean()

    # Eval for profit. NOTE: evaluate_and_extract_metrics computes its mean_S with
    # the SCALED [0,500] mapping, which is WRONG for the offset substrate, so we do
    # NOT use its mean_S here. profit (true env reward) is correct. The collapse
    # label uses converged_S (from the offset log, correct mapping) + profit.
    eval_raw = MultiAgentNetInvMgmt()
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=["node_1"])
    eval_cpt = CPTRewardWrapper(eval_interv, params, reward_scale=1.0, global_reward_weight=w)
    eval_env = OffsetBaseStockWrapper(eval_cpt, S_init=S_init, action_scale=ACTION_SCALE)
    _scaled_mean_S, _, _, _, profit, _ = evaluate_and_extract_metrics(model, eval_env, num_episodes=10, base_seed=seed * 10)

    if os.path.exists(log_file):
        os.remove(log_file)
    return {"w": w, "S_init": S_init, "seed": seed, "converged_S": converged_S,
            "profit": profit, "collapsed": is_collapsed(converged_S, profit)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="0.5,0.85,1.0")
    ap.add_argument("--sinits", default="0,50,150,250")
    ap.add_argument("--seeds", default="42-53")  # 12 seeds
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--procs", type=int, default=min(mp.cpu_count(), 10))
    ap.add_argument("--out", default="docs/expC_figures/expC_raw_data.csv")
    args = ap.parse_args()

    ws = [float(x) for x in args.ws.split(",")]
    sinits = [float(x) for x in args.sinits.split(",")]
    a, b = (args.seeds.split("-") if "-" in args.seeds else (None, None))
    seeds = list(range(int(a), int(b) + 1)) if a else [int(x) for x in args.seeds.split(",")]
    jobs = [(w, si, s, args.timesteps) for w in ws for si in sinits for s in seeds]

    print(f"Experiment C: {len(ws)} w x {len(sinits)} S_init x {len(seeds)} seeds = {len(jobs)} runs @ {args.timesteps}")
    os.makedirs("docs/expC_figures", exist_ok=True)

    rows = []
    with mp.Pool(processes=args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_single, jobs), 1):
            rows.append(r)
            print(f"[{i:3d}/{len(jobs)}] w={r['w']:.2f} S_init={r['S_init']:.0f} seed={r['seed']:2d} -> "
                  f"convS={r['converged_S']:6.1f} collapsed={r['collapsed']}")
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # Summary: does converged_S depend on S_init within each w?
    print("\n=== Exp C: converged_S by (w, S_init) ===")
    print(df.groupby(["w", "S_init"])["converged_S"].agg(["mean", "std"]).to_string())
    print("\nIf converged_S is ~flat across S_init within a w -> w-driven (not initialization).")
    print("If it tracks S_init -> bistable / initialization-driven (basin).")

    # Figure: converged_S vs S_init, one line per w
    fig, ax = plt.subplots(figsize=(8, 6))
    for w in ws:
        g = df[df["w"] == w].groupby("S_init")["converged_S"].mean()
        ax.plot(g.index, g.values, marker="o", label=f"w={w}")
    ax.plot([0, 250], [0, 250], "k:", alpha=0.4, label="converged=init (pure basin)")
    ax.set_xlabel("S_init (initialization)")
    ax.set_ylabel("converged S (last-10k mean)")
    ax.set_title("Exp C: initialization control - converged S vs S_init by w")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/expC_figures/expC_basin.png", dpi=150)
    print("\nSaved docs/expC_figures/expC_basin.png")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
