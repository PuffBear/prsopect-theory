import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper
from experiments.phase3_7_verification import evaluate_and_extract_metrics

def _run_single_config(args):
    seed, low, high, alpha = args
    np.random.seed(seed)
    
    log_file = f"/tmp/incentives_{int(low)}_{int(high)}_{alpha}_{seed}.csv"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    raw_env = MultiAgentNetInvMgmt()
    scripted_nodes = ["node_1"]
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
    
    # Crucial mechanism test: global_reward_weight = alpha
    cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=alpha)
    train_env = ScaledBaseStockWrapper(cpt_env, low_bound=low, high_bound=high, log_file=log_file)
    
    vec_env = ss.pad_action_space_v0(train_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(vec_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=0, base_class='stable_baselines3')
    vec_env.seed = lambda s: None
    
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=150000)
    
    eval_raw = MultiAgentNetInvMgmt()
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
    eval_cpt = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=alpha)
    eval_env = ScaledBaseStockWrapper(eval_cpt, low_bound=low, high_bound=high)
    
    o_qty, inv_lvl, ls_ratio, ep_rew, bw = evaluate_and_extract_metrics(model, eval_env, num_episodes=5)
    return seed, low, high, alpha, o_qty, inv_lvl, ls_ratio, ep_rew, bw

def run_factorial_sweep():
    bounds = [(0.0, 500.0), (200.0, 700.0)]
    alphas = [0.0, 0.25, 0.5, 1.0]
    seeds = list(range(42, 47)) # 5 seeds per config
    
    args_list = []
    for low, high in bounds:
        for alpha in alphas:
            for seed in seeds:
                args_list.append((seed, low, high, alpha))
                
    print(f"\n=======================================================")
    print(f"Starting Phase 3.8 Factorial Sweep (Total Runs: {len(args_list)})")
    print(f"=======================================================")
    os.makedirs("docs/phase3_8_figures", exist_ok=True)

    results = {}
    rows = []  # per-run rows for CSV (added to fix the no-saved-table bug)

    # Frozen collapse definition (docs/collapse_definition.md)
    def _collapsed(mean_S, profit):
        return int((mean_S < 10.0) and (profit <= -128.1 + 1.0))

    with mp.Pool(processes=min(mp.cpu_count(), 10)) as pool:
        for res in pool.imap_unordered(_run_single_config, args_list):
            seed, low, high, alpha, o_qty, inv_lvl, ls_ratio, ep_rew, bw = res

            config_key = (low, high, alpha)
            if config_key not in results:
                results[config_key] = []
            results[config_key].append((o_qty, inv_lvl, ls_ratio, ep_rew, bw))

            # NOTE: this script reports mean order quantity, not learned S, so the
            # collapse label here uses order_qty < 10 as the S-proxy; for a clean
            # S-based label use expA_interior_w.py.
            rows.append({
                "low": low, "high": high, "w": alpha, "seed": seed,
                "mean_order": o_qty, "mean_inventory": inv_lvl,
                "lost_sales": ls_ratio, "profit": ep_rew, "bullwhip": bw,
                "collapsed_proxy": _collapsed(o_qty, ep_rew),
            })
            pd.DataFrame(rows).to_csv("docs/phase3_8_figures/phase3_8_factorial_data.csv", index=False)

            print(f"Bound:[{int(low)},{int(high)}] Alpha:{alpha} Seed:{seed} -> Profit:{ep_rew:.1f}, Qty:{o_qty:.1f}, Inv:{inv_lvl:.1f}, LS:{ls_ratio:.1%}")

    print("\n=== Phase 3.8 Incentive Validation Results ===")
    for low, high in bounds:
        for alpha in alphas:
            res_list = results[(low, high, alpha)]
            o_qtys, inv_lvls, ls_ratios, ep_rews, bws = zip(*res_list)
            print(f"Bounds: [{int(low)}, {int(high)}] | Alpha (Sharing): {alpha}")
            print(f"  Profit : {np.mean(ep_rews):.2f} +/- {np.std(ep_rews):.2f}")
            print(f"  Inv Lvl: {np.mean(inv_lvls):.2f} +/- {np.std(inv_lvls):.2f}")
            print(f"  LS %   : {np.mean(ls_ratios):.2%} +/- {np.std(ls_ratios):.2%}")
            print("----------------------------------------------")
            
def plot_factorial_trajectories():
    os.makedirs("docs/phase3_8_figures", exist_ok=True)
    
    bounds = [(0.0, 500.0), (200.0, 700.0)]
    alphas = [0.0, 0.25, 0.5, 1.0]
    seeds = list(range(42, 47))
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), sharey='row')
    
    for row_idx, (low, high) in enumerate(bounds):
        for col_idx, alpha in enumerate(alphas):
            ax = axes[row_idx, col_idx]
            
            for seed in seeds:
                log_file = f"/tmp/incentives_{int(low)}_{int(high)}_{alpha}_{seed}.csv"
                if os.path.exists(log_file):
                    df = pd.read_csv(log_file)
                    df_agent = df[df['agent'] == 'node_2'].sort_values('step')
                    df_agent['S_smooth'] = df_agent['S'].rolling(1000).mean()
                    ax.plot(df_agent['step'], df_agent['S_smooth'], alpha=0.8, linewidth=1.5, label=f"Seed {seed}")
            
            ax.set_title(f"Bounds: [{int(low)}, {int(high)}] | alpha={alpha}")
            ax.axhline(y=low, color='r', linestyle='--', alpha=0.5, label='Lower Bound')
            ax.axhline(y=high, color='g', linestyle='--', alpha=0.5, label='Upper Bound')
            ax.set_ylim(low - 50, high + 50)
            
            if row_idx == 1:
                ax.set_xlabel("Training Step")
            if col_idx == 0:
                ax.set_ylabel("Learned Base-Stock (S)")
                
    plt.tight_layout()
    plt.savefig("docs/phase3_8_figures/backlog_sharing_trajectories.png", dpi=300)
    print("Saved docs/phase3_8_figures/backlog_sharing_trajectories.png")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    run_factorial_sweep()
    plot_factorial_trajectories()
