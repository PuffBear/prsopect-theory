import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

def evaluate_S(args):
    target_S, alpha, num_episodes = args
    
    # We use a fixed base seed for reproducibility
    base_seed = 42
    
    total_local_profit = 0.0
    total_global_profit = 0.0
    total_ppo_obj = 0.0
    total_ls_ratio = 0.0
    
    # Range used in Phase 3.9
    low_bound = 0.0
    high_bound = 500.0
    
    # Normalize S to [-1, 1] for the ScaledBaseStockWrapper
    # a = 2 * (S - low) / (high - low) - 1
    action_val = 2.0 * (target_S - low_bound) / (high_bound - low_bound) - 1.0
    
    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        np.random.seed(ep_seed)
        
        raw_env = MultiAgentNetInvMgmt()
        scripted_nodes = ["node_1"]
        interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
        
        # lambda=1.0 ensures no PT shaping, only the alpha sharing
        agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
        cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=alpha)
        
        # Note: log_file is omitted so it doesn't write to CSV and slow things down
        env = ScaledBaseStockWrapper(cpt_env, low_bound=low_bound, high_bound=high_bound)
        
        obs, _ = env.reset(seed=ep_seed)
        
        ep_local_profit = 0.0
        ep_global_profit = 0.0
        ep_ppo_obj = 0.0
        
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    # Provide fixed deterministic action
                    real_shape = env.action_space(agent).shape[0]
                    actions[agent] = np.array([action_val] * real_shape, dtype=np.float32)
                    
            obs, rewards, term, trunc, info = env.step(actions)
            
            # Record PPO Objective (what the RL agent actually maximizes)
            if "node_2" in rewards:
                ep_ppo_obj += rewards["node_2"]
            
            # Record local unshaped profit for node_2
            if "node_2" in info:
                ep_local_profit += info["node_2"]["true_reward"]
                
            # Record global unshaped profit
            for agent in rewards:
                ep_global_profit += info[agent]["true_reward"]
                
        # Calculate lost sales
        curr_env = env
        while hasattr(curr_env, 'env') or hasattr(curr_env, 'unwrapped_env'):
            if hasattr(curr_env, 'unwrapped_env'):
                base_env = curr_env.unwrapped_env
                break
            curr_env = curr_env.env
            
        total_D = base_env.D.sum().sum()
        total_U = base_env.U.sum().sum()
        ls_ratio = total_U / total_D if total_D > 0 else 0
        
        total_local_profit += ep_local_profit
        total_global_profit += ep_global_profit
        total_ppo_obj += ep_ppo_obj
        total_ls_ratio += ls_ratio
        
    return (
        target_S, 
        alpha, 
        total_local_profit / num_episodes,
        total_global_profit / num_episodes,
        total_ppo_obj / num_episodes,
        total_ls_ratio / num_episodes
    )

def run_landscape_audit():
    print(f"\n=======================================================")
    print(f"Starting Phase 3.95: Reward Landscape Audit")
    print(f"=======================================================")
    
    S_values = list(range(0, 301, 10))
    alphas = [0.0, 1.0]
    num_episodes = 50
    
    args_list = []
    for alpha in alphas:
        for S in S_values:
            args_list.append((S, alpha, num_episodes))
            
    results = {}
    
    with mp.Pool(processes=min(mp.cpu_count(), 10)) as pool:
        for res in pool.imap_unordered(evaluate_S, args_list):
            S, alpha, loc_prof, glob_prof, ppo_obj, ls_ratio = res
            print(f"Alpha:{alpha:.1f} S:{S:03d} -> LocalProfit:{loc_prof:6.1f} | GlobProfit:{glob_prof:6.1f} | PPO_Obj:{ppo_obj:6.1f} | LS:{ls_ratio:5.1%}")
            
            if alpha not in results:
                results[alpha] = []
            results[alpha].append((S, loc_prof, glob_prof, ppo_obj, ls_ratio))
            
    # Sort results
    for alpha in alphas:
        results[alpha] = sorted(results[alpha], key=lambda x: x[0])
        
    plot_landscape(results)

def plot_landscape(results):
    os.makedirs("docs/phase3_95_figures", exist_ok=True)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    for idx, alpha in enumerate([0.0, 1.0]):
        ax = axes[idx]
        res_list = results[alpha]
        
        S_vals = [r[0] for r in res_list]
        loc_profits = [r[1] for r in res_list]
        glob_profits = [r[2] for r in res_list]
        ppo_objs = [r[3] for r in res_list]
        
        # Plot Global Profit (True Economic Benchmark)
        ax.plot(S_vals, glob_profits, 'g--', linewidth=2, label="Global Profit (Unshaped)")
        
        # Plot PPO Objective
        ax.plot(S_vals, ppo_objs, 'b-', linewidth=3, label="PPO Objective (What it maximizes)")
        
        # Find peak
        best_S = S_vals[np.argmax(ppo_objs)]
        best_obj = np.max(ppo_objs)
        ax.axvline(x=best_S, color='r', linestyle=':', label=f"Optimum S={best_S}")
        ax.plot(best_S, best_obj, 'r*', markersize=12)
        
        title = "Decentralized (Alpha=0.0)" if alpha == 0.0 else "Centralized (Alpha=1.0)"
        ax.set_title(f"{title}\nOptimum Base-Stock S = {best_S}")
        ax.set_xlabel("Target Base-Stock (S)")
        ax.set_ylabel("Expected Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig("docs/phase3_95_figures/reward_landscape.png", dpi=300)
    print("Saved docs/phase3_95_figures/reward_landscape.png")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    run_landscape_audit()
