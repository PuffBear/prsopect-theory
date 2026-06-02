import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

def evaluate_and_extract_metrics(model, eval_env, num_episodes=5, base_seed=42):
    total_order_qty = []
    total_inv = []
    total_lost_sales = []
    total_reward = []
    agent_orders_var = []
    demand_var = []
    
    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        obs, _ = eval_env.reset(seed=ep_seed)
        done = False
        ep_rew = 0
        ep_orders = []
        ep_demand = []
        
        while eval_env.agents:
            actions = {}
            for agent in eval_env.agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    real_shape = eval_env.action_space(agent).shape[0]
                    actions[agent] = action[:real_shape]
                    
            obs, rewards, term, trunc, info = eval_env.step(actions)
            
            curr_env = eval_env
            while hasattr(curr_env, 'env') or hasattr(curr_env, 'unwrapped_env'):
                if hasattr(curr_env, 'unwrapped_env'):
                    base_env = curr_env.unwrapped_env
                    break
                curr_env = curr_env.env
                
            t = base_env.period - 1
            
            step_ppo_orders = []
            for agent in eval_env.possible_agents:
                node_id = eval_env.unwrapped.agent_name_to_id[agent]
                for link in base_env.reorder_links:
                    if link[1] == node_id:
                        step_ppo_orders.append(base_env.R.loc[t, link])
            
            ep_orders.append(np.mean(step_ppo_orders))
            ep_demand.append(base_env.D.loc[t, :].sum())
            
            for agent in rewards:
                ep_rew += info[agent]["true_reward"]
            
            total_order_qty.extend(step_ppo_orders)
                
        total_inv.append(base_env.X.mean().mean())
        total_D = base_env.D.sum().sum()
        total_U = base_env.U.sum().sum()
        ls_ratio = total_U / total_D if total_D > 0 else 0
        total_lost_sales.append(ls_ratio)
        total_reward.append(ep_rew)
        
        if np.var(ep_demand) > 0:
            agent_orders_var.append(np.var(ep_orders) / np.var(ep_demand))
        else:
            agent_orders_var.append(0)
            
    # Also extract the S parameter the model converges to
    obs, _ = eval_env.reset(seed=42)
    s_vals = []
    for agent in eval_env.agents:
        if agent in obs:
            action, _ = model.predict(obs[agent], deterministic=True)
            S = 0.0 + 0.5 * (action[0] + 1.0) * 500.0  # low=0, high=500
            s_vals.append(S)
            
    mean_order = np.mean(total_order_qty) if total_order_qty else 0.0
    return np.mean(s_vals), mean_order, np.mean(total_inv), np.mean(total_lost_sales), np.mean(total_reward), np.mean(agent_orders_var)

def _run_single_seed(args):
    seed, lam = args
    np.random.seed(seed)
    
    warnings.filterwarnings("ignore")
    
    raw_env = MultiAgentNetInvMgmt()
    scripted_nodes = ["node_1"]
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    
    # alpha=1.0 for centralized rewards (the stabilized MARL substrate)
    agent_params = {agent: {"lambda": lam, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
    cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=1.0)
    
    # S bounds [0, 500]
    train_env = ScaledBaseStockWrapper(cpt_env, low_bound=0.0, high_bound=500.0)
    
    vec_env = ss.pad_action_space_v0(train_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(vec_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=0, base_class='stable_baselines3')
    vec_env.seed = lambda s: None
    
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=150000)
    
    # Evaluation
    eval_raw = MultiAgentNetInvMgmt()
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
    eval_cpt = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=1.0)
    eval_env = ScaledBaseStockWrapper(eval_cpt, low_bound=0.0, high_bound=500.0)
    
    mean_S, mean_order, mean_inv, mean_ls, mean_prof, mean_bw = evaluate_and_extract_metrics(model, eval_env, num_episodes=5)
    
    return seed, lam, mean_S, mean_order, mean_inv, mean_ls, mean_prof, mean_bw

def run_phase4_replication():
    print(f"\n=======================================================")
    print(f"Starting Phase 4: Prospect Theory Replication (Centralized)")
    print(f"=======================================================")
    
    lambdas = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    seeds = list(range(42, 52)) # 10 seeds
    
    args_list = []
    for lam in lambdas:
        for seed in seeds:
            args_list.append((seed, lam))
            
    results = []
    
    with mp.Pool(processes=min(mp.cpu_count(), 10)) as pool:
        for res in pool.imap_unordered(_run_single_seed, args_list):
            seed, lam, s, mq, mi, ls, prof, bw = res
            print(f"Lambda:{lam:4.1f} Seed:{seed:2d} -> S:{s:6.1f} | Prof:{prof:6.1f} | Inv:{mi:6.1f} | BW:{bw:4.2f}")
            results.append({
                "Lambda": lam,
                "Seed": seed,
                "Mean S": s,
                "Mean Order": mq,
                "Mean Inventory": mi,
                "Lost Sales": ls,
                "Profit": prof,
                "Bullwhip": bw
            })
            
    df = pd.DataFrame(results)
    
    os.makedirs("docs/phase4_figures", exist_ok=True)
    df.to_csv("docs/phase4_figures/phase4_raw_data.csv", index=False)
    
    # Plotting with Confidence Intervals
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.lineplot(data=df, x="Lambda", y="Mean S", errorbar=('ci', 95), marker='o', ax=axes[0, 0])
    axes[0, 0].set_title("Learned Base-Stock (S) vs Loss Aversion (λ)")
    
    sns.lineplot(data=df, x="Lambda", y="Profit", errorbar=('ci', 95), marker='o', ax=axes[0, 1])
    axes[0, 1].set_title("True Economic Profit vs Loss Aversion (λ)")
    
    sns.lineplot(data=df, x="Lambda", y="Mean Inventory", errorbar=('ci', 95), marker='o', ax=axes[1, 0])
    axes[1, 0].set_title("Network Inventory vs Loss Aversion (λ)")
    
    sns.lineplot(data=df, x="Lambda", y="Bullwhip", errorbar=('ci', 95), marker='o', ax=axes[1, 1])
    axes[1, 1].set_title("Bullwhip Effect vs Loss Aversion (λ)")
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("docs/phase4_figures/pt_replication_ci.png", dpi=300)
    print("Saved docs/phase4_figures/pt_replication_ci.png")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    run_phase4_replication()
