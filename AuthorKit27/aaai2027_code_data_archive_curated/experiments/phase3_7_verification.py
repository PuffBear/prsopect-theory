import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

def evaluate_and_extract_metrics(model, eval_env, num_episodes=5, base_seed=42):
    """
    Evaluates the model over num_episodes and extracts true economic metrics.
    Varies the seed per episode to expose true environment variance.
    """
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
                    # model.predict returns an action in [-1, 1] because that's our action_space!
                    action, _ = model.predict(obs[agent], deterministic=True)
                    real_shape = eval_env.action_space(agent).shape[0]
                    actions[agent] = action[:real_shape]
                    
            step_actions = actions.copy()
            obs, rewards, term, trunc, info = eval_env.step(step_actions)
            
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
            
    mean_order = np.mean(total_order_qty) if total_order_qty else 0.0
    return mean_order, np.mean(total_inv), np.mean(total_lost_sales), np.mean(total_reward), np.mean(agent_orders_var)

def run_ablation(low_bound, high_bound, seed=42, total_steps=100000):
    name = f"[{int(low_bound)}, {int(high_bound)}]"
    log_file = f"/tmp/scaled_logs_{int(low_bound)}_{int(high_bound)}.csv"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    print(f"\n--- Running Ablation: {name} ---")
    np.random.seed(seed)
    
    raw_env = MultiAgentNetInvMgmt()
    scripted_nodes = ["node_1"]
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
    cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    
    train_env = ScaledBaseStockWrapper(cpt_env, low_bound=low_bound, high_bound=high_bound, log_file=log_file)
    
    vec_env = ss.pad_action_space_v0(train_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(vec_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=0, base_class='stable_baselines3')
    vec_env.seed = lambda s: None
    
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=total_steps)
    
    df = pd.read_csv(log_file)
    print(f"{name} DataFrame loaded with shape {df.shape}")
    
    eval_raw = MultiAgentNetInvMgmt()
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
    eval_cpt = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    eval_env = ScaledBaseStockWrapper(eval_cpt, low_bound=low_bound, high_bound=high_bound)
    
    _, _, _, profit, _ = evaluate_and_extract_metrics(model, eval_env, num_episodes=5)
    print(f"{name} Profit: {profit:.2f}")
    
    return df, profit

def generate_ablation_plots():
    os.makedirs("docs/phase3_7_figures", exist_ok=True)
    
    ranges = [
        (0.0, 500.0),
        (100.0, 600.0),
        (200.0, 700.0),
        (300.0, 800.0)
    ]
    
    dfs = {}
    profits = {}
    for low, high in ranges:
        df, profit = run_ablation(low, high)
        dfs[(low, high)] = df
        profits[(low, high)] = profit
        
    plt.figure(figsize=(14, 10))
    
    for i, (low, high) in enumerate(ranges):
        df = dfs[(low, high)]
        
        plt.subplot(2, 2, i+1)
        
        # Track S over time for a single agent to avoid clutter
        df_agent = df[df['agent'] == 'node_2']
        # Smooth with a rolling window of 1000 steps
        df_agent = df_agent.sort_values('step')
        df_agent['S_smooth'] = df_agent['S'].rolling(1000).mean()
        
        plt.plot(df_agent['step'], df_agent['S'], alpha=0.1, color='blue')
        plt.plot(df_agent['step'], df_agent['S_smooth'], color='blue', linewidth=2)
        
        mu_S = df_agent['S'].mean()
        std_S = df_agent['S'].std()
        
        plt.title(f"Range [{int(low)}, {int(high)}] | Profit: {profits[(low, high)]:.0f}\nS = {mu_S:.1f} +/- {std_S:.1f}")
        plt.xlabel("Training Step")
        plt.ylabel("Target Base-Stock (S)")
        plt.axhline(y=low, color='r', linestyle='--', alpha=0.5, label='Lower Bound')
        plt.axhline(y=high, color='g', linestyle='--', alpha=0.5, label='Upper Bound')
        plt.ylim(low - 50, high + 50)
        plt.legend(loc="upper right")
        
    plt.tight_layout()
    plt.savefig("docs/phase3_7_figures/ablation_S_trajectories.png", dpi=300)
    print("Saved docs/phase3_7_figures/ablation_S_trajectories.png")
    
    return dfs

def _run_single_seed(args):
    seed, low, high = args
    np.random.seed(seed)
    
    raw_env = MultiAgentNetInvMgmt()
    scripted_nodes = ["node_1"]
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
    cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    train_env = ScaledBaseStockWrapper(cpt_env, low_bound=low, high_bound=high)
    
    vec_env = ss.pad_action_space_v0(train_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(vec_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=0, base_class='stable_baselines3')
    vec_env.seed = lambda s: None
    
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=150000)
    
    eval_raw = MultiAgentNetInvMgmt()
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
    eval_cpt = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    eval_env = ScaledBaseStockWrapper(eval_cpt, low_bound=low, high_bound=high)
    
    o_qty, inv_lvl, ls_ratio, ep_rew, bw = evaluate_and_extract_metrics(model, eval_env, num_episodes=5)
    return seed, o_qty, inv_lvl, ls_ratio, ep_rew, bw

def run_20_seed_replication(low=200.0, high=700.0):
    import multiprocessing as mp
    
    print(f"\n==============================================")
    print(f"Starting 20-Seed Replication on [{int(low)}, {int(high)}]")
    print(f"==============================================")
    
    seeds = list(range(42, 62))
    args_list = [(seed, low, high) for seed in seeds]
    
    results = []
    with mp.Pool(processes=min(mp.cpu_count(), 10)) as pool:
        for res in pool.imap_unordered(_run_single_seed, args_list):
            seed, o_qty, inv_lvl, ls_ratio, ep_rew, bw = res
            print(f"Seed {seed:02d} -> Profit: {ep_rew:.2f}, Qty: {o_qty:.2f}, Bullwhip: {bw:.2f}")
            results.append((o_qty, inv_lvl, ls_ratio, ep_rew, bw))
        
    print("\n=== Phase 3.7 Verification Results (20 Seeds) ===")
    o_qtys, inv_lvls, ls_ratios, ep_rews, bws = zip(*results)
    print(f"PPO Mean Order Quantity : {np.mean(o_qtys):.2f} +/- {np.std(o_qtys):.2f}")
    print(f"Network Inventory Level : {np.mean(inv_lvls):.2f} +/- {np.std(inv_lvls):.2f}")
    print(f"Lost Sales Ratio        : {np.mean(ls_ratios):.2%}")
    print(f"True Economic Profit    : {np.mean(ep_rews):.2f} +/- {np.std(ep_rews):.2f}")
    print(f"Bullwhip Ratio          : {np.mean(bws):.2f} +/- {np.std(bws):.2f}")
    print("=================================================")

if __name__ == "__main__":
    # Prevent multiprocessing fork bomb on mac
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    generate_ablation_plots()
    run_20_seed_replication(200.0, 700.0)
