import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.base_stock_wrapper import BaseStockWrapper

class ShiftedBaseStockWrapper(BaseStockWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_spaces = {
            agent: gym.spaces.Box(low=200.0, high=700.0, shape=self.env.action_space(agent).shape, dtype=np.float64)
            for agent in self.possible_agents
        }
import gymnasium as gym

def train_and_plot():
    seed = 42
    np.random.seed(seed)
    
    # 1. Setup Environment
    raw_env = MultiAgentNetInvMgmt()
    scripted_nodes = ["node_1"]
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
    cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    train_env = ShiftedBaseStockWrapper(cpt_env)
    
    train_env = ss.pad_action_space_v0(train_env)
    train_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
    train_env = ss.concat_vec_envs_v1(train_env, 1, num_cpus=1, base_class='stable_baselines3')
    train_env.seed = lambda s: None
    
    # 2. Train Model
    print("Training Phase 3.5 Success Model...")
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=150000)
    
    # 3. Evaluate and Collect Trajectory
    eval_raw = MultiAgentNetInvMgmt()
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
    eval_cpt = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    eval_env = ShiftedBaseStockWrapper(eval_cpt)
    
    obs, _ = eval_env.reset()
    
    records = []
    
    while eval_env.agents:
        actions = {}
        for agent in eval_env.agents:
            if agent in obs:
                action, _ = model.predict(obs[agent], deterministic=True)
                real_shape = eval_env.action_space(agent).shape[0]
                actions[agent] = action[:real_shape]
                
        # Get underlying env period before stepping
        base_env = eval_raw.unwrapped_env
        t = base_env.period
        
        obs, rewards, term, trunc, info = eval_env.step(actions)
        
        # Extract metrics for period t
        # Network Inventory is sum of all inventory nodes
        net_inv = base_env.X.loc[t, :].sum()
        
        # Market Demand
        demand = base_env.D.loc[t, :].sum()
        
        # PPO Agent Orders (distributors and factories)
        ppo_orders = 0
        for agent in eval_env.possible_agents:
            node_id = eval_env.unwrapped.agent_name_to_id[agent]
            for link in base_env.reorder_links:
                if link[1] == node_id:
                    ppo_orders += base_env.R.loc[t, link]
        
        records.append({
            'Period': t,
            'Network Inventory': net_inv,
            'Market Demand': demand,
            'PPO Orders': ppo_orders
        })
        
    df = pd.DataFrame(records)
    
    # 4. Plot
    os.makedirs("docs/phase3_5_figures", exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['Period'], df['Market Demand'], label='Market Demand', color='orange', linestyle='--', marker='o')
    plt.plot(df['Period'], df['PPO Orders'], label='PPO Agents Total Order Quantity', color='blue', marker='x')
    plt.title('Phase 3.5: Successful Continuous Base-Stock Dynamics (Action Range S  in  [200, 700])')
    plt.ylabel('Quantity')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df['Period'], df['Network Inventory'], label='Total Network Inventory', color='green', linewidth=2)
    plt.xlabel('Simulation Period')
    plt.ylabel('Inventory Units')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("docs/phase3_5_figures/success_dynamics.png", dpi=300)
    print("Saved plot to docs/phase3_5_figures/success_dynamics.png")

if __name__ == "__main__":
    train_and_plot()
