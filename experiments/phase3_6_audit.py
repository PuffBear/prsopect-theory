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
from env.base_stock_wrapper import BaseStockWrapper
from pettingzoo.utils import BaseParallelWrapper

global_action_logs = []

class LoggingBaseStockWrapper(BaseParallelWrapper):
    """
    Inherits from BaseParallelWrapper (like BaseStockWrapper) but logs internal action mechanics.
    """
    def __init__(self, env, low_bound=0.0, high_bound=500.0):
        super().__init__(env)
        self.action_spaces = {
            agent: gym.spaces.Box(low=low_bound, high=high_bound, shape=self.env.action_space(agent).shape, dtype=np.float64)
            for agent in self.possible_agents
        }
        self.step_counter = 0
        self.log_file = f"/tmp/action_logs_{int(high_bound)}.csv"
        # Write header only on init
        if not os.path.exists(self.log_file) or getattr(self, '_cleared', False) == False:
            with open(self.log_file, "w") as f:
                f.write("step,agent,S,IP,Q\n")
            self._cleared = True

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        base_env = self.unwrapped.unwrapped_env
        t = base_env.period
        
        translated_actions = {}
        for agent, action in actions.items():
            # Action is the raw scalar output by PPO, mapped by SB3 from [-1, 1] to the Box bounds.
            S = action[0]
            node_id = self.unwrapped.agent_name_to_id[agent]
            
            # 1. On-Hand Inventory
            on_hand = base_env.X.loc[t, node_id]
            
            # 2. Pipeline Inventory (Incoming orders not yet arrived)
            pipeline = 0
            for link in base_env.reorder_links:
                if link[1] == node_id:
                    pipeline += base_env.Y.loc[t, link]
                    
            # 3. Backlog (Unfulfilled retail demand, only applies to retailers)
            backlog = 0
            if node_id in base_env.retail and t > 0:
                for link in base_env.retail_links:
                    if link[0] == node_id:
                        backlog += base_env.U.loc[t-1, link]
                        
            # Total Inventory Position
            IP = on_hand + pipeline - backlog
            
            R = max(0, S - IP)
            translated_actions[agent] = np.array([R], dtype=np.float64)
            
            with open(self.log_file, "a") as f:
                f.write(f"{self.step_counter},{agent},{S},{IP},{R}\n")
            
        self.step_counter += 1
        return self.env.step(translated_actions)
        
    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.action_spaces[agent]

def run_audit(name, low_bound, high_bound, total_steps=50000):
    log_file = f"/tmp/action_logs_{int(high_bound)}.csv"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    print(f"\n--- Running Audit: {name} ---")
    seed = 42
    np.random.seed(seed)
    
    raw_env = MultiAgentNetInvMgmt()
    scripted_nodes = ["node_1"]
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
    cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    
    # Use Logging wrapper
    train_env = LoggingBaseStockWrapper(cpt_env, low_bound=low_bound, high_bound=high_bound)
    
    vec_env = ss.pad_action_space_v0(train_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(vec_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=0, base_class='stable_baselines3')
    vec_env.seed = lambda s: None
    
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=total_steps)
    
    df = pd.read_csv(log_file)
    print(f"[{name}] DataFrame loaded from {log_file} with shape {df.shape}")
    return df

def generate_audit_plots():
    os.makedirs("docs/phase3_6_figures", exist_ok=True)
    
    df_orig = run_audit("Original Bounds [0, 500]", 0.0, 500.0)
    df_shift = run_audit("Shifted Bounds [200, 700]", 200.0, 700.0)
    
    # Plot 1: Histograms of S and Q for both
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(df_orig['S'], bins=50, color='red', alpha=0.7)
    plt.title("Original Bounds: Target Base-Stock (S)")
    plt.xlabel("S")
    
    plt.subplot(2, 2, 2)
    plt.hist(df_orig['Q'], bins=50, color='red', alpha=0.7)
    plt.title("Original Bounds: Order Quantity (Q)")
    plt.xlabel("Q")
    
    plt.subplot(2, 2, 3)
    plt.hist(df_shift['S'], bins=50, color='blue', alpha=0.7)
    plt.title("Shifted Bounds: Target Base-Stock (S)")
    plt.xlabel("S")
    
    plt.subplot(2, 2, 4)
    plt.hist(df_shift['Q'], bins=50, color='blue', alpha=0.7)
    plt.title("Shifted Bounds: Order Quantity (Q)")
    plt.xlabel("Q")
    
    plt.tight_layout()
    plt.savefig("docs/phase3_6_figures/action_histograms.png", dpi=300)
    print("Saved docs/phase3_6_figures/action_histograms.png")
    
    # Plot 2: P(Q > 0) over training
    # Group by steps (bins of 1000)
    df_orig['StepGroup'] = (df_orig['step'] // 1000) * 1000
    df_shift['StepGroup'] = (df_shift['step'] // 1000) * 1000
    
    p_orig = df_orig.groupby('StepGroup')['Q'].apply(lambda x: (x > 0).mean())
    p_shift = df_shift.groupby('StepGroup')['Q'].apply(lambda x: (x > 0).mean())
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_orig.index, p_orig.values, label="Original Bounds [0, 500]", color='red', linewidth=2)
    plt.plot(p_shift.index, p_shift.values, label="Shifted Bounds [200, 700]", color='blue', linewidth=2)
    plt.title("P(Q > 0) Throughout Training (The Gradient Dead Zone)")
    plt.xlabel("Environment Step")
    plt.ylabel("Probability of Placing an Order")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig("docs/phase3_6_figures/prob_order_over_time.png", dpi=300)
    print("Saved docs/phase3_6_figures/prob_order_over_time.png")

if __name__ == "__main__":
    generate_audit_plots()
