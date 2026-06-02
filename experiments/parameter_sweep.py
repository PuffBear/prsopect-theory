import os
import sys
import numpy as np
import pandas as pd
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from agents.cpt_wrapper import CPTRewardWrapper
from utils.metrics import calculate_bullwhip_effect, calculate_systemic_lost_sales

def run_evaluation(model, agent_params):
    test_env = MultiAgentNetInvMgmt()
    test_env = CPTRewardWrapper(test_env, agent_params)
    obs, info = test_env.reset()
    
    inventory_history = []
    
    while test_env.agents:
        actions = {}
        for agent in test_env.agents:
            if agent in obs:
                action, _ = model.predict(obs[agent], deterministic=True)
                real_shape = test_env.action_space(agent).shape[0]
                actions[agent] = action[:real_shape]
                
        base_env = test_env.env.unwrapped_env
        inventory_history.append(base_env.X.iloc[base_env.period].values.copy())
        
        obs, rewards, term, trunc, info = test_env.step(actions)
        
    base_env = test_env.env.unwrapped_env
    bw = calculate_bullwhip_effect(base_env)
    ls = calculate_systemic_lost_sales(base_env)
    
    inventory_history = np.array(inventory_history)
    inv_var = np.mean(np.var(inventory_history, axis=0))  # Average variance across all nodes
    
    return bw, ls, inv_var

def train_and_eval(num_loss_averse, seed):
    # Set random seeds
    np.random.seed(seed)
    
    env = MultiAgentNetInvMgmt()
    agents = env.agents
    agent_params = {}
    for i, agent in enumerate(agents):
        if i < num_loss_averse:
            agent_params[agent] = {"lambda": 5.0, "alpha": 1.0, "beta": 1.0}
        else:
            agent_params[agent] = {"lambda": 1.0, "alpha": 1.0, "beta": 1.0}
            
    env = CPTRewardWrapper(env, agent_params)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    
    model = PPO("MlpPolicy", env, verbose=0, n_steps=128, batch_size=256)
    model.learn(total_timesteps=60000)
    
    return run_evaluation(model, agent_params)

if __name__ == "__main__":
    # There are 6 agents total in the network. We will sweep from 0 to 6 loss-averse agents.
    num_averse_list = [0, 1, 2, 3, 4, 5, 6]
    seeds = [42, 1337, 2026]
    
    results = []
    
    os.makedirs("docs/results", exist_ok=True)
    
    total_runs = len(num_averse_list) * len(seeds)
    current_run = 0
    
    for num_averse in num_averse_list:
        phi = num_averse / 6.0
        for seed in seeds:
            current_run += 1
            print(f"[{current_run}/{total_runs}] Running num_averse={num_averse} (phi={phi:.2f}), seed={seed}...")
            try:
                bw, ls, inv_var = train_and_eval(num_averse, seed)
                results.append({
                    "num_averse": num_averse,
                    "phi": phi,
                    "seed": seed,
                    "bullwhip": bw,
                    "lost_sales": ls,
                    "inventory_variance": inv_var
                })
                print(f"  -> Bullwhip: {bw:.2f}, Lost Sales: {ls:.2%}, Variance: {inv_var:.2f}")
            except Exception as e:
                print(f"  -> Failed: {e}")
                
    df = pd.DataFrame(results)
    df.to_csv("docs/results/sweep_data.csv", index=False)
    print("Saved sweep results to docs/results/sweep_data.csv")
