import os
import sys
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from agents.cpt_wrapper import CPTRewardWrapper
from utils.metrics import calculate_bullwhip_effect, calculate_systemic_lost_sales

def train_and_visualize(lambda_value, fraction_loss_averse, scenario_name, timesteps=100000):
    # Setup
    env = MultiAgentNetInvMgmt()
    agents = env.agents
    num_loss_averse = int(len(agents) * fraction_loss_averse)
    agent_params = {}
    for i, agent in enumerate(agents):
        if i < num_loss_averse:
            agent_params[agent] = {"lambda": lambda_value, "alpha": 1.0, "beta": 1.0}
        else:
            agent_params[agent] = {"lambda": 1.0, "alpha": 1.0, "beta": 1.0}
            
    env = CPTRewardWrapper(env, agent_params)
    
    # Wrap for SB3
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    
    print(f"Training {scenario_name} for {timesteps} steps...")
    model = PPO("MlpPolicy", env, verbose=0, n_steps=128, batch_size=256)
    model.learn(total_timesteps=timesteps)
    
    # Evaluate
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
        
        # Log inventory before step
        base_env = test_env.env.unwrapped_env
        inventory_history.append(base_env.X.iloc[base_env.period].values.copy())
        
        obs, rewards, term, trunc, info = test_env.step(actions)
        
    base_env = test_env.env.unwrapped_env
    bw = calculate_bullwhip_effect(base_env)
    ls = calculate_systemic_lost_sales(base_env)
    
    print(f"*** {scenario_name} ***")
    print(f"Bullwhip Effect: {bw:.2f}")
    print(f"Systemic Lost Sales: {ls:.2%}")
    print("*" * 50)
    
    # Plot Inventory
    inventory_history = np.array(inventory_history)
    plt.figure(figsize=(12, 6))
    for stage in range(inventory_history.shape[1]):
        plt.plot(inventory_history[:, stage], label=f'Stage {stage+1} (Node {stage+1})', linewidth=2)
        
    plt.title(f"{scenario_name}: Inventory Levels Over Time\nBullwhip: {bw:.2f} | Lost Sales: {ls:.2%}")
    plt.xlabel("Time Period")
    plt.ylabel("Inventory on Hand")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("docs/assets", exist_ok=True)
    filename = scenario_name.replace(" ", "_").lower() + ".png"
    plt.savefig(f"docs/assets/{filename}", dpi=150)
    print(f"Saved visualization to docs/assets/{filename}")

if __name__ == "__main__":
    train_and_visualize(lambda_value=1.0, fraction_loss_averse=0.0, scenario_name="Baseline_Rational", timesteps=100000)
    train_and_visualize(lambda_value=5.0, fraction_loss_averse=0.3, scenario_name="Cascade_Averse", timesteps=100000)
