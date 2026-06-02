import os
import sys
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from agents.cpt_wrapper import CPTRewardWrapper
from utils.metrics import calculate_bullwhip_effect, calculate_systemic_lost_sales

def run_sb3_experiment(lambda_value, fraction_loss_averse):
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
    
    model = PPO("MlpPolicy", env, verbose=0, n_steps=60)
    model.learn(total_timesteps=2000)
    
    # Evaluate
    test_env = MultiAgentNetInvMgmt()
    test_env = CPTRewardWrapper(test_env, agent_params)
    obs, info = test_env.reset()
    
    while test_env.agents:
        actions = {}
        for agent in test_env.agents:
            if agent in obs:
                action, _ = model.predict(obs[agent], deterministic=True)
                real_shape = test_env.action_space(agent).shape[0]
                actions[agent] = action[:real_shape]
        obs, rewards, term, trunc, info = test_env.step(actions)
        
    base_env = test_env.env.env
    bw = calculate_bullwhip_effect(base_env)
    ls = calculate_systemic_lost_sales(base_env)
    
    print(f"*** FINAL EVALUATION (Lambda={lambda_value}, Fraction High-Lambda={fraction_loss_averse:.0%}) ***")
    print(f"Bullwhip Effect: {bw:.2f}")
    print(f"Systemic Lost Sales: {ls:.2%}")
    print("*" * 50)

if __name__ == "__main__":
    print("Running Baseline (Homogeneous Rational Agents, lambda=1.0)...")
    run_sb3_experiment(lambda_value=1.0, fraction_loss_averse=0.0)
    
    print("\nRunning High Loss Aversion Cascade Simulation (lambda=5.0 for 30% of agents)...")
    run_sb3_experiment(lambda_value=5.0, fraction_loss_averse=0.3)
