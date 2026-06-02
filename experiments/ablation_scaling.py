import os
import sys
import numpy as np
import pandas as pd
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from agents.cpt_wrapper import CPTRewardWrapper
from utils.learning_diagnostics import record_learning_diagnostics

def train_ablation(phi, scale, seed=0, total_timesteps=100000):
    np.random.seed(seed)
    env = MultiAgentNetInvMgmt()
    agents = env.agents
    
    num_loss_averse = int(np.round(len(agents) * phi))
    loss_averse_nodes = np.random.choice(agents, size=num_loss_averse, replace=False)
    
    agent_params = {}
    for agent in agents:
        if agent in loss_averse_nodes:
            agent_params[agent] = {"lambda": 5.0, "alpha": 1.0, "beta": 1.0}
        else:
            agent_params[agent] = {"lambda": 1.0, "alpha": 1.0, "beta": 1.0}
            
    train_env = CPTRewardWrapper(env, agent_params, reward_scale=scale)
    train_env = ss.pad_action_space_v0(train_env)
    train_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
    train_env = ss.concat_vec_envs_v1(train_env, 1, num_cpus=1, base_class='stable_baselines3')
    
    eval_env_raw = MultiAgentNetInvMgmt()
    eval_env = CPTRewardWrapper(eval_env_raw, agent_params, reward_scale=scale)
    
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=256)
    
    diag_results = []
    blocks = 5
    steps_per_block = total_timesteps // blocks
    
    for block in range(blocks):
        model.learn(total_timesteps=steps_per_block, reset_num_timesteps=False)
        ts = (block + 1) * steps_per_block
        
        o_qty, inv_lvl, ls_ratio, ep_rew = record_learning_diagnostics(model, eval_env, num_episodes=3)
        print(f"  [{ts:6d}] Order: {o_qty:.2f} | Inv: {inv_lvl:.2f} | LS: {ls_ratio:.2%} | Rew: {ep_rew:.2f}")
        
        # Extract reward diagnostics
        raw_rewards = []
        scaled_rewards = []
        cpt_rewards = []
        
        for _ in range(1):
            obs, _ = eval_env.reset()
            done = False
            while eval_env.agents:
                actions = {}
                for agent in eval_env.agents:
                    if agent in obs:
                        action, _ = model.predict(obs[agent], deterministic=True)
                        real_shape = eval_env.action_space(agent).shape[0]
                        actions[agent] = action[:real_shape]
                        
                obs, rewards, term, trunc, info = eval_env.step(actions)
                for agent, i in info.items():
                    if "raw_reward" in i:
                        raw_rewards.append(i["raw_reward"])
                        scaled_rewards.append(i["scaled_reward"])
                        cpt_rewards.append(i["cpt_reward"])
                        
        diag_results.append({
            "phi": phi,
            "scale": scale,
            "timestep": ts,
            "order_qty": o_qty,
            "inventory": inv_lvl,
            "lost_sales": ls_ratio,
            "true_reward": ep_rew,
            "raw_reward_mean": np.mean(raw_rewards) if raw_rewards else 0,
            "raw_reward_std": np.std(raw_rewards) if raw_rewards else 0,
            "scaled_reward_mean": np.mean(scaled_rewards) if scaled_rewards else 0,
            "scaled_reward_std": np.std(scaled_rewards) if scaled_rewards else 0,
            "cpt_reward_mean": np.mean(cpt_rewards) if cpt_rewards else 0,
            "cpt_reward_std": np.std(cpt_rewards) if cpt_rewards else 0
        })
        
    return diag_results

if __name__ == "__main__":
    phi_values = [0.0, 1.0]
    scales = [1.0, 10.0, 100.0]
    total_timesteps = 100_000
    
    all_diagnostics = []
    os.makedirs("docs/results", exist_ok=True)
    
    for phi in phi_values:
        for scale in scales:
            print(f"\nRunning Ablation: phi={phi}, scale={scale}...")
            diags = train_ablation(phi, scale, seed=0, total_timesteps=total_timesteps)
            all_diagnostics.extend(diags)
            
    df = pd.DataFrame(all_diagnostics)
    df.to_csv("docs/results/ablation_scaling.csv", index=False)
    print("\nAblation study complete! Results saved to docs/results/ablation_scaling.csv")
