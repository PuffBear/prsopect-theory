import os
import sys
import numpy as np
import pandas as pd
import or_gym
import gymnasium as gym
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Environment Wrapper
class Phase0Wrapper(gym.Wrapper):
    def __init__(self, env, lam=1.0):
        super().__init__(env)
        self.lam = lam

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
        
        # Save true reward in info
        info["true_reward"] = reward
        
        # Apply Prospect Theory Shaping
        # Scale down to prevent explosion
        r = reward / 100.0
        if r >= 0:
            shaped_reward = r
        else:
            shaped_reward = -self.lam * (-r)
            
        return obs, shaped_reward, terminated, truncated, info
        
import types

def create_env(lam):
    # InvManagement-v1 is a single-agent linear supply chain
    env = or_gym.make("InvManagement-v1")
    
    original_reset = env.unwrapped.reset
    def patched_reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = original_reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs
        return obs, {}
    env.unwrapped.reset = types.MethodType(patched_reset, env.unwrapped)
    
    env = Phase0Wrapper(env, lam=lam)
    return env

# 2. Evaluation Logic
def evaluate_policy(model, env, num_episodes=1):
    all_orders = []
    all_inv = []
    all_backlog = []
    all_true_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_true_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            base_env = env.unwrapped
            all_orders.append(np.mean(action))
            all_inv.append(np.mean(base_env.I))
            
            if hasattr(base_env, 'B'):
                all_backlog.append(np.mean(base_env.B))
            elif hasattr(base_env, 'U'):
                all_backlog.append(np.mean(base_env.U))
            else:
                all_backlog.append(0)
                
            ep_true_reward += info["true_reward"]
            
        all_true_rewards.append(ep_true_reward)
        
    return np.mean(all_orders), np.mean(all_inv), np.mean(all_backlog), np.mean(all_true_rewards)

# 3. Main Training Loop
if __name__ == "__main__":
    lambdas = [1.0, 2.0, 5.0, 10.0]
    blocks = 30
    steps_per_block = 5000
    
    results = []
    os.makedirs("docs/results", exist_ok=True)
    
    for lam in lambdas:
        print(f"--- Training Phase 0 for lambda = {lam} ---")
        
        train_env = create_env(lam)
        eval_env = create_env(lam)
        
        model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=64)
        
        for block in range(blocks):
            model.learn(total_timesteps=steps_per_block, reset_num_timesteps=False)
            
            timestep = (block + 1) * steps_per_block
            mean_order, mean_inv, mean_backlog, mean_reward = evaluate_policy(model, eval_env, num_episodes=3)
            
            print(f"[{timestep:6d}] Order: {mean_order:.2f} | Inv: {mean_inv:.2f} | Backlog: {mean_backlog:.2f} | Reward: {mean_reward:.2f}")
            
            results.append({
                "lambda": lam,
                "timestep": timestep,
                "order_qty": mean_order,
                "inventory": mean_inv,
                "backlog": mean_backlog,
                "true_reward": mean_reward
            })
            
    df = pd.DataFrame(results)
    df.to_csv("docs/results/phase0_data.csv", index=False)
    print("Saved results to docs/results/phase0_data.csv")
