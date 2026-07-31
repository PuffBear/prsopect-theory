import os
import sys
import json
import numpy as np
import pandas as pd
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from agents.cpt_wrapper import CPTRewardWrapper
from utils.learning_diagnostics import record_learning_diagnostics
from utils.phase1_metrics import compute_bullwhip, compute_lost_sales_ratio, compute_inventory_variance, get_mean_economic_reward

def run_evaluation_and_log_traces(model, test_env, phi, seed):
    obs, info = test_env.reset()
    done = False
    
    trace_data = []
    timestep = 0
    
    while test_env.agents:
        actions = {}
        for agent in test_env.agents:
            if agent in obs:
                action, _ = model.predict(obs[agent], deterministic=True)
                real_shape = test_env.action_space(agent).shape[0]
                actions[agent] = action[:real_shape]
        
        base_env = test_env.env.unwrapped_env
        
        obs, rewards, term, trunc, info = test_env.step(actions)
        
        for agent in test_env.possible_agents:
            if agent in actions:
                order_qty = np.mean(actions[agent])
                # agent names are typically 'agent_0', 'agent_1' etc
                node_idx = int(agent.split('_')[1])
                inv = base_env.X.iloc[base_env.period].values[node_idx] if node_idx < len(base_env.X.columns) else 0
                true_reward = info.get(agent, {}).get("true_reward", rewards.get(agent, 0))
                
                trace_data.append({
                    "timestep": timestep,
                    "node_id": agent,
                    "order_quantity": order_qty,
                    "inventory": inv,
                    "true_reward": true_reward
                })
        timestep += 1
        
    base_env = test_env.env.unwrapped_env
    bw = compute_bullwhip(base_env)
    ls = compute_lost_sales_ratio(base_env)
    inv_var = compute_inventory_variance(base_env)
    mean_reward = get_mean_economic_reward(base_env)
    
    trace_df = pd.DataFrame(trace_data)
    os.makedirs("logs/agent_traces", exist_ok=True)
    trace_df.to_csv(f"logs/agent_traces/trace_phi_{phi}_seed_{seed}.csv", index=False)
    
    return bw, ls, inv_var, mean_reward

def train_phase1(phi, seed, total_timesteps=250000):
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
            
    metadata = {
        "phi": phi,
        "seed": seed,
        "loss_averse_nodes": list(loss_averse_nodes)
    }
    os.makedirs("docs/results/metadata", exist_ok=True)
    with open(f"docs/results/metadata/meta_phi_{phi}_seed_{seed}.json", "w") as f:
        json.dump(metadata, f)
        
    train_env = CPTRewardWrapper(env, agent_params)
    train_env = ss.pad_action_space_v0(train_env)
    train_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
    train_env = ss.concat_vec_envs_v1(train_env, 1, num_cpus=1, base_class='stable_baselines3')
    
    eval_env_raw = MultiAgentNetInvMgmt()
    eval_env = CPTRewardWrapper(eval_env_raw, agent_params)
    
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=256)
    
    diag_results = []
    blocks = 10
    steps_per_block = total_timesteps // blocks
    
    for block in range(blocks):
        model.learn(total_timesteps=steps_per_block, reset_num_timesteps=False)
        ts = (block + 1) * steps_per_block
        
        o_qty, inv_lvl, ls_ratio, ep_rew = record_learning_diagnostics(model, eval_env, num_episodes=3)
        print(f"  [{ts:6d}] Order: {o_qty:.2f} | Inv: {inv_lvl:.2f} | LS: {ls_ratio:.2%} | Rew: {ep_rew:.2f}")
        
        diag_results.append({
            "phi": phi,
            "seed": seed,
            "timestep": ts,
            "order_qty": o_qty,
            "inventory": inv_lvl,
            "lost_sales": ls_ratio,
            "true_reward": ep_rew
        })
        
    bw, ls, inv_var, economic_reward = run_evaluation_and_log_traces(model, eval_env, phi, seed)
    
    return bw, ls, inv_var, economic_reward, diag_results

if __name__ == "__main__":
    phi_values = [0.0, 0.5, 1.0]
    seeds = [0, 1]
    total_timesteps = 250_000
    
    results = []
    all_diagnostics = []
    os.makedirs("docs/results", exist_ok=True)
    
    total_runs = len(phi_values) * len(seeds)
    current_run = 0
    
    for phi in phi_values:
        for seed in seeds:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Running phi={phi}, seed={seed}...")
            
            bw, ls, inv_var, econ_rew, diags = train_phase1(phi, seed, total_timesteps)
            
            all_diagnostics.extend(diags)
            results.append({
                "phi": phi,
                "seed": seed,
                "bullwhip": bw,
                "lost_sales": ls,
                "inventory_variance": inv_var,
                "economic_reward": econ_rew
            })
            
            print(f"  -> Bullwhip: {bw:.2f}, LS: {ls:.2%}, Var: {inv_var:.2f}, Profit: {econ_rew:.2f}")
            
    pd.DataFrame(results).to_csv("docs/results/phase1_results_pilot.csv", index=False)
    
    diag_df = pd.DataFrame(all_diagnostics)
    diag_path = "docs/results/learning_diagnostics.csv"
    if os.path.exists(diag_path):
        existing = pd.read_csv(diag_path)
        diag_df = pd.concat([existing, diag_df], ignore_index=True)
    diag_df.to_csv(diag_path, index=False)
    
    print("\nPhase 1 Pilot complete! Saved to docs/results/phase1_results_pilot.csv")
