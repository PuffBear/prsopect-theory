import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.offset_base_stock_wrapper import OffsetBaseStockWrapper

def _run_single_seed(args):
    seed, S_init = args
    np.random.seed(seed)
    
    # Training Environment
    raw_env = MultiAgentNetInvMgmt()
    scripted_nodes = ["node_1"]
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    # alpha=0.0 because we want to test the decentralized deceptive landscape
    agent_params = {agent: {"lambda": 1.0, "alpha": 0.0, "beta": 1.0} for agent in raw_env.agents}
    cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    
    log_file = f"/tmp/offset_logs_{S_init}_{seed}.csv"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    train_env = OffsetBaseStockWrapper(cpt_env, S_init=S_init, action_scale=250.0, log_file=log_file)
    
    vec_env = ss.pad_action_space_v0(train_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(vec_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=0, base_class='stable_baselines3')
    vec_env.seed = lambda s: None
    
    # Train PPO
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=150000)
    
    # Extract trajectory
    df = pd.read_csv(log_file)
    df_agent = df[df['agent'] == 'node_2'].sort_values('step')
    
    # Compute the converged S (average of last 10,000 steps)
    final_S = df_agent.tail(10000)['S'].mean()
    
    return S_init, seed, final_S, df_agent

def run_basin_study():
    print(f"\n=======================================================")
    print(f"Starting Phase 3.96: Basin-of-Attraction Study")
    print(f"=======================================================")
    
    S_inits = [0, 25, 50, 80, 150, 250]
    seeds = [42, 43, 44]
    
    args_list = []
    for S_init in S_inits:
        for seed in seeds:
            args_list.append((seed, S_init))
            
    results = {}
    trajectories = {}
    
    with mp.Pool(processes=min(mp.cpu_count(), 10)) as pool:
        for res in pool.imap_unordered(_run_single_seed, args_list):
            S_init, seed, final_S, df_agent = res
            print(f"S_init={S_init:3d} | Seed={seed} -> Converged S={final_S:.1f}")
            
            if S_init not in results:
                results[S_init] = []
                trajectories[S_init] = []
            results[S_init].append(final_S)
            trajectories[S_init].append(df_agent)
            
    # Plotting
    os.makedirs("docs/phase3_96_figures", exist_ok=True)
    plt.figure(figsize=(15, 10))
    
    for i, S_init in enumerate(S_inits):
        plt.subplot(2, 3, i+1)
        
        for df_agent in trajectories[S_init]:
            # Smooth with a rolling window of 1000 steps
            S_smooth = df_agent['S'].rolling(1000).mean()
            plt.plot(df_agent['step'], S_smooth, linewidth=2, alpha=0.8)
            
        plt.title(f"Init: S={S_init}\nConverged S = {np.mean(results[S_init]):.1f}")
        plt.xlabel("Training Step")
        plt.ylabel("Target Base-Stock (S)")
        plt.ylim(-10, 300)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='S=0 Basin')
        plt.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='S=80 Peak')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig("docs/phase3_96_figures/basin_attraction.png", dpi=300)
    print("Saved docs/phase3_96_figures/basin_attraction.png")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    run_basin_study()
