import os
import sys
import numpy as np
import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.base_stock_wrapper import BaseStockWrapper
from experiments.phase3_validation import evaluate_and_extract_metrics

class ShiftedBaseStockWrapper(BaseStockWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_spaces = {
            agent: gym.spaces.Box(low=200.0, high=700.0, shape=self.env.action_space(agent).shape, dtype=np.float64)
            for agent in self.possible_agents
        }

def run_replication():
    print("\nStarting Phase 3.6 Replication: Shifted Base-Stock (5 Seeds)...")
    seeds = [42, 43, 44, 45, 46]
    
    results = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        np.random.seed(seed)
        
        raw_env = MultiAgentNetInvMgmt()
        scripted_nodes = ["node_1"]
        
        interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
        agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
        cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=0.0)
        
        train_env = ShiftedBaseStockWrapper(cpt_env)
        train_env = ss.pad_action_space_v0(train_env)
        train_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
        train_env = ss.concat_vec_envs_v1(train_env, 1, num_cpus=0, base_class='stable_baselines3')
        train_env.seed = lambda s: None
        
        eval_raw = MultiAgentNetInvMgmt()
        eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
        eval_cpt = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=0.0)
        eval_env = ShiftedBaseStockWrapper(eval_cpt)
        
        model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
        model.learn(total_timesteps=150000)
        
        o_qty, inv_lvl, ls_ratio, ep_rew, bw = evaluate_and_extract_metrics(model, eval_env, num_episodes=5)
        
        print(f"Seed {seed} -> Order Qty: {o_qty:.2f}, Profit: {ep_rew:.2f}, Bullwhip: {bw:.2f}")
        results.append((o_qty, inv_lvl, ls_ratio, ep_rew, bw))
        
    print("\n=== Phase 3.6 Replication Results (5 Seeds) ===")
    o_qtys, inv_lvls, ls_ratios, ep_rews, bws = zip(*results)
    print(f"PPO Mean Order Quantity : {np.mean(o_qtys):.2f} +/- {np.std(o_qtys):.2f}")
    print(f"Network Inventory Level : {np.mean(inv_lvls):.2f} +/- {np.std(inv_lvls):.2f}")
    print(f"Lost Sales Ratio        : {np.mean(ls_ratios):.2%}")
    print(f"True Economic Profit    : {np.mean(ep_rews):.2f} +/- {np.std(ep_rews):.2f}")
    print(f"Bullwhip Ratio          : {np.mean(bws):.2f} +/- {np.std(bws):.2f}")
    print("===============================================")
    
    if np.mean(ep_rews) > 0 and np.mean(ls_ratios) < 0.2:
        print("\nROBUSTNESS VERIFIED! The Shifted Action Range consistently solves the environment.")
    else:
        print("\nREPLICATION FAILED! The +330 profit was a statistical mirage.")

if __name__ == "__main__":
    run_replication()
