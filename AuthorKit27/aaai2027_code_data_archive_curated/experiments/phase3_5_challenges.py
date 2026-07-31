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

def setup_and_run(challenge_name, modify_env_fn, wrapper_class=BaseStockWrapper, seed=42):
    print(f"\n==============================================")
    print(f"Running Challenge: {challenge_name}")
    print(f"==============================================")
    np.random.seed(seed)
    
    raw_env = MultiAgentNetInvMgmt()
    scripted_nodes = ["node_1"]
    
    # Apply challenge modifications to the raw environment
    modify_env_fn(raw_env.unwrapped_env)
    
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
    cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    
    # Use specified Base-Stock wrapper
    train_env = wrapper_class(cpt_env)
    train_env = ss.pad_action_space_v0(train_env)
    train_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
    train_env = ss.concat_vec_envs_v1(train_env, 1, num_cpus=1, base_class='stable_baselines3')
    train_env.seed = lambda s: None
    
    eval_raw = MultiAgentNetInvMgmt()
    modify_env_fn(eval_raw.unwrapped_env)
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
    eval_cpt = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=0.0)
    eval_env = wrapper_class(eval_cpt)
    
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
    model.learn(total_timesteps=150000)
    
    o_qty, inv_lvl, ls_ratio, ep_rew, bw = evaluate_and_extract_metrics(model, eval_env, num_episodes=5)
    
    print(f"\nResults for {challenge_name}:")
    print(f"PPO Mean Order Quantity : {o_qty:.2f}")
    print(f"Network Inventory Level : {inv_lvl:.2f}")
    print(f"Lost Sales Ratio        : {ls_ratio:.2%}")
    print(f"True Economic Profit    : {ep_rew:.2f}")
    print(f"Bullwhip Ratio          : {bw:.2f}")
    
    if bw > 0.01:
        print(f"-> SUCCESS: {challenge_name} fixed the Gradient Dead Zone!")
    else:
        print(f"-> FAILURE: PPO still collapsed under {challenge_name}.")

def empty_warehouse_mod(base_env):
    for j in base_env.graph.nodes():
        if 'I0' in base_env.graph.nodes[j]:
            base_env.graph.nodes[j]['I0'] = 0
    base_env.reset()

def short_episode_mod(base_env):
    base_env.num_periods = 100
    base_env.reset()

def no_mod(base_env):
    pass

if __name__ == "__main__":
    # Challenge 1: Empty Warehouse (I0 = 0)
    setup_and_run("The Empty Warehouse (I0 = 0)", empty_warehouse_mod)
    
    # Challenge 2: Short Episode (num_periods = 100)
    setup_and_run("The Short Episode (100 Periods)", short_episode_mod)
    
    # Challenge 3: Action Range (S in [200, 700])
    setup_and_run("The Action Range (S in [200, 700])", no_mod, wrapper_class=ShiftedBaseStockWrapper)
