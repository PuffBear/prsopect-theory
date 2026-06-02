import os
import sys
import numpy as np
import pandas as pd
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper

def evaluate_and_extract_metrics(model, eval_env, num_episodes=5):
    """
    Runs the deterministic policy for num_episodes and extracts Bullwhip, Profit, etc.
    """
    total_order_qty = []
    total_inv = []
    total_lost_sales = []
    total_reward = []
    
    # Track order arrays per agent for Bullwhip
    agent_orders = {agent: [] for agent in eval_env.possible_agents}
    
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_rew = 0
        
        while eval_env.agents:
            actions = {}
            for agent in eval_env.agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    real_shape = eval_env.action_space(agent).shape[0]
                    actions[agent] = action[:real_shape]
                    
            step_actions = actions.copy()
            obs, rewards, term, trunc, info = eval_env.step(step_actions)
            for agent in rewards:
                ep_rew += info[agent]["true_reward"]
            
            for agent, action in actions.items():
                total_order_qty.extend(action)
                agent_orders[agent].append(np.sum(action))
                
        # Unpack environments to grab raw matrix state
        base_env = eval_env.env.env.unwrapped_env if hasattr(eval_env.env, 'env') else eval_env.unwrapped.unwrapped_env
        total_inv.append(base_env.X.mean().mean())
        
        total_D = base_env.D.sum().sum()
        total_U = base_env.U.sum().sum()
        ls_ratio = total_U / total_D if total_D > 0 else 0
        total_lost_sales.append(ls_ratio)
        total_reward.append(ep_rew)
        
    mean_order = np.mean(total_order_qty) if total_order_qty else 0.0
    
    # Calculate Bullwhip Ratio
    # Base network demand variance is approximately 20 (Poisson with lambda=20)
    market_demand_variance = 20.0
    bullwhips = []
    for agent, orders in agent_orders.items():
        if len(orders) > 0:
            agent_var = np.var(orders)
            bullwhips.append(agent_var / market_demand_variance)
            
    mean_bullwhip = np.mean(bullwhips) if bullwhips else 1.0
    
    return mean_bullwhip, mean_order, np.mean(total_inv), np.mean(total_lost_sales), np.mean(total_reward)


def run_sweep_configuration(protocol, phi, total_timesteps=250000, seed=42):
    print(f"\n[{protocol}] Running phi={phi}...")
    np.random.seed(seed)
    
    raw_env = MultiAgentNetInvMgmt()
    
    scripted_nodes = ["node_1"] if protocol == "scripted" else []
    global_weight = 0.0 if protocol == "scripted" else 1.0
    
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    ppo_agents = interv_env.possible_agents
    
    # Assign Loss Aversion
    num_averse = int(phi * len(ppo_agents))
    averse_agents = np.random.choice(ppo_agents, size=num_averse, replace=False)
    
    agent_params = {}
    for agent in raw_env.agents:
        if agent in averse_agents:
            agent_params[agent] = {"lambda": 5.0, "alpha": 1.0, "beta": 1.0}
        else:
            agent_params[agent] = {"lambda": 1.0, "alpha": 1.0, "beta": 1.0}
            
    train_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=global_weight)
    train_env = ss.pad_action_space_v0(train_env)
    train_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
    train_env = ss.concat_vec_envs_v1(train_env, 1, num_cpus=1, base_class='stable_baselines3')
    
    eval_raw = MultiAgentNetInvMgmt()
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
    eval_env = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=global_weight)
    
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=256)
    model.learn(total_timesteps=total_timesteps)
    
    bullwhip, o_qty, inv_lvl, ls_ratio, ep_rew = evaluate_and_extract_metrics(model, eval_env, num_episodes=5)
    print(f"  Result -> Bullwhip: {bullwhip:.2f} | Order: {o_qty:.2f} | LS: {ls_ratio:.2%} | Profit: {ep_rew:.2f}")
    
    return {
        "protocol": protocol,
        "phi": phi,
        "bullwhip": bullwhip,
        "order_qty": o_qty,
        "inventory": inv_lvl,
        "lost_sales": ls_ratio,
        "profit": ep_rew,
    }

if __name__ == "__main__":
    phi_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    protocols = ["scripted", "team_reward"]
    
    all_results = []
    os.makedirs("docs/results", exist_ok=True)
    
    for protocol in protocols:
        for phi in phi_values:
            res = run_sweep_configuration(protocol, phi, total_timesteps=250000, seed=42)
            all_results.append(res)
            
    df = pd.DataFrame(all_results)
    df.to_csv("docs/results/phase1_double_sweep.csv", index=False)
    print("\nPhase 1 Double Sweep complete! Results saved to docs/results/phase1_double_sweep.csv")
