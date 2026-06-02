import os
import sys
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.base_stock_wrapper import BaseStockWrapper

def evaluate_and_extract_metrics(model, eval_env, num_episodes=5):
    """
    Runs the deterministic policy for num_episodes and extracts Bullwhip, Profit, etc.
    """
    total_order_qty = []
    total_inv = []
    total_lost_sales = []
    total_reward = []
    
    # For Bullwhip Ratio
    agent_orders_var = []
    demand_var = []
    
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_rew = 0
        
        ep_orders = []
        ep_demand = []
        
        while eval_env.agents:
            actions = {}
            for agent in eval_env.agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    real_shape = eval_env.action_space(agent).shape[0]
                    actions[agent] = action[:real_shape]
                    
            step_actions = actions.copy()
            obs, rewards, term, trunc, info = eval_env.step(step_actions)
            
            # The BaseStockWrapper translates the S_action into a real Order Quantity.
            # We want to measure the variance of the TRUE order quantities!
            # The true orders placed by the PPO nodes are logged in the underlying env.
            # Get the unwrapped env. Depending on wrapper nesting, we must dig down.
            curr_env = eval_env
            while hasattr(curr_env, 'env') or hasattr(curr_env, 'unwrapped_env'):
                if hasattr(curr_env, 'unwrapped_env'):
                    base_env = curr_env.unwrapped_env
                    break
                curr_env = curr_env.env
                
            t = base_env.period - 1
            
            # Calculate true PPO orders for this step
            step_ppo_orders = []
            for agent in eval_env.possible_agents:
                node_id = eval_env.unwrapped.agent_name_to_id[agent]
                for link in base_env.reorder_links:
                    if link[1] == node_id:
                        step_ppo_orders.append(base_env.R.loc[t, link])
            
            ep_orders.append(np.mean(step_ppo_orders))
            
            # True market demand
            ep_demand.append(base_env.D.loc[t, :].sum())
            
            for agent in rewards:
                ep_rew += info[agent]["true_reward"]
            
            total_order_qty.extend(step_ppo_orders)
                
        total_inv.append(base_env.X.mean().mean())
        
        total_D = base_env.D.sum().sum()
        total_U = base_env.U.sum().sum()
        ls_ratio = total_U / total_D if total_D > 0 else 0
        total_lost_sales.append(ls_ratio)
        total_reward.append(ep_rew)
        
        if np.var(ep_demand) > 0:
            agent_orders_var.append(np.var(ep_orders) / np.var(ep_demand))
        else:
            agent_orders_var.append(0)
            
    mean_order = np.mean(total_order_qty) if total_order_qty else 0.0
    return mean_order, np.mean(total_inv), np.mean(total_lost_sales), np.mean(total_reward), np.mean(agent_orders_var)


def run_phase3_validation():
    print("\nStarting Phase 3 Validation: Base-Stock Parameterization (5 Seeds)...")
    seeds = [42, 43, 44, 45, 46]
    
    results = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        np.random.seed(seed)
        
        raw_env = MultiAgentNetInvMgmt()
        scripted_nodes = ["node_1"]
        
        interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
        
        # All Rational Agents (phi=0%)
        agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
                
        cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=0.0)
        
        # Wrap in BaseStock!
        train_env = BaseStockWrapper(cpt_env)
        
        train_env = ss.pad_action_space_v0(train_env)
        train_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
        train_env = ss.concat_vec_envs_v1(train_env, 1, num_cpus=1, base_class='stable_baselines3')
        train_env.seed = lambda s: None
        
        eval_raw = MultiAgentNetInvMgmt()
        eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
        eval_cpt = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=0.0)
        eval_env = BaseStockWrapper(eval_cpt)
        
        model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
        model.learn(total_timesteps=150000)
        
        o_qty, inv_lvl, ls_ratio, ep_rew, bw = evaluate_and_extract_metrics(model, eval_env, num_episodes=5)
        
        print(f"Seed {seed} -> Order Qty: {o_qty:.2f}, Profit: {ep_rew:.2f}, Bullwhip: {bw:.2f}")
        results.append((o_qty, inv_lvl, ls_ratio, ep_rew, bw))
        
    print("\n=== Phase 3 Validation Results (5 Seeds) ===")
    o_qtys, inv_lvls, ls_ratios, ep_rews, bws = zip(*results)
    print(f"PPO Mean Order Quantity : {np.mean(o_qtys):.2f} ± {np.std(o_qtys):.2f}")
    print(f"Network Inventory Level : {np.mean(inv_lvls):.2f} ± {np.std(inv_lvls):.2f}")
    print(f"Lost Sales Ratio        : {np.mean(ls_ratios):.2%}")
    print(f"True Economic Profit    : {np.mean(ep_rews):.2f} ± {np.std(ep_rews):.2f}")
    print(f"Bullwhip Ratio          : {np.mean(bws):.2f} ± {np.std(bws):.2f}")
    print("============================================")
    
    if np.mean(bws) > 0.01:
        print("\nSUCCESS! The network exhibited non-zero Bullwhip variance!")
    else:
        print("\nFAILURE! The network consistently collapsed to a zero-variance policy across all seeds.")

if __name__ == "__main__":
    run_phase3_validation()
