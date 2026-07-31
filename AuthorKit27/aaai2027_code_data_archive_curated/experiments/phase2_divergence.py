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
    Runs the deterministic policy for num_episodes and extracts Profit, etc.
    """
    total_order_qty = []
    total_inv = []
    total_lost_sales = []
    total_reward = []
    
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
                
        # Unpack environments to grab raw matrix state
        base_env = eval_env.env.env.unwrapped_env if hasattr(eval_env.env, 'env') else eval_env.unwrapped.unwrapped_env
        total_inv.append(base_env.X.mean().mean())
        
        total_D = base_env.D.sum().sum()
        total_U = base_env.U.sum().sum()
        ls_ratio = total_U / total_D if total_D > 0 else 0
        total_lost_sales.append(ls_ratio)
        total_reward.append(ep_rew)
        
    mean_order = np.mean(total_order_qty) if total_order_qty else 0.0
    return mean_order, np.mean(total_inv), np.mean(total_lost_sales), np.mean(total_reward)


def run_divergence_configuration(lam, total_timesteps=150000, seed=42):
    print(f"\nRunning lambda={lam}...")
    np.random.seed(seed)
    
    raw_env = MultiAgentNetInvMgmt()
    scripted_nodes = ["node_1"]
    global_weight = 0.0
    
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    
    # 3 out of 5 nodes are averse (60%)
    averse_agents = ['node_2', 'node_4', 'node_6']
    
    agent_params = {}
    for agent in raw_env.agents:
        if agent in averse_agents:
            agent_params[agent] = {"lambda": lam, "alpha": 1.0, "beta": 1.0}
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
    
    o_qty, inv_lvl, ls_ratio, ep_rew = evaluate_and_extract_metrics(model, eval_env, num_episodes=5)
    print(f"  Result -> Order: {o_qty:.2f} | Inv: {inv_lvl:.2f} | LS: {ls_ratio:.2%} | Profit: {ep_rew:.2f}")
    
    return {
        "lambda": lam,
        "order_qty": o_qty,
        "inventory": inv_lvl,
        "lost_sales": ls_ratio,
        "profit": ep_rew,
    }

if __name__ == "__main__":
    lambda_values = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    
    all_results = []
    os.makedirs("docs/results", exist_ok=True)
    
    for lam in lambda_values:
        res = run_divergence_configuration(lam, total_timesteps=150000, seed=42)
        all_results.append(res)
            
    df = pd.DataFrame(all_results)
    df.to_csv("docs/results/phase2_divergence.csv", index=False)
    print("\nPhase 2 Divergence Sweep complete! Results saved to docs/results/phase2_divergence.csv")
