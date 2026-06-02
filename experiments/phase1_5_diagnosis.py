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

def record_learning_diagnostics(model, env, num_episodes=3):
    total_order_qty = []
    total_inv = []
    total_lost_sales = []
    total_reward = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0
        
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    real_shape = env.action_space(agent).shape[0]
                    actions[agent] = action[:real_shape]
                    
            obs, rewards, term, trunc, info = env.step(actions)
            for agent in rewards:
                ep_rew += info[agent]["true_reward"]
            
            for agent, action in actions.items():
                total_order_qty.extend(action)
                
        # Unpack back to base env to grab exact metrics
        base_env = env.env.env.unwrapped_env
        total_inv.append(base_env.X.mean().mean())
        
        total_D = base_env.D.sum().sum()
        total_U = base_env.U.sum().sum()
        ls_ratio = total_U / total_D if total_D > 0 else 0
        total_lost_sales.append(ls_ratio)
        total_reward.append(ep_rew)
        
    mean_order = np.mean(total_order_qty) if total_order_qty else 0.0
    return mean_order, np.mean(total_inv), np.mean(total_lost_sales), np.mean(total_reward)

def run_experiment(exp_name, scripted_nodes, total_timesteps=100000, model_to_resume=None):
    print(f"\nRunning {exp_name} with scripted_nodes={scripted_nodes}")
    np.random.seed(0)
    
    raw_env = MultiAgentNetInvMgmt()
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    
    agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
            
    train_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0)
    train_env = ss.pad_action_space_v0(train_env)
    train_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
    train_env = ss.concat_vec_envs_v1(train_env, 1, num_cpus=1, base_class='stable_baselines3')
    
    eval_raw = MultiAgentNetInvMgmt()
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
    eval_env = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0)
    
    if model_to_resume is not None:
        model = PPO.load(model_to_resume, env=train_env)
    else:
        model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=256)
    
    diag_results = []
    blocks = 5
    steps_per_block = total_timesteps // blocks
    
    for block in range(blocks):
        model.learn(total_timesteps=steps_per_block, reset_num_timesteps=False)
        ts = (block + 1) * steps_per_block
        
        o_qty, inv_lvl, ls_ratio, ep_rew = record_learning_diagnostics(model, eval_env, num_episodes=3)
        print(f"  [{ts:6d}] PPO Mean Order: {o_qty:.2f} | Net Inv: {inv_lvl:.2f} | LS: {ls_ratio:.2%} | PPO Rew: {ep_rew:.2f}")
        
        diag_results.append({
            "experiment": exp_name,
            "timestep": ts,
            "order_qty": o_qty,
            "inventory": inv_lvl,
            "lost_sales": ls_ratio,
            "true_reward": ep_rew,
        })
        
    return diag_results, model

if __name__ == "__main__":
    all_diags = []
    os.makedirs("docs/results", exist_ok=True)
    
    # Experiment A: Minimal Intervention Curve
    diags_a0, _ = run_experiment("A0_0_scripted", [])
    all_diags.extend(diags_a0)
    
    diags_a1, _ = run_experiment("A1_1_scripted", ["node_1"])
    all_diags.extend(diags_a1)
    
    diags_a2, _ = run_experiment("A2_3_scripted", ["node_1", "node_2", "node_3"])
    all_diags.extend(diags_a2)
    
    # Experiment B: Location Sensitivity
    diags_b2, _ = run_experiment("B2_Distributors", ["node_2", "node_3"])
    all_diags.extend(diags_b2)
    
    diags_b3, _ = run_experiment("B3_Factories", ["node_4", "node_5", "node_6"])
    all_diags.extend(diags_b3)
    
    # Experiment C: Freeze-and-Release
    print("\nRunning Experiment C: Freeze-and-Release...")
    diags_c_freeze, trained_model = run_experiment("C_Freeze", ["node_1"], total_timesteps=100000)
    all_diags.extend(diags_c_freeze)
    
    # Save and load to dynamically adjust n_envs (from 5 back to 6)
    trained_model.save("docs/results/temp_freeze_model")
    
    diags_c_release, _ = run_experiment("C_Release", [], total_timesteps=100000, model_to_resume="docs/results/temp_freeze_model")
    for d in diags_c_release:
        d["timestep"] += 100000
    all_diags.extend(diags_c_release)
    
    df = pd.DataFrame(all_diags)
    df.to_csv("docs/results/phase1_5_diagnosis.csv", index=False)
    print("\nPhase 1.5 Diagnosis complete! Results saved to docs/results/phase1_5_diagnosis.csv")
