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
from experiments.phase1_5_diagnosis import record_learning_diagnostics

def run_cure_experiment(exp_name, global_weight, total_timesteps=100000):
    print(f"\nRunning {exp_name} with global_reward_weight={global_weight}")
    np.random.seed(0)
    
    # NO SCRIPTED NODES - testing if reward sharing alone can cure the collapse
    raw_env = MultiAgentNetInvMgmt()
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=[])
    
    agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
            
    train_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=global_weight)
    train_env = ss.pad_action_space_v0(train_env)
    train_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
    train_env = ss.concat_vec_envs_v1(train_env, 1, num_cpus=1, base_class='stable_baselines3')
    
    eval_raw = MultiAgentNetInvMgmt()
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=[])
    eval_env = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=global_weight)
    
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=256)
    
    diag_results = []
    blocks = 5
    steps_per_block = total_timesteps // blocks
    
    for block in range(blocks):
        model.learn(total_timesteps=steps_per_block, reset_num_timesteps=False)
        ts = (block + 1) * steps_per_block
        
        o_qty, inv_lvl, ls_ratio, ep_rew = record_learning_diagnostics(model, eval_env, num_episodes=3)
        print(f"  [{ts:6d}] PPO Mean Order: {o_qty:.2f} | Net Inv: {inv_lvl:.2f} | LS: {ls_ratio:.2%} | True Rew: {ep_rew:.2f}")
        
        diag_results.append({
            "experiment": exp_name,
            "weight": global_weight,
            "timestep": ts,
            "order_qty": o_qty,
            "inventory": inv_lvl,
            "lost_sales": ls_ratio,
            "true_reward": ep_rew,
        })
        
    return diag_results

if __name__ == "__main__":
    all_diags = []
    os.makedirs("docs/results", exist_ok=True)
    
    # C0: Baseline (w=0.0)
    diags_c0 = run_cure_experiment("C0_w0.0", 0.0)
    all_diags.extend(diags_c0)
    
    # C1: 10% Global (w=0.1)
    diags_c1 = run_cure_experiment("C1_w0.1", 0.1)
    all_diags.extend(diags_c1)
    
    # C2: 25% Global (w=0.25)
    diags_c2 = run_cure_experiment("C2_w0.25", 0.25)
    all_diags.extend(diags_c2)
    
    # C3: 50% Global (w=0.5)
    diags_c3 = run_cure_experiment("C3_w0.5", 0.5)
    all_diags.extend(diags_c3)
    
    # C4: Pure Team Reward (w=1.0)
    diags_c4 = run_cure_experiment("C4_w1.0", 1.0)
    all_diags.extend(diags_c4)
    
    df = pd.DataFrame(all_diags)
    df.to_csv("docs/results/phase1_5_5_cure.csv", index=False)
    print("\nPhase 1.5.5 Cure Study complete! Results saved to docs/results/phase1_5_5_cure.csv")
