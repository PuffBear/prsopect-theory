import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper
from experiments.phase3_7_verification import evaluate_and_extract_metrics

class MetricsCallback(BaseCallback):
    """
    Custom callback to extract internal learning metrics (Value Loss, Explained Variance)
    to prove temporal credit assignment difficulty.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.value_losses = []
        self.explained_variances = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is not None:
            # The logger stores recent values
            nv = self.logger.name_to_value
            if "train/value_loss" in nv:
                self.value_losses.append(nv["train/value_loss"])
            if "train/explained_variance" in nv:
                self.explained_variances.append(nv["train/explained_variance"])

def _run_single_config(args):
    seed, lead_time, alpha = args
    np.random.seed(seed)
    
    # We fix bounds to [0, 500] to observe the collapse
    low, high = 0.0, 500.0
    log_file = f"/tmp/leadtime_{lead_time}_{alpha}_{seed}.csv"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    raw_env = MultiAgentNetInvMgmt(uniform_lead_time=lead_time)
    scripted_nodes = ["node_1"]
    interv_env = DiagnosticWrapper(raw_env, scripted_nodes=scripted_nodes)
    agent_params = {agent: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0} for agent in raw_env.agents}
    
    cpt_env = CPTRewardWrapper(interv_env, agent_params, reward_scale=1.0, global_reward_weight=alpha)
    train_env = ScaledBaseStockWrapper(cpt_env, low_bound=low, high_bound=high, log_file=log_file)
    
    vec_env = ss.pad_action_space_v0(train_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(vec_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=0, base_class='stable_baselines3')
    vec_env.seed = lambda s: None
    
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=128, batch_size=256, seed=seed)
    
    callback = MetricsCallback()
    model.learn(total_timesteps=150000, callback=callback)
    
    eval_raw = MultiAgentNetInvMgmt(uniform_lead_time=lead_time)
    eval_interv = DiagnosticWrapper(eval_raw, scripted_nodes=scripted_nodes)
    eval_cpt = CPTRewardWrapper(eval_interv, agent_params, reward_scale=1.0, global_reward_weight=alpha)
    eval_env = ScaledBaseStockWrapper(eval_cpt, low_bound=low, high_bound=high)
    
    # Vary evaluation seed
    o_qty, inv_lvl, ls_ratio, ep_rew, bw = evaluate_and_extract_metrics(model, eval_env, num_episodes=5, base_seed=seed*10)
    
    mean_vloss = np.mean(callback.value_losses) if callback.value_losses else 0.0
    mean_exp_var = np.mean(callback.explained_variances) if callback.explained_variances else 0.0
    
    # Extract learned S
    learned_S = 0.0
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if not df.empty:
            # average over the last 10000 steps to get the converged learned S
            df_agent = df[df['agent'] == 'node_2']
            learned_S = df_agent.tail(10000)['S'].mean()
            
    return seed, lead_time, alpha, learned_S, ep_rew, ls_ratio, mean_vloss, mean_exp_var

def run_leadtime_sweep():
    lead_times = [0, 1, 2, 4, 8]
    alphas = [0.0, 1.0]
    seeds = list(range(42, 47))
    
    args_list = []
    for lead_time in lead_times:
        for alpha in alphas:
            for seed in seeds:
                args_list.append((seed, lead_time, alpha))
                
    print(f"\n=======================================================")
    print(f"Starting Phase 3.9 Lead-Time Causality Sweep (Total Runs: {len(args_list)})")
    print(f"=======================================================")
    
    results = {}
    
    with mp.Pool(processes=min(mp.cpu_count(), 10)) as pool:
        for res in pool.imap_unordered(_run_single_config, args_list):
            seed, lead_time, alpha, learned_S, ep_rew, ls_ratio, mean_vloss, mean_exp_var = res
            
            config_key = (lead_time, alpha)
            if config_key not in results:
                results[config_key] = []
            results[config_key].append((learned_S, ep_rew, ls_ratio, mean_vloss, mean_exp_var))
            
            print(f"L:{lead_time} Alpha:{alpha} Seed:{seed} -> S:{learned_S:.1f}, Profit:{ep_rew:.1f}, LS:{ls_ratio:.1%}, VLoss:{mean_vloss:.1f}, ExpVar:{mean_exp_var:.3f}")
            
    print("\n=== Phase 3.9 Lead-Time Causality Results ===")
    
    # Save results to a dataframe for easy plotting later
    records = []
    for lead_time in lead_times:
        for alpha in alphas:
            res_list = results[(lead_time, alpha)]
            S_vals, ep_rews, ls_ratios, vlosses, exp_vars = zip(*res_list)
            records.append({
                'LeadTime': lead_time,
                'Alpha': alpha,
                'Mean_S': np.mean(S_vals),
                'Std_S': np.std(S_vals),
                'Mean_Profit': np.mean(ep_rews),
                'Std_Profit': np.std(ep_rews),
                'Mean_VLoss': np.mean(vlosses),
                'Mean_ExpVar': np.mean(exp_vars)
            })
            print(f"Lead Time: {lead_time} | Alpha: {alpha}")
            print(f"  Learned S : {np.mean(S_vals):.2f} ± {np.std(S_vals):.2f}")
            print(f"  Profit    : {np.mean(ep_rews):.2f} ± {np.std(ep_rews):.2f}")
            print(f"  LS %      : {np.mean(ls_ratios):.2%} ± {np.std(ls_ratios):.2%}")
            print(f"  VLoss     : {np.mean(vlosses):.2f}")
            print(f"  ExpVar    : {np.mean(exp_vars):.3f}")
            print("----------------------------------------------")
            
    df_results = pd.DataFrame(records)
    df_results.to_csv("docs/phase3_9_figures/leadtime_results.csv", index=False)
    
def plot_leadtime_results():
    os.makedirs("docs/phase3_9_figures", exist_ok=True)
    df = pd.read_csv("docs/phase3_9_figures/leadtime_results.csv")
    
    lead_times = sorted(df['LeadTime'].unique())
    alphas = sorted(df['Alpha'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for alpha in alphas:
        df_a = df[df['Alpha'] == alpha]
        label = "Decentralized (α=0)" if alpha == 0.0 else "Centralized (α=1)"
        color = 'red' if alpha == 0.0 else 'blue'
        
        axes[0, 0].errorbar(df_a['LeadTime'], df_a['Mean_S'], yerr=df_a['Std_S'], marker='o', label=label, color=color, capsize=5, linewidth=2)
        axes[0, 1].errorbar(df_a['LeadTime'], df_a['Mean_Profit'], yerr=df_a['Std_Profit'], marker='s', label=label, color=color, capsize=5, linewidth=2)
        axes[1, 0].plot(df_a['LeadTime'], df_a['Mean_VLoss'], marker='^', label=label, color=color, linewidth=2)
        axes[1, 1].plot(df_a['LeadTime'], df_a['Mean_ExpVar'], marker='d', label=label, color=color, linewidth=2)
        
    axes[0, 0].set_title("Learned Base-Stock (S) vs Lead Time")
    axes[0, 0].set_xlabel("Lead Time (Timesteps)")
    axes[0, 0].set_ylabel("Learned S")
    axes[0, 0].set_xticks(lead_times)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title("True Economic Profit vs Lead Time")
    axes[0, 1].set_xlabel("Lead Time (Timesteps)")
    axes[0, 1].set_ylabel("Profit")
    axes[0, 1].set_xticks(lead_times)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title("Value Loss (Credit Assignment Difficulty)")
    axes[1, 0].set_xlabel("Lead Time (Timesteps)")
    axes[1, 0].set_ylabel("Mean Value Loss")
    axes[1, 0].set_xticks(lead_times)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title("Explained Variance")
    axes[1, 1].set_xlabel("Lead Time (Timesteps)")
    axes[1, 1].set_ylabel("Mean Explained Variance")
    axes[1, 1].set_xticks(lead_times)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("docs/phase3_9_figures/leadtime_causality.png", dpi=300)
    print("Saved docs/phase3_9_figures/leadtime_causality.png")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.makedirs("docs/phase3_9_figures", exist_ok=True)
    run_leadtime_sweep()
    plot_leadtime_results()
