import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def record_learning_diagnostics(model, eval_env, num_episodes=1):
    all_orders = []
    all_inv = []
    all_ls = []
    all_true_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_true_reward = 0
        
        while eval_env.agents:
            actions = {}
            for agent in eval_env.agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    real_shape = eval_env.action_space(agent).shape[0]
                    actions[agent] = action[:real_shape]
                    all_orders.append(np.mean(action))
                    
            base_env = eval_env.env.unwrapped_env
            all_inv.append(np.mean(base_env.X.iloc[base_env.period].values))
            
            # calculate lost sales in this step
            if hasattr(base_env, 'U'):
                all_ls.append(np.mean(base_env.U.iloc[base_env.period].values))
            elif hasattr(base_env, 'LS'):
                all_ls.append(np.mean(base_env.LS.iloc[base_env.period].values))
                
            obs, rewards, term, trunc, info = eval_env.step(actions)
            
            # extract true reward from info if we stored it, else just use the unshaped reward calculation
            # for now, we will track mean of rewards returned by environment (which are shaped), 
            # or we can pull true unshaped reward. Our wrapper returns true_reward in info.
            for agent in rewards.keys():
                if "true_reward" in info.get(agent, {}):
                    ep_true_reward += info[agent]["true_reward"]
                else:
                    # fallback to just rewards if true_reward isn't exposed properly
                    ep_true_reward += rewards[agent]
                    
        all_true_rewards.append(ep_true_reward)
        
    return np.mean(all_orders), np.mean(all_inv), np.mean(all_ls), np.mean(all_true_rewards)

def plot_learning_diagnostics(csv_path="docs/results/learning_diagnostics.csv"):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    # Combine phi and seed to uniquely identify runs
    df['run_id'] = df['phi'].astype(str) + "_s" + df['seed'].astype(str)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.lineplot(data=df, x="timestep", y="order_qty", hue="phi", ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Mean Order Quantity over Training")
    
    sns.lineplot(data=df, x="timestep", y="inventory", hue="phi", ax=axes[0, 1], palette="viridis")
    axes[0, 1].set_title("Mean Inventory Level over Training")
    
    sns.lineplot(data=df, x="timestep", y="lost_sales", hue="phi", ax=axes[1, 0], palette="viridis")
    axes[1, 0].set_title("Mean Lost Sales over Training")
    
    sns.lineplot(data=df, x="timestep", y="true_reward", hue="phi", ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Mean Episode Reward over Training")
    
    plt.tight_layout()
    os.makedirs("docs/phase1_figures", exist_ok=True)
    plt.savefig("docs/phase1_figures/learning_diagnostics.png", dpi=150)
    print("Saved diagnostics plot to docs/phase1_figures/learning_diagnostics.png")
