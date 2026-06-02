import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_cure():
    data_path = "docs/results/phase1_5_5_cure.csv"
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    df = pd.read_csv(data_path)
    os.makedirs("docs/phase1_figures", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="timestep", y="order_qty", hue="experiment", palette="viridis", linewidth=2.5)
    plt.title("Exp 1.5.5: Global Reward Sharing Cure Study")
    plt.ylabel("Mean Order Quantity (PPO Agents)")
    plt.savefig("docs/phase1_figures/exp_cure_order_qty.png", dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="timestep", y="lost_sales", hue="experiment", palette="viridis", linewidth=2.5)
    plt.title("Exp 1.5.5: Lost Sales vs Reward Sharing")
    plt.ylabel("Lost Sales Ratio")
    plt.savefig("docs/phase1_figures/exp_cure_lost_sales.png", dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="timestep", y="true_reward", hue="experiment", palette="viridis", linewidth=2.5)
    plt.title("Exp 1.5.5: True Economic Profit vs Reward Sharing")
    plt.ylabel("True Reward")
    plt.savefig("docs/phase1_figures/exp_cure_true_reward.png", dpi=150)
    plt.close()
    
    print("Saved Cure plots to docs/phase1_figures/")

if __name__ == "__main__":
    plot_cure()
