import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_phase0():
    data_path = "docs/results/phase0_data.csv"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Convert lambda to string for categorical hue
    df['lambda'] = df['lambda'].astype(str)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Order Qty
    sns.lineplot(data=df, x="timestep", y="order_qty", hue="lambda", palette="flare", ax=axes[0, 0])
    axes[0, 0].set_title("Average Order Quantity")
    
    # Inventory
    sns.lineplot(data=df, x="timestep", y="inventory", hue="lambda", palette="flare", ax=axes[0, 1])
    axes[0, 1].set_title("Mean Inventory Level")
    
    # Backlog
    sns.lineplot(data=df, x="timestep", y="backlog", hue="lambda", palette="flare", ax=axes[1, 0])
    axes[1, 0].set_title("Mean Backlog / Stockouts")
    
    # Reward
    sns.lineplot(data=df, x="timestep", y="true_reward", hue="lambda", palette="flare", ax=axes[1, 1])
    axes[1, 1].set_title("True Unshaped Reward")
    
    plt.tight_layout()
    os.makedirs("docs/assets", exist_ok=True)
    plt.savefig("docs/assets/phase0_validation.png", dpi=150)
    print("Saved plot to docs/assets/phase0_validation.png")

if __name__ == "__main__":
    plot_phase0()
