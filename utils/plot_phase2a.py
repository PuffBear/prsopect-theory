import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_phase2a():
    data_path = "docs/results/phase2a_replication.csv"
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    df = pd.read_csv(data_path)
    os.makedirs("docs/phase2a_figures", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Mean Order Quantity with CI and scatter
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="lambda", y="order_qty", errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5, color="darkorange", label="Mean & 95% CI")
    sns.scatterplot(data=df, x="lambda", y="order_qty", color="black", s=50, alpha=0.6, label="Individual Seeds")
    plt.title("Phase 2A: PPO Mean Order vs Loss Aversion (λ)", fontsize=16)
    plt.ylabel("Mean Order Quantity")
    plt.xlabel("Loss Aversion Intensity (λ) for Averse Agents")
    plt.legend()
    plt.savefig("docs/phase2a_figures/exp_phase2a_order_qty.png", dpi=150)
    plt.close()
    
    # Plot 2: Inventory Hoarding
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="lambda", y="inventory", errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5, color="crimson", label="Mean & 95% CI")
    sns.scatterplot(data=df, x="lambda", y="inventory", color="black", s=50, alpha=0.6, label="Individual Seeds")
    plt.title("Phase 2A: Network Inventory vs Loss Aversion (λ)", fontsize=16)
    plt.ylabel("Mean Network Inventory Level")
    plt.xlabel("Loss Aversion Intensity (λ) for Averse Agents")
    plt.legend()
    plt.savefig("docs/phase2a_figures/exp_phase2a_inventory.png", dpi=150)
    plt.close()

    # Plot 3: Economic Profit
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="lambda", y="profit", errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5, color="navy", label="Mean & 95% CI")
    sns.scatterplot(data=df, x="lambda", y="profit", color="black", s=50, alpha=0.6, label="Individual Seeds")
    plt.title("Phase 2A: Economic Profit vs Loss Aversion (λ)", fontsize=16)
    plt.ylabel("True Economic Profit")
    plt.xlabel("Loss Aversion Intensity (λ) for Averse Agents")
    plt.legend()
    plt.savefig("docs/phase2a_figures/exp_phase2a_profit.png", dpi=150)
    plt.close()
    
    # Plot 4: Lost Sales
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="lambda", y="lost_sales", errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5, color="purple", label="Mean & 95% CI")
    sns.scatterplot(data=df, x="lambda", y="lost_sales", color="black", s=50, alpha=0.6, label="Individual Seeds")
    plt.title("Phase 2A: Lost Sales vs Loss Aversion (λ)", fontsize=16)
    plt.ylabel("Lost Sales Ratio")
    plt.xlabel("Loss Aversion Intensity (λ) for Averse Agents")
    plt.legend()
    plt.savefig("docs/phase2a_figures/exp_phase2a_lost_sales.png", dpi=150)
    plt.close()

    print("Saved Phase 2A Replication plots to docs/phase2a_figures/")

if __name__ == "__main__":
    plot_phase2a()
