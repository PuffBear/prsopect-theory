import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_phase2():
    data_path = "docs/results/phase2_divergence.csv"
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    df = pd.read_csv(data_path)
    os.makedirs("docs/phase2_figures", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Mean Order Quantity
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="lambda", y="order_qty", marker='o', markersize=8, linewidth=2.5, color="darkorange")
    plt.title("Phase 2: Mean Order Quantity vs Loss Aversion (λ)", fontsize=16)
    plt.ylabel("Mean Order Quantity")
    plt.xlabel("Loss Aversion Intensity (λ) for Averse Agents")
    plt.savefig("docs/phase2_figures/exp_phase2_order_qty.png", dpi=150)
    plt.close()
    
    # Plot 2: Inventory Hoarding
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="lambda", y="inventory", marker='o', markersize=8, linewidth=2.5, color="crimson")
    plt.title("Phase 2: Network Inventory vs Loss Aversion (λ)", fontsize=16)
    plt.ylabel("Mean Network Inventory Level")
    plt.xlabel("Loss Aversion Intensity (λ) for Averse Agents")
    plt.savefig("docs/phase2_figures/exp_phase2_inventory.png", dpi=150)
    plt.close()

    # Plot 3: Economic Profit
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="lambda", y="profit", marker='o', markersize=8, linewidth=2.5, color="navy")
    plt.title("Phase 2: Economic Profit vs Loss Aversion (λ)", fontsize=16)
    plt.ylabel("True Economic Profit")
    plt.xlabel("Loss Aversion Intensity (λ) for Averse Agents")
    plt.savefig("docs/phase2_figures/exp_phase2_profit.png", dpi=150)
    plt.close()
    
    # Plot 4: Lost Sales
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="lambda", y="lost_sales", marker='o', markersize=8, linewidth=2.5, color="purple")
    plt.title("Phase 2: Lost Sales vs Loss Aversion (λ)", fontsize=16)
    plt.ylabel("Lost Sales Ratio")
    plt.xlabel("Loss Aversion Intensity (λ) for Averse Agents")
    plt.savefig("docs/phase2_figures/exp_phase2_lost_sales.png", dpi=150)
    plt.close()

    print("Saved Phase 2 Divergence plots to docs/phase2_figures/")

if __name__ == "__main__":
    plot_phase2()
