import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_phase1():
    data_path = "docs/results/phase1_double_sweep.csv"
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    df = pd.read_csv(data_path)
    os.makedirs("docs/phase1_figures", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Bullwhip Ratio
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="phi", y="bullwhip", hue="protocol", marker='o', markersize=8, linewidth=2.5, palette=["forestgreen", "royalblue"])
    plt.title("Phase 1: Supply Chain Bullwhip vs Loss Aversion (φ)", fontsize=16)
    plt.ylabel("Bullwhip Ratio (Var(Orders) / Var(Demand))")
    plt.xlabel("Fraction of Loss-Averse Agents (φ)")
    plt.savefig("docs/phase1_figures/exp_phase1_bullwhip.png", dpi=150)
    plt.close()
    
    # Plot 2: Economic Profit
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="phi", y="profit", hue="protocol", marker='o', markersize=8, linewidth=2.5, palette=["forestgreen", "royalblue"])
    plt.title("Phase 1: Economic Profit vs Loss Aversion (φ)", fontsize=16)
    plt.ylabel("True Economic Profit")
    plt.xlabel("Fraction of Loss-Averse Agents (φ)")
    plt.savefig("docs/phase1_figures/exp_phase1_profit.png", dpi=150)
    plt.close()

    # Plot 3: Mean Order Quantity
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="phi", y="order_qty", hue="protocol", marker='o', markersize=8, linewidth=2.5, palette=["forestgreen", "royalblue"])
    plt.title("Phase 1: Mean Order Quantity vs Loss Aversion (φ)", fontsize=16)
    plt.ylabel("Mean Order Quantity")
    plt.xlabel("Fraction of Loss-Averse Agents (φ)")
    plt.savefig("docs/phase1_figures/exp_phase1_order_qty.png", dpi=150)
    plt.close()

    print("Saved Phase 1 Double Sweep plots to docs/phase1_figures/")

if __name__ == "__main__":
    plot_phase1()
