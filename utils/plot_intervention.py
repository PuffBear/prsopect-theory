import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_intervention():
    data_path = "docs/results/intervention_results.csv"
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    df = pd.read_csv(data_path)
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="timestep", y="order_qty", linewidth=3, color='forestgreen', label='PPO Nodes (w/ Scripted Retailer)')
    # Show the ablation baseline reference (which collapsed to ~0.0 - 0.25)
    plt.axhline(y=0.25, color='red', linestyle='--', label='Ablation Baseline (Unscripted)')
    plt.title("Intervention Experiment: Recovery of PPO Learning", fontsize=16)
    plt.ylabel("Mean Order Quantity (PPO Agents)")
    plt.xlabel("Timesteps")
    plt.legend()
    
    os.makedirs("docs/phase1_figures", exist_ok=True)
    plt.savefig("docs/phase1_figures/intervention_order_qty.png", dpi=150)
    plt.close()
    print("Saved plot to docs/phase1_figures/intervention_order_qty.png")

if __name__ == "__main__":
    plot_intervention()
