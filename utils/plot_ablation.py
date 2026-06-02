import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_ablation():
    data_path = "docs/results/ablation_scaling.csv"
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    df = pd.read_csv(data_path)
    
    # Make scale a string for categorical plotting
    df['scale'] = df['scale'].astype(str)
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="timestep", y="order_qty", hue="scale", style="phi", palette="Set1", linewidth=2.5)
    plt.title("Ablation: Mean Order Quantity (Scaling Effect)", fontsize=16)
    plt.ylabel("Mean Order Quantity")
    plt.xlabel("Timesteps")
    
    os.makedirs("docs/phase1_figures", exist_ok=True)
    plt.savefig("docs/phase1_figures/ablation_order_qty.png", dpi=150)
    plt.close()
    
    # Print the terminal diagnostic report for the variance
    final_df = df[df.timestep == df.timestep.max()]
    print("\n================== ABLATION DIAGNOSTICS ==================")
    print(final_df[['phi', 'scale', 'order_qty', 'true_reward', 'raw_reward_std', 'scaled_reward_std', 'cpt_reward_std']].to_string(index=False))
    print("==========================================================")

if __name__ == "__main__":
    plot_ablation()
