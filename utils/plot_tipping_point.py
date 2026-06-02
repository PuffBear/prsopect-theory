import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results():
    data_path = "docs/results/sweep_data.csv"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
        
    df = pd.DataFrame(pd.read_csv(data_path))
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Bullwhip Ratio
    sns.lineplot(data=df, x="phi", y="bullwhip", marker="o", ax=axes[0], color="red", errorbar='sd')
    axes[0].set_title("Behavioral Tipping Point: Bullwhip Effect")
    axes[0].set_xlabel(r"Fraction of Loss Averse Agents ($\phi$)")
    axes[0].set_ylabel("Bullwhip Ratio")
    
    # Plot 2: Lost Sales
    sns.lineplot(data=df, x="phi", y="lost_sales", marker="o", ax=axes[1], color="orange", errorbar='sd')
    axes[1].set_title("Supply Chain Starvation")
    axes[1].set_xlabel(r"Fraction of Loss Averse Agents ($\phi$)")
    axes[1].set_ylabel("Systemic Lost Sales Ratio")
    
    # Plot 3: Inventory Variance
    sns.lineplot(data=df, x="phi", y="inventory_variance", marker="o", ax=axes[2], color="blue", errorbar='sd')
    axes[2].set_title("Macro-Economic Volatility")
    axes[2].set_xlabel(r"Fraction of Loss Averse Agents ($\phi$)")
    axes[2].set_ylabel("Mean Inventory Variance")
    
    plt.tight_layout()
    
    os.makedirs("docs/assets", exist_ok=True)
    plt.savefig("docs/assets/tipping_point.png", dpi=150)
    print("Saved tipping point plot to docs/assets/tipping_point.png")

if __name__ == "__main__":
    plot_results()
