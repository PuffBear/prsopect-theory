import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_diagnosis():
    data_path = "docs/results/phase1_5_diagnosis.csv"
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    df = pd.read_csv(data_path)
    os.makedirs("docs/phase1_figures", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Experiment A
    df_a = df[df.experiment.str.startswith("A")].copy()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_a, x="timestep", y="order_qty", hue="experiment", palette="tab10", linewidth=2.5)
    plt.title("Exp A: Minimal Intervention Curve")
    plt.ylabel("Mean Order Quantity (PPO Agents)")
    plt.savefig("docs/phase1_figures/exp_A_minimal_intervention.png", dpi=150)
    plt.close()
    
    # 2. Experiment B
    df_b = df[df.experiment.isin(["A0_0_scripted", "A1_1_scripted", "B2_Distributors", "B3_Factories"])].copy()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_b, x="timestep", y="order_qty", hue="experiment", palette="Set2", linewidth=2.5)
    plt.title("Exp B: Location Sensitivity")
    plt.ylabel("Mean Order Quantity (PPO Agents)")
    plt.savefig("docs/phase1_figures/exp_B_location_sensitivity.png", dpi=150)
    plt.close()
    
    # 3. Experiment C
    df_c = df[df.experiment.isin(["C_Freeze", "C_Release"])].copy()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_c, x="timestep", y="order_qty", hue="experiment", palette="Set1", linewidth=2.5)
    plt.axvline(x=100000, color='red', linestyle='--', label='Unscript Node 1')
    plt.title("Exp C: Freeze-and-Release")
    plt.ylabel("Mean Order Quantity (PPO Agents)")
    plt.legend()
    plt.savefig("docs/phase1_figures/exp_C_freeze_and_release.png", dpi=150)
    plt.close()
    
    print("Saved Exp A, B, C plots to docs/phase1_figures/")

if __name__ == "__main__":
    plot_diagnosis()
