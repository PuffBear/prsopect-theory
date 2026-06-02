import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Plot Prospect Theory Value Function
x = np.linspace(-100, 100, 400)
alpha, beta = 0.88, 0.88
lambda_normal = 1.0
lambda_averse = 5.0

def pt_value(x, lam, a, b):
    return np.where(x >= 0, x**a, -lam * ((-x)**b))

y_normal = pt_value(x, lambda_normal, alpha, beta)
y_averse = pt_value(x, lambda_averse, alpha, beta)

plt.figure(figsize=(10, 6))
plt.plot(x, y_normal, label=f"Rational Agent ($\\lambda$={lambda_normal})", linewidth=2, color='blue')
plt.plot(x, y_averse, label=f"Loss Averse Agent ($\\lambda$={lambda_averse})", linewidth=2, color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(alpha=0.3)
plt.title("Prospect Theory Value Function: Reward Shaping")
plt.xlabel("Raw Environment Reward (Profit/Loss)")
plt.ylabel("Perceived Reward (Agent Utility)")
plt.legend()
plt.tight_layout()
os.makedirs("docs/assets", exist_ok=True)
plt.savefig("docs/assets/pt_curve.png", dpi=150)
print("Saved pt_curve.png")
