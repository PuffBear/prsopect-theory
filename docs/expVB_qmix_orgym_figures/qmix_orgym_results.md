# QMIX on or-gym -- value-based w_crit decomposition

n = 800 runs. Bootstrap n=2000. Estimator identical to PPO/MAPPO (analyze_expN.fit_logistic).

| condition | w_crit | 95% CI | slope | p | n |
|---|---|---|---|---|---|
| curvature (l=1, a=0.88) | -2.160 | [-18.994, 17.271] | -0.45 | 6.84e-01 | 200 |
| loss aversion l=2 | -2.144 | [-13.607, 16.503] | -0.58 | 7.40e-01 | 100 |
| loss aversion l=3 | 1.486 | [-6.290, 8.316] | 1.81 | 2.68e-01 | 100 |
| loss aversion l=4 | 3.205 | [-12.511, 16.672] | 0.64 | 7.08e-01 | 100 |
| loss aversion l=5 | -1.674 | [-9.429, 18.343] | -0.79 | 6.77e-01 | 100 |
| loss aversion l=6 | 1.367 | [-2.759, 6.834] | 2.27 | 1.78e-01 | 100 |
| loss aversion l=7 | -0.715 | [-12.801, 13.592] | -1.10 | 5.18e-01 | 100 |

- **Loss-aversion excess** (w_crit[l7@0.88] - w_crit[l1@0.88]) = **+1.445**
- **Pre-registered verdict:** DIRECTIONAL (rises, but CIs overlap -> scope as exploratory)
