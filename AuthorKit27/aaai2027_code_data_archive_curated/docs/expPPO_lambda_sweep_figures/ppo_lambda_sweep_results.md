# PPO Full Lambda Sweep - w_crit by lambda (or-gym, alpha=beta=0.88)

700 runs total (combined). Bootstrap n=2000.

Monotonicity check: FAIL

| lambda | w_crit | CI_lo | CI_hi | slope | p | n |
|---|---|---|---|---|---|---|
| 1 | 0.586 | 0.539 | 0.624 | -15.621 | 1.13e-06 | 100 |
| 2 | 0.715 | 0.687 | 0.745 | -29.793 | 7.27e-06 | 100 |
| 3 | 0.774 | 0.751 | 0.798 | -32.775 | 1.27e-05 | 100 |
| 4 | 0.806 | 0.766 | 0.841 | -40.667 | 8.18e-06 | 100 |
| 5 | 0.839 | 0.788 | 0.845 | -151.893 | 9.34e-01 | 100 |
| 6 | 0.836 | 0.806 | 0.860 | -26.828 | 4.49e-06 | 100 |
| 7 | 0.851 | 0.845 | 0.858 | -158.915 | 9.71e-01 | 100 |

Figures: [ppo_full_lambda_collapse_curves.png](ppo_full_lambda_collapse_curves.png), [ppo_wcrit_vs_lambda_full.png](ppo_wcrit_vs_lambda_full.png)
