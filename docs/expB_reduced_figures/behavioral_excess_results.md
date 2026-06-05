# Behavioral-excess decomposition (Exp B-Reduced vs Exp N null)

Null (α=β=1): w_crit = 0.685 [0.657, 0.710].

| λ | w_crit | w_crit CI | behavioral_excess | excess CI | significant |
|---|---|---|---|---|---|
| null | 0.685 | [0.657, 0.710] | 0 (baseline) | — | — |
| 1 | 0.586 | [0.539, 0.624] | -0.100 | [-0.172, -0.033] | no |
| 3 | 0.774 | [0.751, 0.798] | 0.089 | [0.041, 0.141] | yes |
| 7 | 0.851 | [0.845, 0.858] | 0.166 | [0.135, 0.202] | yes |

## Verdict
**BEHAVIORAL CLAIM CONFIRMED: loss aversion shifts isocline**

significant := excess_ci_lo > 0 (conservative difference CI).

Decomposition: excess(λ=1) ≈ curvature effect (α=β=0.88 vs 1); the increment with λ isolates loss aversion.

Figures: [fig2_behavioral_decomposition.png](fig2_behavioral_decomposition.png), [fig2b_behavioral_excess_bar.png](fig2b_behavioral_excess_bar.png)
