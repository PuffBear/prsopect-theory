# MAPPO on LBF -- third-environment breadth result

n = 1439 runs. Collapse theta = 0.626 (collapsed-vs-coordinated midpoint, same calibration method as MPE). Bootstrap n=2000. Estimator identical to PPO/MAPPO.

| condition | w_crit | 95% CI | slope | p | n |
|---|---|---|---|---|---|
| null (l=1,a=1) | 0.077 | [-0.015, 0.135] | -7.58 | 7.36e-07 | 180 |
| curvature (l=1, a=0.88) | 0.256 | [0.188, 0.319] | -6.63 | 1.49e-11 | 180 |
| loss aversion l=2 | 0.529 | [0.456, 0.597] | -5.75 | 1.21e-13 | 179 |
| loss aversion l=3 | 0.732 | [0.660, 0.820] | -4.90 | 4.09e-10 | 180 |
| loss aversion l=4 (saturated\*) | 0.850 | [0.774, 0.978] | -4.72 | 5.57e-08 | 180 |
| loss aversion l=5 (saturated\*) | 0.857 | [0.768, 1.020] | -4.07 | 1.19e-07 | 180 |
| loss aversion l=6 (saturated\*) | 0.921 | [0.835, 1.084] | -5.02 | 9.51e-07 | 180 |
| loss aversion l=7 (saturated\*) | 1.117 | [0.910, 1.743] | -2.70 | 1.75e-04 | 180 |

- **Curvature shift** = **+0.179**
- **Loss-aversion excess** (l7 - l1) = **+0.861**
- **\*Saturated** at the w-grid ceiling (exploratory): lambda in [4, 5, 6, 7].
- **Pre-registered verdict:** PASS (w_crit rises, CIs non-overlapping)

**Threshold robustness** (loss-aversion direction is invariant to theta):

- theta=0.50: lambda1->7 = [0.01 0.22 0.47 0.41 0.57 0.66 0.77]  monotone=False
- theta=0.62: lambda1->7 = [0.22 0.5  0.71 0.84 0.85 0.91 1.02]  monotone=True
- theta=0.63: lambda1->7 = [0.26 0.53 0.73 0.85 0.86 0.92 1.12]  monotone=True
- theta=0.70: lambda1->7 = [0.32 0.63 0.78 0.88 0.9  0.97 1.28]  monotone=True
