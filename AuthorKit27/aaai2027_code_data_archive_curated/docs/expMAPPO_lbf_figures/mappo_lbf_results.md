# MAPPO on LBF -- third-environment breadth result

n = 1599 runs. Collapse theta = 0.626 (collapsed-vs-coordinated midpoint, same calibration method as MPE). Bootstrap n=2000. Estimator identical to PPO/MAPPO.

| condition | w_crit | 95% CI | slope | p | n |
|---|---|---|---|---|---|
| null (l=1,a=1) | 0.078 | [-0.013, 0.134] | -7.65 | 4.12e-07 | 200 |
| curvature (l=1, a=0.88) | 0.255 | [0.188, 0.317] | -6.78 | 1.87e-12 | 200 |
| loss aversion l=2 | 0.517 | [0.449, 0.583] | -6.16 | 8.53e-16 | 199 |
| loss aversion l=3 | 0.695 | [0.631, 0.761] | -5.54 | 1.66e-12 | 200 |
| loss aversion l=4 | 0.832 | [0.767, 0.925] | -4.93 | 3.21e-09 | 200 |
| loss aversion l=5 (saturated\*) | 0.880 | [0.792, 1.031] | -3.89 | 5.16e-08 | 200 |
| loss aversion l=6 (saturated\*) | 0.915 | [0.845, 1.040] | -5.09 | 1.79e-07 | 200 |
| loss aversion l=7 (saturated\*) | 1.221 | [0.976, 2.013] | -2.39 | 3.79e-04 | 200 |

- **Curvature shift** = **+0.177**
- **Loss-aversion excess** (l7 - l1) = **+0.967**
- **\*Saturated** at the w-grid ceiling (exploratory): lambda in [5, 6, 7].
- **Pre-registered verdict:** PASS (w_crit rises, CIs non-overlapping)

**Threshold robustness** (loss-aversion direction is invariant to theta):

- theta=0.50: lambda1->7 = [0.01 0.22 0.44 0.39 0.53 0.61 0.82]  monotone=False
- theta=0.62: lambda1->7 = [0.22 0.49 0.68 0.82 0.87 0.9  1.13]  monotone=True
- theta=0.63: lambda1->7 = [0.25 0.52 0.69 0.83 0.88 0.92 1.22]  monotone=True
- theta=0.70: lambda1->7 = [0.32 0.62 0.75 0.87 0.94 0.97 1.37]  monotone=True

**Affine-law transfer** (supply-chain reference rho=0.61, RMSE=0.014): g/c=2.928 from lambda=1, best rho=1.000, RMSE=0.168 on 4 in-grid lambdas -> **does NOT transfer** (form is substrate-specific; the *method* generalizes, the affine *constants/form* do not).
