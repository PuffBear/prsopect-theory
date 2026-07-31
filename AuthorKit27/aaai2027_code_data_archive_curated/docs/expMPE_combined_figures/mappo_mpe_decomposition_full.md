# MPE/MAPPO behavioral decomposition (20-seed, full lambda sweep)

Null (alpha=beta=1) + curvature/loss-aversion sweep lambda=1..7 (alpha=beta=0.88). Sources: mappo_mpe_v2 + mappo_mpe_lambda_sweep.
Collapse threshold theta=0.144 (occupancy), recalibrated from this run's null w=0/w=1 endpoints.

| condition | n | w_crit | 95% CI | Firth slope | p | excess vs null | excess 95% CI |
|---|---|---|---|---|---|---|---|
| null | 220 | 0.735 | [0.703, 0.770] | -16.0 | 6.21e-10 | 0.000 | --- |
| lambda1 | 220 | 0.777 | [0.738, 0.817] | -12.9 | 1.28e-10 | +0.042 | [-0.009, +0.092] |
| lambda2 | 220 | 0.710 | [0.677, 0.745] | -15.3 | 2.24e-10 | -0.025 | [-0.074, +0.022] |
| lambda3 | 220 | 0.836 | [0.803, 0.871] | -17.1 | 1.70e-08 | +0.101 | [+0.055, +0.150] |
| lambda4 | 220 | 0.760 | [0.728, 0.794] | -17.0 | 2.14e-09 | +0.025 | [-0.023, +0.072] |
| lambda5 | 220 | 0.821 | [0.787, 0.856] | -16.5 | 7.39e-09 | +0.086 | [+0.038, +0.133] |
| lambda6 | 220 | 0.866 | [0.830, 0.911] | -12.9 | 9.83e-09 | +0.130 | [+0.078, +0.184] |
| lambda7 | 220 | 0.893 | [0.860, 0.936] | -16.2 | 1.75e-07 | +0.158 | [+0.108, +0.212] |
