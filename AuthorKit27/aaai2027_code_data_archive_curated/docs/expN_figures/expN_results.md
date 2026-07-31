# Exp N — null model (v(x)=x) vs PT(λ=1) @300k

Null model: 180 runs. Logistic slope(w)=-27.349, p=1.397e-06, McFadden R²=0.801.
**w_crit(null) = 0.685** [0.657, 0.710] (bootstrap, n=2000).

## Per-w collapse fraction (null, Wilson 95% CI)

| w | collapse | CI |
|---|---|---|
| 0.00 | 20/20 = 1.00 | [0.84, 1.00] |
| 0.20 | 20/20 = 1.00 | [0.84, 1.00] |
| 0.40 | 20/20 = 1.00 | [0.84, 1.00] |
| 0.50 | 20/20 = 1.00 | [0.84, 1.00] |
| 0.60 | 17/20 = 0.85 | [0.64, 0.95] |
| 0.70 | 10/20 = 0.50 | [0.30, 0.70] |
| 0.80 | 0/20 = 0.00 | [0.00, 0.16] |
| 0.90 | 0/20 = 0.00 | [0.00, 0.16] |
| 1.00 | 0/20 = 0.00 | [0.00, 0.16] |

## Comparison

- w_crit(null model) = 0.685 [0.657, 0.710]
- w_crit(PT λ=1 @300k) = 0.666 [0.635, 0.694] (slope=-22.046, p=3.207e-07, McFadden R²=0.641, n=140)
- **behavioral_excess = -0.020** (w_crit(PT) − w_crit(null))

Figure: [expN_vs_PT_comparison.png](expN_vs_PT_comparison.png)

Note: both conditions use λ=1, α=1, β=1 (v(x)=x). The PT(λ=1) dataset is the no-loss-aversion condition, so behavioral_excess near 0 is the expected null outcome and serves as a reproducibility/measurement check.
