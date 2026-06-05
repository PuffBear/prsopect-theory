# Exp B-Reduced — w_crit by loss aversion λ (α=β=0.88)

300 runs total. Bootstrap n=2000 (run-level).

| λ | w_crit | CI_lo | CI_hi | b1 (slope) | p | McFadden R² | n |
|---|---|---|---|---|---|---|---|
| 1 | 0.586 | 0.539 | 0.624 | -15.621 | 1.13e-06 | 0.461 | 100 |
| 3 | 0.774 | 0.751 | 0.798 | -32.775 | 1.27e-05 | 0.709 | 100 |
| 7 | 0.851 | 0.845 | 0.858 | -158.915 | 9.71e-01 | 0.771 | 100 |

## Behavioral test

**BEHAVIORAL CLAIM CONFIRMED**
- w_crit(λ=7) CI_lo=0.845 > w_crit(λ=1) CI_hi=0.624

Figures: [expB_wcrit_by_lambda.png](expB_wcrit_by_lambda.png), [expB_wcrit_vs_lambda.png](expB_wcrit_vs_lambda.png)

(b1 is the statsmodels w-coefficient; the spec's P=1/(1+exp(b0+b1·w)) uses b1=−slope. w_crit=−b0/slope is identical either way.)
