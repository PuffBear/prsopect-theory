# Exp T: Training-Horizon Collapse-Boundary Results

## Estimated w_crit by Horizon

| Horizon | w_crit | 95% CI lower | 95% CI upper |
|---------|--------|--------------|--------------|
| 150k | 1.9579 | 1.2675 | 4.0026 |
| 300k | 0.7088 | 0.5703 | 0.8329 |
| 600k | -0.6046 | -2.9054 | 0.1189 |

## Decay-Model Fit

| Model | R² |
|-------|----|
| Power law: $a T^{-b} + c$ **(best)** | 0.8279 |
| Log decay: $a / \log(T) + c$ | 0.0600 |

**Best model:** Power law: $a T^{-b} + c$
  Parameters: a=887887144266.9890, b=2.2507, c=0.0000

## Collapse Rates by (w, T)

| Horizon | w | P(collapse) | N |
|---------|---|-------------|---|
| 150k | 0.50 | 1.000 | 20 |
| 150k | 0.60 | 1.000 | 20 |
| 150k | 0.70 | 0.850 | 20 |
| 150k | 0.80 | 0.550 | 20 |
| 300k | 0.50 | 0.950 | 20 |
| 300k | 0.60 | 0.800 | 20 |
| 300k | 0.70 | 0.400 | 20 |
| 300k | 0.80 | 0.000 | 20 |
| 600k | 0.50 | 0.350 | 20 |
| 600k | 0.55 | 0.300 | 20 |
| 600k | 0.60 | 0.200 | 20 |
| 600k | 0.65 | 0.100 | 20 |
| 600k | 0.70 | 0.100 | 20 |
| 600k | 0.75 | 0.100 | 20 |
| 600k | 0.80 | 0.000 | 20 |

## Interpretation

w_crit decreases as training horizon increases, indicating that agents require *less* reward centralization to coordinate when given more training time. The collapse boundary shifts left with T, consistent with improved policy quality reducing reliance on shared incentives.

