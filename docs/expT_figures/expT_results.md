# Exp T: Training-Horizon Collapse-Boundary Results

> **Scope caveat.** This is a THREE-POINT fit (T = 150k, 300k, 600k). With
> a 2-parameter model that leaves one residual degree of freedom, so R^2
> cannot discriminate functional forms and no model selection is claimed.
> The per-horizon w_crit values are themselves logistic *extrapolations*:
> at 150k the collapse curve has not crossed 0.5 within the sampled w-grid
> (w_crit > 1) and at 600k it has nearly cleared it (point estimate < 0),
> so both endpoints lie outside the meaningful [0,1] range and carry wide
> CIs. We therefore report this as a SIGN-AND-DIRECTION result (w_crit
> decreases with T) plus the theory-driven Kramers form for reference,
> NOT a calibrated asymptotic bound.

## Estimated w_crit by Horizon

| Horizon | w_crit | 95% CI lower | 95% CI upper |
|---------|--------|--------------|--------------|
| 150k | 1.9579 | 1.2675 | 4.0026 |
| 300k | 0.7088 | 0.5703 | 0.8329 |
| 600k | -0.6046 | -2.9054 | 0.1189 |

## Candidate decay forms (BOTH reported; no verdict)

The Kramers / Arrhenius barrier law is the theory-driven prediction
(mean escape time ~ exp(barrier), so the marginal w to hold P(escape | T)
fixed falls linearly in ln T: w_crit = c - a ln T). The power law is an
unconstrained empirical comparison. R^2 is shown for completeness only --
with 3 points it does NOT select between them.

| Form | R^2 | parameters |
|------|-----|------------|
| Kramers (theory): $c - a\,\ln T$ | 0.9994 | a=1.844, c=23.96 |
| Power law (empirical): $a T^{-b} + c$ | 0.6185 | a=1.45e+05, b=0.969, c=1.963e-06 |

**Kramers slope** a = 1.844 (w_crit per unit ln T). a > 0 means the
coordination boundary recedes as training lengthens, the predicted
direction. Magnitude is indicative only given the caveats above.

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

w_crit decreases as training horizon increases: agents require *less* reward centralization to coordinate when given more training time, the direction the Kramers escape picture predicts. We claim only the sign and direction of this dependence -- the three-point fit does not support a calibrated asymptotic bound or a selection between the log and power-law forms.

