# Experiment C — basin / initialization control (result)

144 runs: w ∈ {0.5, 0.85, 1.0} (spanning the measured transition w_crit≈0.82) ×
S_init ∈ {0, 50, 150, 250} × 12 seeds, offset wrapper, λ=1, 150k steps.

**Question:** at a fixed 150k budget, is the converged policy determined by *where it
was initialized* (S_init → basin/bistability) or by *reward centralization* (w)?

## Result: w-driven, NOT initialization-driven

Converged S (last-10k mean from the offset training log — correct offset mapping):

| w \ S_init | 0 | 50 | 150 | 250 |
|---|---|---|---|---|
| 0.50 | 1.2 | 1.1 | 0.8 | 0.8 |
| 0.85 | 18.5 | 16.3 | 13.9 | 14.1 |
| 1.00 | 39.5 | 26.0 | 22.4 | 22.6 |

Within each `w`, converged S does **not** track `S_init`:
`corr(S_init, converged_S)` = −0.11 (w=0.5), −0.38 (w=0.85), −0.36 (w=1.0) — weakly
*negative*, the opposite of the `converged ≈ init` (corr ≈ +1) signature a true
bistable basin would produce. The converged outcome is set by `w`, not by where the
policy started.

Corrected collapse fraction (converged_S<10 AND profit≤−127.1):

| w \ S_init | 0 | 50 | 150 | 250 |
|---|---|---|---|---|
| 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.85 | 0.00 | 0.00 | 0.00 | 0.08 |
| 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Initialization has at most a marginal effect (one cell, S_init=250 @ w=0.85, at 0.08).

## Interpretation, combined with the other Exp A diagnostics

The collapse transition is **not** explained away by any of the three standard
confounds:
- **not a critic failure** (critic diagnostic: collapsed runs fit the value function
  as well or better),
- **not a permanent basin / initialization artifact** (this experiment: outcome is
  w-driven, independent of S_init),
- it **is** a metastable, slow-convergence phenomenon gated by `w` (horizon check:
  transition-collapse resolves with more training).

Net: at a fixed training budget, reward centralization `w` controls whether the
cooperative system escapes the zero-order region. `w_crit` is a budget-conditioned
property of the learning dynamics, defended against the basin confound.

## Caveat / fix logged

The original `expC_raw_data.csv` `collapsed`/`eval_mean_S` columns were first
computed by reusing expA's `evaluate_and_extract_metrics`, whose `mean_S` uses the
**scaled [0,500]** mapping — wrong for the offset substrate. `profit` and
`converged_S` (offset log) were always correct. The collapse label was recomputed
from `converged_S + profit` (`collapsed_fixed` column added; runner patched so future
runs label from `converged_S`). All numbers above use the corrected label.
