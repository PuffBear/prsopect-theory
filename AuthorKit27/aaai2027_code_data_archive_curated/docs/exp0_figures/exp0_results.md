# Experiment 0 — convergence / horizon study (RESULT)

24 runs: w ∈ {0.7, 0.8, 0.9} × 8 seeds, scaled [0,500] substrate, λ=1, trained to
600k with collapse logged at 150k / 300k / 600k via segmented `learn()`.

**Question (blocking):** is the collapse transition a property of the equilibria
(`w_crit` stable with more training) or a training-budget artifact (`w_crit` slides
left without bound)?

## Result: real transition, but `w_crit = 0.82` was an undertraining artifact

Collapse fraction by (w, checkpoint):

| w | 150k | 300k | 600k |
|---|---|---|---|
| 0.7 | 0.75 | 0.12 | 0.12 |
| 0.8 | 0.75 | 0.00 | 0.00 |
| 0.9 | 0.38 | 0.00 | 0.00 |

Figure: [exp0_convergence.png](exp0_convergence.png).

**Two findings, both decisive:**

1. **The transition is REAL, not an unbounded horizon artifact.** Collapse fraction
   drops sharply 150k→300k, then is **identical at 300k and 600k** for all three
   w-levels. It converges to a stable structure (w=0.7 → 12% residual, w≥0.8 → 0%)
   rather than melting to zero with ever-more training. This is the "stabilizes"
   world: there is a genuine equilibrium phase transition.

2. **`w_crit = 0.82` (Exp A, 150k) was severely inflated by undertraining.** At
   convergence (≥300k), w=0.8 and w=0.9 fully recover (0% collapse) and even w=0.7
   holds at only 12%. The converged transition sits **below w=0.7**, far left of the
   150k estimate. At 150k these runs were mid-escape, not collapsed (e.g. w=0.8
   mean_S 0 → 85 → 161 across the three checkpoints).

## Consequences

- **Exp A's `w_crit = 0.822` must be reported as a 150k-budget readout, not the
  system's critical centralization.** The converged `w_crit` is < 0.7 and needs its
  own estimation (an Exp A rerun at ≥300k on a low-centred w-grid).
- **Stopping Exp B was correct** — run at 150k it would have measured a horizon-
  contaminated boundary.
- **The program is in the favorable branch:** a real transition exists, so the
  λ-interaction question (does loss aversion move the *converged* `w_crit`?) is worth
  pursuing — but only at the converged horizon.

## Next (pre-committed branch, now selected)

1. **Re-estimate the converged transition:** Exp A protocol at **300k** (converged;
   300k=600k so 300k suffices and halves cost vs 600k) on a low-centred w-grid,
   e.g. w ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, 20 seeds → clean converged `w_crit`.
2. **Then B-redesigned at 300k** on a w-grid straddling the *converged* `w_crit`,
   × λ — the unconfounded interaction claim.

300k (not 600k) is the working horizon: the plateau shows convergence is reached by
300k, so 600k buys nothing and doubles compute.
