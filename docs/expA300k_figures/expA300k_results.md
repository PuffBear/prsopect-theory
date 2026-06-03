# Converged w_crit re-estimation (Exp A protocol @ 300k)

140 runs: w ∈ {0.2,…,0.8} × 20 seeds, scaled [0,500] substrate, λ=1, single
`learn()` to 300k (the horizon Exp 0 identified as converged). Analyzed with the
frozen `analyze_expA.py` logistic pipeline.

## Converged transition

Per-cell collapse fraction (Wilson 95% CI):

| w | collapse | Wilson CI |
|---|---|---|
| 0.2 | 1.00 | [0.84, 1.00] |
| 0.3 | 1.00 | [0.84, 1.00] |
| 0.4 | 1.00 | [0.84, 1.00] |
| 0.5 | 0.95 | [0.76, 0.99] |
| 0.6 | 0.80 | [0.58, 0.92] |
| 0.7 | 0.40 | [0.22, 0.61] |
| 0.8 | 0.00 | [0.00, 0.16] |

**Logistic:** b1 = −22.05 (p = 3.2×10⁻⁷, CI [−30.5, −13.6]);
**`w_crit` = 0.666, bootstrap CI [0.635, 0.694]**; transition width Δw = 0.199;
quadratic adds nothing (p = 0.43 → monotone). Figure:
[expA300k_logistic.png](expA300k_logistic.png).

**Headline correction:** the converged critical centralization is **`w_crit` ≈ 0.67**,
not the 0.82 measured at 150k. The 150k value was inflated by undertraining; at the
converged horizon the transition shifts left by ~0.15 and is slightly *sharper*
(Δw 0.199 vs 0.278 @150k). The transition is real, sharp, and monotone — the central
result survives the horizon correction, with a corrected location.

## Reconciliation with Exp 0 — a reproducibility finding

Exp 0 (8 seeds, segmented 150k+150k training) reported w=0.7 @300k = **0.12**. This
run (20 seeds, single `learn()` to 300k) gives w=0.7 @300k = **0.40**. Checking the
shared seeds 42–49:

| | w=0.7 @300k collapse |
|---|---|
| this run, seeds 42–49 | 0.50 (4/8) |
| Exp 0, seeds 42–49 | 0.12 (1/8) |
| **disagree on** | **3/8 seeds (43, 47, 49)** |

Same seed, same w, same 300k target, **different collapse outcome**. The only
difference is the training path: Exp 0 restarts `learn()` at the 150k boundary
(discarding the partial rollout, re-collecting), this run trains 300k in one call.

**Interpretation:** near the transition the collapse outcome is **metastable and
path-sensitive even at fixed seed** — a minor training-procedure perturbation flips
borderline runs. This is a genuine property of the dynamics (consistent with the
metastable-convergence picture), not a bug. Consequences:

1. Exp 0's w=0.7 = 0.12 was **small-sample + path-luck optimism**; the robust
   20-seed value is **0.40**, and the converged `w_crit` is **~0.67, not "<0.7"** as
   Exp 0 implied.
2. The "300k = 600k plateau" from Exp 0 rests on those same 8 seeds. Given the
   metastability, **300k-convergence is not fully established for the full seed
   distribution** — the 8/20 w=0.7 runs that are collapsed at 300k here may or may
   not recover by 600k. This needs a 20-seed 300k→600k check at the transition cells
   before `w_crit ≈ 0.67` is locked as the converged value.

## Status of the number

`w_crit ≈ 0.67 [0.635, 0.694]` is the clean estimate **at 300k**. Whether 300k is
truly the plateau for the full seed distribution (vs. ~0.6x at 600k) is the one
remaining open item, reopened by the metastability finding. Recommended cheap check
before B-redesigned: w ∈ {0.6, 0.7} × 20 seeds → 600k, compare collapse fraction to
the 300k values here.
