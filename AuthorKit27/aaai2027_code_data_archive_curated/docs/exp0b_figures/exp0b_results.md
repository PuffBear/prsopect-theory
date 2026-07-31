# Exp 0b — result (horizon lock + metastability origin)

104 runs. Frozen interpretation rules from [exp0b_preregistration.md](../exp0b_preregistration.md) §4.

## Q1 — horizon: **SLIDING** (decisive)

Piece 1, 20 seeds, single `learn()` to 600k vs the 300k values:

| w | 300k | 600k (Wilson CI) | drop |
|---|---|---|---|
| 0.6 | 0.80 | 0.35 [0.18, 0.57] | **−0.45** |
| 0.7 | 0.40 | 0.20 [0.08, 0.42] | **−0.20** |

Both drops exceed the frozen 0.15 threshold → **Q1 = SLIDING**. The collapse boundary
is **not converged at 300k**; it keeps moving left with more training
(`w_crit`: 0.82@150k → 0.67@300k → ~0.55 implied @600k, still falling). Figure:
[exp0b_horizon_slide.png](exp0b_horizon_slide.png).

## Q2 — metastability origin: **INCONCLUSIVE**, and the design knob was wrong

Piece 2, per-seed flip rate between segmented and single arms (seeds 42–49):

| w | regime | flip rate | Wilson CI |
|---|---|---|---|
| 0.3 | deep-collapse | 0/8 = 0.00 | [0.00, 0.32] |
| 0.6 | transition | 0/8 = 0.00 | [0.00, 0.32] |
| 0.7 | transition | 0/8 = 0.00 | [0.00, 0.32] |
| 0.8 | deep-recovery | 0/8 = 0.00 | [0.00, 0.32] |

transition flip-rate = 0.00, deep flip-rate = 0.00, difference = 0.00 → neither
CRITICAL-SIGNATURE (needs diff ≥0.25) nor GENERIC-NONDET (needs deep >0.15) →
**Q2 = INCONCLUSIVE**.

**Why INCONCLUSIVE — a design flaw, reported honestly.** The segmented-vs-single
perturbation produces **zero** flips at *every* cell, including the transition. That
means: under fixed threading (OMP=1, this run), the model state at 300k is
deterministic given the seed, so segmented (150k+150k) and single (300k) `learn()`
converge to identical outcomes. The procedural knob I chose is **not actually a
perturbation** under controlled threading — so it cannot probe Q2's critical-vs-generic
question.

The original flip that motivated Q2 (Exp 0 segmented w=0.7@300k = 0.12 vs expA300k
single = 0.50, 3/8 disagreeing) was therefore **not** caused by the segmented/single
procedure. The two prior runs differed in threading/parallelism (Exp 0: procs=10,
default multithreading, load≈46; expA300k: different conditions). The most likely
trigger of the original flip is **low-level numerical nondeterminism (thread-count /
reduction-order)**, to which borderline-near-transition seeds are sensitive. Q2 would
need that knob (RNG/threading jitter at fixed seed), not segmented-vs-single, to be
answered.

## Combined verdict: Q1 = SLIDING, Q2 = INCONCLUSIVE

Per the frozen §4 table, **this is not a STABLE row, so Experiment B does not launch.**
The substance is the worst-case spirit: the **horizon confound is alive** (boundary
still sliding at 600k) and the metastability that might have rescued it as a genuine
critical point is **not demonstrated** (the test was mis-designed; the controlled
perturbation shows no sensitivity).

What is NOT yet established: whether the low-w cells (≤0.5) stay collapsed at long
horizon. Only w∈{0.6,0.7} were extended to 600k. So "there is no transition at all"
is not proven — only that the boundary location is horizon-dependent and unconverged
through 600k.

## Frozen-rule action + recommended path

§4 literal (INCONCLUSIVE): "+12 seeds at ambiguous cells, re-evaluate, no B." But the
0/8-at-every-cell result is a clean null on the *chosen* knob, not underpowering more
seeds would fix — and Q1=SLIDING dominates and gates B regardless. The honest path is
the SLIDING branch:

- **(a) Asymptotic-horizon definition:** establish collapse at a horizon where the
  boundary stops moving. Risk: even 600k isn't converged, so this may be very
  expensive, and "collapse" may keep shrinking toward "undertrained."
- **(b) Pivot to Exp D** (heterogeneous-λ minority): a *relative* comparison at matched
  horizon and matched w is robust to a sliding absolute boundary — adding averse
  agents either does or doesn't shift collapse *relative to all-rational at the same
  budget*. This sidesteps the horizon confound entirely and is the least-occupied
  ground.

Recommendation: **(b)**, unless a converged absolute `w_crit` is specifically needed.
B remains gated until either the horizon is locked (a) or the claim is reframed as
relative (b).
