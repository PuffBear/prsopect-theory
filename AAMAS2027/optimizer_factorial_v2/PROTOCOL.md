# Optimizer and reward-normalization factorial (v2)

Status at freeze: **unexecuted**.

Frozen source-manifest SHA-256: `6f82cf0b812c5ca7315d509c6f18cec0039cb0fdd62405dcfcb435d4f4d75a0b`. Static compilation passed locally; the execution smoke test awaits the cluster environment because the local analysis environment does not contain PyTorch.

This experiment follows the completed B/A/S control with a prospective four-arm factorial. It tests whether the measured crossing shifts survive removal of actor--critic optimizer coupling and online reward normalization.

## Fixed factors

- Optimizer: standard joint global clipping at 0.5, or disjoint actor/critic Adam optimizers with no gradient clipping.
- Training reward: raw transformed reward, or `VecNormalize(norm_obs=False, norm_reward=True, gamma=.99, clip_reward=10, epsilon=1e-8)`.
- Arms: B curvature-only; A losses-only times 7; S both branches times 7; R both branches times 3.451, matching A's observed pooled transformed-reward RMS relative to B.
- Mixture grid: 0, .2, .4, .5, .6, .7, .85, 1.
- Seeds: 42--51.
- Reference: -2.6281398333333335, frozen from the independent 60,000-sample calibration used by the completed direct control.
- Budget: 300k requested agent transitions.

The complete factorial contains 1,280 trajectories. The 240 standard-optimizer/raw-reward B/A/S trajectories already exist and are reused only after their source, seeds, grid, reference, and evaluation protocol pass an equality gate; otherwise those cells are rerun. Thus the expected new-run count is 1,040.

## Diagnostics

Every minibatch records actor, critic, shared, and total gradient norms before and after the declared clipping operation, plus the actor share of squared gradient norm, policy loss, value loss, KL, PPO clip fraction, reward moments, raw and normalized advantage moments, and return-target moments.

## Primary comparisons

1. A-S within each optimizer/reward-normalization cell: sign-asymmetry contrast.
2. S-B and R-B: symmetric scale effects at fixed scale factors.
3. Interaction of A-S with optimizer mode and reward normalization.
4. Interaction of S-B with optimizer mode and reward normalization: test of the clipping/scale pathway.

All crossings use Firth logistic regression and 2,000 paired training-seed-block bootstrap draws with RNG seed 20261015. Fits and intervals retain finite out-of-grid draws. Practical equivalence is declared only when the entire A-S interval lies within [-.05, .05].

## Interpretation

- An A-S interval excluding zero after both interventions is evidence for a sign-asymmetric effect under this implementation.
- Collapse of A-B and S-B after uncoupling or normalization supports a reward-scale/optimizer mechanism.
- Persistence of A-B and S-B with A-S near zero supports a scale effect outside the tested clipping pathway.
- Any other pattern is reported as inconclusive; no arm or cell is added after outcomes are inspected.
