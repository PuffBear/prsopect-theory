# Control A — reference-point pre-registration record

**This file is the pre-registration anchor for Control A.** The z0 reference points below
were measured by a pilot and FROZEN here **before** any Control A main-sweep analysis was
run. They are not to be changed afterward.

## Frozen reference points (z0)

| arm | z0 value | definition |
|---|---|---|
| `zero` | `0.0` | paper condition (kink at 0) |
| `mean` | `-2.9902` | empirical mean of the blended per-step reward |
| `median` | `-6.7650` | empirical median of the blended per-step reward |
| `neg_half_sigma` | `-5.1373` | `-0.5 * std`, std = 10.275 of the blended reward |

Stored in `refpoint_z0.json` (same directory), read by
`experiments/exp_control_a_refpoint.py`.

## Provenance (how z0 was measured)

- Pilot job: **PBS 128831** (sharc `cpu` queue, compute1), exit 0, walltime 00:07:47.
- Command: `exp_control_a_refpoint.py --pilot --freeze_z0 --pilot_timesteps 20000
  --pilot_ws 0.5,0.95 --pilot_seeds 42` (lambda=1).
- Measured on: 2026-07-22 23:59 (cluster local), pooled blended per-step reward,
  n=450 steps (150 at w=0.5, 300 at w=0.95).
- Reported scale: mean=-2.990, median=-6.765, std=10.275 → -0.5σ=-5.137.

## Integrity

z0 is a single fixed value **per arm** (not per-w, not per-lambda) — a reference point the
agent plausibly holds, applied uniformly across the whole w×lambda grid. It was measured
at lambda=1 only (the no-loss-aversion condition), so it does **not** peek at the
loss-aversion boundaries it is used to test. No z0 value was tuned to produce or remove a
boundary shift. Per the user's decision, this is recorded as a written pre-registration
anchor rather than a git commit.
