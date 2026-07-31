# SNR probe-state sweep: pre-registered, non-tunable negative (180 runs, 150k)

**Verdict: the probe coordinate mu(s*) RECOVERS the qualitative two-basin shape
the grand mean destroyed (genuine advance), but the mean-field Langevin mechanism
is NOT predictive even on this coordinate -- the basin ENDPOINT ORDERING the
model requires is absent in the data, so the theory-derived (pre-registered)
drift band cannot even be formed. This is a clean, non-tunable negative.**

## The advance over the first (grand-mean) negative
The grand-mean mu washed out all structure (survivors and collapsers both ~-0.2
to -0.3, no w-trend; see snr_mechanism_diagnosis.md). The probe coordinate
mu(s*) -- the deterministic action at a FIXED inventory-position percentile --
is a genuine, w-ordered coordinate: its mean climbs monotonically with w
(approx -0.55 @ w=0.5 -> ~0 @ w=0.9), recovering a low-w "lazy" corner near the
mu=-1.0 floor and a high-w "cooperative" region. So we DID identify a coordinate
in which the basins are visible -- a real theoretical result (the mean-field
variable is the state-conditional action, not the visited-state-averaged action).

## Why it is still not predictive (and why we do NOT claim it is)
1. Band sensitivity = researcher DOF. Varying the contested-band parameter
   mu_coop over [-0.10, +0.10] flips the "predictive" verdict on/off and changes
   WHICH probe is flagged; the crossing point estimate sits at ~0.64-0.65 (below
   w_crit=0.822) and only touches 0.822 when the bootstrap CI balloons wide
   enough to swallow it. Tuning mu_coop to land on 0.822 would be p-hacking.
   REJECTED as a method.
2. Pre-registered band (the disciplined fix) is UNDEFINED. Rule fixed before
   looking at any crossing: mu_coop = cooperative fixed point = mean end-mu(s*)
   of SURVIVORS at the highest w; lazy = collapsed fixed point = mean end-mu(s*)
   of COLLAPSERS at the lowest w (basin endpoints, independent of the boundary
   location). The data gives coop < lazy on ALL THREE probes:

   | probe | coop (surv @ w=0.9) | lazy (coll @ w=0.5) | coop > lazy? |
   |-------|---------------------|---------------------|--------------|
   | lo    | -0.3245             | -0.1970             | False        |
   | med   | -0.3258             | -0.1976             | False        |
   | hi    | -0.3078             | -0.2017             | False        |

   The two-basin ordering the Langevin model assumes (coordinated sits ABOVE
   collapsed) is inverted, so the contested band (lazy, mu_coop) is empty. The
   mechanism is falsified at the coordinate level by a pre-registered test --
   not by a tuned one.

## What is solid (keep for the paper)
- Boundary REPRODUCED a third time: collapse-fraction=0.5 crossing = 0.822.
- The predictive coordinate is IDENTIFIED (state-conditional action mu(s*)) and
  shown to carry w-ordered structure the grand mean lacked; within a fixed w,
  survivors sit above collapsers in the populated mid-grid (w=0.7-0.8).

## Bottom line for framing
Demote cleanly: the mean-field theory is CONSISTENT and now COORDINATE-IDENTIFIED
(we know what the Langevin variable is and show it has the right qualitative
form), but it is NOT a quantitatively predictive locator of w_crit -- a
pre-registered endpoint test falsifies the two-basin structure required for the
drift crossing. Do NOT tune the band to manufacture a crossing. The honest pair
"boundary reproduced + coordinate identified + pre-registered crossing fails" is
a stronger, review-proof story than a tuned positive.
