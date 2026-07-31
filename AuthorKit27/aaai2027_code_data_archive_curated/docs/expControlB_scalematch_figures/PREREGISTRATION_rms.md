# Control B — scale-match pre-registration record

**This file is the pre-registration anchor for Control B.** The per-lambda RMS output
scales below were measured by a pilot and FROZEN here **before** any Control B main-sweep
analysis was run. Not to be changed afterward.

## Frozen output_scale (stdmatched arm)

`output_scale(lambda) = rms(lambda) / rms(1)` — divides each lambda's transformed (CPT)
reward so every lambda has lambda=1's RMS magnitude, isolating the loss ASYMMETRY from the
reward AMPLITUDE.

| lambda | rms(CPT reward) | output_scale = rms(λ)/rms(1) |
|---|---|---|
| 1 | 7.931 | 1.000 |
| 3 | 17.060 | 2.151 |
| 7 | 31.873 | 4.019 |

Stored in `rms_output_scale.json` (same dir), read by
`experiments/exp_control_b_scalematch.py`. The `raw` arm uses output_scale=1 (baseline).

## Provenance

- Pilot job: **PBS 128854** (sharc `cpu` queue, compute8), exit 0.
- Command: `exp_control_b_scalematch.py --pilot --freeze_rms --pilot_timesteps 20000
  --pilot_ws 0.5,0.95 --pilot_seeds 42` (measures raw CPT-reward RMS per lambda in {1,3,7}).
- Measured on: 2026-07-23 (cluster local), pooled n=300 CPT-reward samples per lambda.

## Integrity

The RMS was measured on the RAW (output_scale=1) CPT reward, so the normalizer does not
peek at the boundary it is used to test. No value was tuned. Per the user's decision,
recorded as a written pre-registration anchor rather than a git commit (mirrors
[../expControlA_refpoint_figures/PREREGISTRATION_z0.md]).
