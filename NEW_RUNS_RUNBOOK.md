# Runbook — Three New Experiments (Reviewer Points 2, 3, 4)

These three runs close the gaps flagged in the AAAI revision. All reuse the
existing substrate, optimizer (PPO), CPT machinery, collapse label, grid, and
20-seed convention, so outputs feed the existing logistic/bootstrap/Firth
estimators unchanged. HPC: `ssh <user>@10.1.4.95`, PBS/Torque, `conda activate marl`.

| Reviewer pt | Driver | PBS | Array | Runs | New output CSV |
|---|---|---|---|---|---|
| 2 — extra horizons | `experiments/exp_horizon_extra.py` | `run_horizon_extra.pbs` | 0–319 | 320 | `docs/expHorizon_extra_figures/horizon_extra_raw_data.csv` |
| 4 — finer high-λ grid | `experiments/exp_highlambda_finegrid.py` | `run_highlambda_finegrid.pbs` | 0–119 | 120 | `docs/expHighLambda_finegrid_figures/highlambda_finegrid_raw_data.csv` |
| 3 — cascade (position + continuous λ) | `experiments/exp_cascade_continuous.py` | `run_cascade_continuous.pbs` | 0–1099 | 1100 | `docs/expCascade_continuous_figures/cascade_continuous_raw_data.csv` |

## What each one shows

**Point 2 — extra horizons (450k, 900k @ α=β=1).** Two new matched horizons so
the metastable slide and the −ln T / Kramers trend rest on four points, not two.
Interior-w grid (0.40–0.80) where w_crit sits at longer horizons. Merges with the
existing 150k/300k boundaries for the fit.

**Point 4 — finer high-λ grid (w∈{0.80, 0.90} at λ∈{5,6,7}).** Two interpolating
interior points across the steep high-λ transition (currently resolved by the
single w=0.85 cell). Concatenate with `ppo_lambda_sweep_raw_data.csv`
(identical schema) and re-run the Firth fit to tighten the λ≥5 CIs.

**Point 3 — cascade thickening.** Arm A: a single λ=7 node placed at every chain
position node_2…node_6 (downstream→upstream) — traces w_crit vs. position. Arm B:
the fixed downstream node (node_2) swept over λ∈{1,2,3,4,5,7} — traces w_crit vs.
bias strength at the critical position. Turns the binary-{1,7}, four-config result
into a position curve and a dose-response curve on the same topology.

## Run order (copy/paste on the login node)

```bash
# 0) sync code up
rsync -avz experiments *.pbs <you>@10.1.4.95:~/prospect-theory/

# 1) WIRING SANITY (no real training) — confirms env builds + index mapping
python3 experiments/exp_horizon_extra.py --run_idx 0 --smoke \
    --out /tmp/smoke_horizon.csv
python3 experiments/exp_highlambda_finegrid.py --run_idx 0 --timesteps 20000 \
    --out /tmp/smoke_highlam.csv
python3 experiments/exp_cascade_continuous.py --run_idx 0 --timesteps 20000 \
    --out /tmp/smoke_cascade.csv
# each should print one run line and write one CSV row.

# 2) submit the arrays
qsub run_horizon_extra.pbs
qsub run_highlambda_finegrid.pbs
qsub run_cascade_continuous.pbs

# 3) watch
qstat -tu $USER

# 4) when finished, pull
rsync -avz <you>@10.1.4.95:~/prospect-theory/docs/ ./docs/
```

## Analysis (after data lands)

- **Horizons:** estimate w_crit per horizon (existing bootstrap logistic), then
  fit w_crit vs. ln T across {150k, 300k, 450k, 900k}; report the slope and the
  extrapolated T→∞ asymptote with a CI. Drops into Results §"sharp, metastable
  boundary" (replaces the queued-footnote).
- **High-λ grid:** `cat` the new rows onto `ppo_lambda_sweep_raw_data.csv`, re-run
  `analyze_firth_lambda_sweep.py`; report the tightened λ∈{5,6,7} CIs in
  Table (full sweep) in the Supplement and the body sweep sentence.
- **Cascade:** w_crit per cell. Arm A → a w_crit-vs-position curve (expect a
  monotone downstream→upstream falloff, sharpening the "cascades upstream not
  downstream" claim). Arm B → a w_crit-vs-λ dose-response at the downstream node
  (expect monotone rise toward the homogeneous-LA 0.851). Both promote the
  minority section from binary to continuous and replace its queued-footnote.

## Compute note
Walltimes: horizon array is 12h/run-slot (the 900k cells are ~3× the 300k
baseline); the other two are 6h, matching the existing het/sweep jobs. Each array
element is one (cell, w, seed) run, one CPU, append-with-filelock to a shared CSV —
identical to `exp_heterogeneous.py`, so failure/restart semantics are the same.
