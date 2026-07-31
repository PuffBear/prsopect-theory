# REANALYSIS — Part 2 (no new training)

Robustness of the loss-aversion result: does the **λ=1→7 rightward shift in `w_crit`**
survive nonparametric estimation, seed-paired uncertainty, and the raw (un-fitted) data?

**Verdict: yes, in all three environments.** The shift is present under logistic, Firth,
AND nonparametric isotonic regression, and the seed-paired bootstrap difference CI
**excludes zero** in every substrate. Full tables, per-w raw counts, and figures:
[docs/reanalysis_controls/REANALYSIS_controls.md](docs/reanalysis_controls/REANALYSIS_controls.md).
Script: [experiments/reanalysis_controls.py](experiments/reanalysis_controls.py).

## Headline

| Substrate | `w_crit` λ-lo → λ-hi (Firth) | Δ (isotonic) | Seed-paired bootstrap Δ CI |
|---|---|---|---|
| Supply chain (λ1→7) | 0.585 → 0.854 (**+0.269**) | +0.284 | **[0.232, 0.306]** ✓ excludes 0 |
| MPE (λ2→7) | 0.747 → 0.955 (**+0.208**) | similar | **[0.156, 0.275]** ✓ excludes 0 |
| LBF (λ1→7) | 0.162 → 1.229 (**+1.03**) | in-grid through λ6 | **[0.789, 1.664]** ✓ excludes 0 |

All three estimators agree on the **direction and approximate magnitude** — the shift is
not an artifact of the logistic functional form. (LBF's λ=7 boundary is flagged as
*extrapolated* beyond the tested w-grid; its magnitude there is a lower bound, direction
unaffected. MPE/LBF collapse labels were recomputed with the paper's recalibrated θ, since
the stored CSV `collapsed` columns sit under an un-recalibrated threshold — see below.)

## Part-2 asks → what was delivered

1. **Isotonic alongside logistic; λ shift similar under both** — done, all 3 substrates.
   `sklearn.isotonic` (monotone-decreasing in w), overlaid with logistic + Firth in
   `*_estimator_overlay.png`; per-λ `w_crit` table compares all three.
2. **Seed-blocked / paired bootstrap** — done. Resampling unit = **seed**; one seed
   multiset is drawn per replicate and applied to *both* λ, so the λ-difference is paired.
   2000 replicates; difference CI from replicate-wise differences.
3. **Raw collapse counts k/n at every w** — done; printed as fraction *and* raw k/n
   pivots per substrate (not only fitted sigmoids).
4. **Continuous metrics around the boundary** — done: supply base-stock **and** profit vs
   w (+ a base-stock-vs-profit scatter showing the two-variable label), MPE occupancy,
   LBF food/mean_reward — median, IQR band, and raw scatter so smoothness vs bimodality
   is visible (`*_boundary_metrics.png`, `supply_basestock_vs_profit.png`).
5. **Explicit method reporting** — in the report header:
   - Bootstrap replicates: **2000**.
   - Resampling unit: **seed** (seed-blocked / paired across λ).
   - Separation handling: **Firth** penalized logistic for separated/steep cells; logistic
     MLE L2-fallback on non-finite SE; isotonic is separation-free; degenerate bootstrap
     resamples skipped, usable-replicate count reported.
   - Difference CI: recomputed **within each paired replicate**, 2.5/97.5 percentiles.

## Language note (Part-2 request)

Where the underlying continuous metric is smooth rather than bimodal across the boundary
(see the boundary-metric plots), describe the object as a **threshold-defined behavioral
crossing**, not a physical phase transition. The reanalysis does not assume bimodality;
`w_crit` is defined as the P(collapse)=0.5 crossing of a monotone fit.

## Faithfulness caveat (MPE / LBF collapse labels)

The stored `collapsed` column in the MPE and LBF raw CSVs is under an **un-recalibrated**
threshold; with it, MPE's boundary sits at w≥1 and LBF is near-all-zero (degenerate). The
reanalysis therefore recomputes the collapse label with the **same recalibrated θ the
paper's own analyzers use**:
- MPE: `θ` from `env.mpe_coop_nav.load_occupancy_threshold()` (fallback 0.164), collapse =
  `mean_occupancy < θ` — mirrors `analyze_mpe`.
- LBF: `θ` = midpoint of the collapsed regime (highest-λ, lowest-w mean_reward) and the
  coordinated regime (null / highest-w) — mirrors `analyze_mappo_lbf.recalibrate_theta`.
The recomputed θ is reported per substrate in the detail file. Supply chain uses its
stored, already-frozen label unchanged.

## Not yet done (Part 1 controls — need training)

The three decisive controls (A reference-point, B scale-matched, C reward-unit) require
new runs and are tracked separately. Highest priority: **Control A** (reference point
`z0`), testing `z0 ∈ {0, empirical mean, ±σ-offset, substrate target} × λ∈{1,3,7}`.
