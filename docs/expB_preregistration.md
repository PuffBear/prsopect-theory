# Pre-registration — Experiment B: the w×λ collapse surface (frozen 2026-06-03)

**Committed while still blind to Experiment A's `w_crit`.** At the time of this
commit, the Exp A sweep has produced no analyzed result (no logistic fit has been
run; the raw CSV has not been inspected for any w→collapse relationship). This
document fixes how Experiment B will be read *before* the answer is known. It must
not be edited after Exp B data is inspected; changes require a dated v2 file with
justification.

The collapse label is the frozen definition in
[docs/collapse_definition.md](collapse_definition.md):
`collapsed := (mean_S < 10) AND (profit <= -128.1 + 1.0)`.

---

## 1. Question

Does loss aversion `λ` move the reward-centralization collapse boundary `w_crit`?
I.e., on the learnable base-stock substrate, is the critical centralization at
which the cooperative MARL system tips into collapse a function of the agents'
loss aversion?

## 2. Design (frozen)

| Knob | Levels |
|---|---|
| `w` (reward centralization = global_reward_weight) | {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} |
| `λ` (homogeneous loss aversion) | {1, 2, 3, 5, 7, 10} |
| seeds per (w, λ) cell | ≥ 15 |
| substrate | `scaled_base_stock_wrapper` [0, 500] |
| α, β PT curvature | 1.0 |
| training | 150k steps, PPO MlpPolicy, n_steps=128, batch_size=256 |
| eval | deterministic, ≥10 episodes, eval-seed varied per episode |

6 × 6 × 15 = 540 runs (tight-budget fallback: 4 w × 4 λ × 15 = 240, documented if
used). Outcome is **binary collapse**, analyzed by logistic regression — never OLS
on the floored continuous `mean_S`.

**Grid-coverage contingency (declared in advance).** The primary `w` grid above is
frozen. If Experiment A returns a `w_crit` that the grid brackets poorly (e.g.
`w_crit` outside [0.2, 0.8]), interior `w` levels *may be added* to improve
boundary coverage. That is an additive amendment only: the six pre-registered `w`
levels and the entire analysis below remain unchanged, and any added levels will be
labeled as a post-hoc grid extension.

## 3. Primary model (frozen)

```
P(collapsed) ~ logit(b0 + b1*w + b2*log(λ) + b3*(w * log(λ)))
```

- **`b3` (the w×log λ interaction) is the single pre-registered primary estimand.**
- Significance assessed two ways, both reported: (i) Wald 95% CI on `b3`;
  (ii) likelihood-ratio test of the full model vs. the reduced model without the
  interaction term.
- Quasi-separation handled by the same regularized-logistic fallback used in
  `analyze_expA.py` (BFGS MLE; L2 ridge `alpha=1e-3` fallback on singular Hessian).

## 4. The boundary estimand and headline figure (frozen)

The critical centralization as a function of λ:

```
w_crit(λ) = -(b0 + b2*log λ) / (b1 + b3*log λ)
```

- Plotted as `w_crit` vs `λ` with a **bootstrap 95% CI band** (resample whole runs
  with replacement, clustered by seed; refit; recompute `w_crit(λ)`; 2.5/97.5
  percentiles). Same bootstrap machinery as Exp A.
- **Headline figure (frozen as the paper's main figure):** a heatmap of the
  *empirical* per-cell `P(collapse)` over the (w × λ) grid, with the fitted
  `w_crit(λ)` curve and its CI band overlaid. This figure is the headline
  regardless of which way the result comes out (see §5).

## 5. Decision rule — symmetric and pre-committed

Define the boundary shift over the tested λ range:
`Δw_crit = w_crit(λ_max=10) − w_crit(λ_min=1)`.

**Outcome T — TILTED boundary → "loss aversion shifts the collapse boundary."**
Declared iff **both**: (a) `b3` Wald CI excludes 0 *and* the LRT p < 0.05, and
(b) `|Δw_crit| ≥ 0.1` (loss aversion moves critical centralization by at least 0.1
on the w scale, with the bootstrap CI on `Δw_crit` also excluding 0). This is the
behavioral-phase-transition result.

**Outcome V — VERTICAL boundary → "centralization dominates; λ is a red herring."**
Declared iff `b3` CI includes 0 **or** `|Δw_crit| < 0.1`. 

> **Binding commitment (the point of pre-registering now):** Outcome V is written
> up as the study's primary finding with the **same prominence** as Outcome T —
> same headline figure (§4), same billing in the title and abstract, same length of
> treatment. It is a clean negative result against the entire behavioral
> supply-chain framing (a behavioral bias does *not* move the structural collapse
> boundary), and it is reported as such, not relegated to an appendix, footnote, or
> "limitations" paragraph. Phrasing committed in advance: *"Reward centralization
> governs the collapse transition; loss aversion does not move it."*

Either outcome is a complete result. The only disallowed outcome is an
underpowered/ambiguous one, which §2 (≥15 seeds, binary logistic) and §6
(falsification controls) are designed to prevent.

## 6. Edge cases and controls (frozen)

- **Censored boundary.** If for some λ slice the per-cell collapse fraction is
  ~constant across w (no `w_crit` in [0,1]), report `w_crit` as censored (<0 or >1)
  for that slice, show it on the figure as an arrow/annotation, and exclude it from
  the `Δw_crit` computation while stating the censoring explicitly. A fully flat
  surface (collapse ≈ const everywhere) is reported as Outcome V by construction.
- **Floor/ceiling check.** Report the global collapse rate; if it is <5% or >95%
  across the whole grid, the surface is uninformative and that is stated plainly.
- **Seed-as-random-effect.** A mixed-effects logistic (w, log λ, interaction fixed;
  seed random) is fit as a secondary confirmation that `b3` survives seed
  clustering.
- **Initialization confound.** Defended by Experiment C (basin control); B does not
  re-litigate it.
- **Multiple comparisons.** `b3` is the sole primary test. Per-cell Wilson
  fractions, the mixed model, and continuous secondary outcomes (profit, lost
  sales) are explicitly secondary and labeled descriptive.

## 7. What gets reported regardless of outcome

1. Per-(w, λ) collapse-fraction table with Wilson 95% CIs.
2. Full logistic coefficient table (b0, b1, b2, b3) with CIs and the LRT.
3. The headline heatmap + `w_crit(λ)` boundary with CI band.
4. `Δw_crit` with bootstrap CI and the §5 outcome label.
5. The secondary mixed-effects fit.

The implementation of this plan is committed alongside this file as
`experiments/analyze_expB.py`; the function `decide_outcome()` therein encodes §5
exactly.
