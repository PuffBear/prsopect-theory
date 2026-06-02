# Pre-registered Collapse Definition (frozen 2026-06-03)

This file is committed **before** any new (w, λ) sweep results are generated, per
the Experiment Spec v1 §0 discipline. It must not be edited after new sweep data
is collected. Any change requires a new dated file (`collapse_definition_v2.md`)
with explicit justification.

## Definition

A single training run is labeled **collapsed** iff:

```
collapsed(run) := (mean_S < S_FLOOR_THRESHOLD) AND (profit <= PROFIT_FLOOR + EPSILON)
```

with the frozen constants:

| Constant | Value |
|---|---|
| `S_FLOOR_THRESHOLD` | `10.0` |
| `PROFIT_FLOOR` | `-128.1` |
| `EPSILON` | `1.0` |

`mean_S` = mean learned target base-stock at evaluation; `profit` = mean true
(un-shaped) economic profit at evaluation.

## Justification (from existing data, Day 0)

Pooled across all 95 existing runs (`phase4_raw_data.csv` w=1 centralized, n=70;
decentralized w=0 rows of `master_seed_dataset.csv`, n=25):

- `mean_S` is **bimodal**: a dominant spike of 40/95 runs in `[0,10)`, the count
  dropping to 5 in `[10,20)`, then a broad spread of recovered runs out to ~186.
  Figure: [docs/expA_figures/mean_S_bimodality.png](expA_figures/mean_S_bimodality.png).
  The `S_FLOOR_THRESHOLD = 10` cut sits in the trough at the edge of the collapse
  spike.
- The collapse count is **insensitive** to the exact threshold: 32 collapsed at
  S_floor ∈ {5, 10, 15}, 35 at {20, 30}. The label is not produced by threshold
  tuning.
- The **AND** is required: 8 runs have `mean_S < 10` but positive profit (they are
  *not* collapsed). S-only labels 40, profit-only labels 36, the AND labels 32.

## Why this is the dependent variable

`Mean_S` regressed continuously (the prior `Mean_S ~ log(λ)`, slope −24.99,
R²=0.115) is an OLS line through this two-cluster mixture; the slope tracks the
*fraction of collapsed seeds per bin*, not a graded behavioral response. All
Experiment A–D analyses therefore model `collapsed` as a **binary** outcome via
logistic regression, never OLS on the floored continuous `Mean_S`.

## Existing endpoint evidence (motivates Experiment A)

Collapse fraction at the two reward-centralization endpoints already present in the
data:

| `w` (= global_reward_weight / "Alpha") | collapse fraction |
|---|---|
| 0.0 (decentralized) | 25/25 = **100%** |
| 1.0 (centralized) | 7/70 = **10%** |

Experiment A maps the unmeasured interior `w ∈ {0.1 … 0.9}` to locate `w_crit`.

## Pre-registered success / falsification criteria

- **Experiment A success:** monotone logistic fit `P(collapsed) ~ logit(b0 + b1·w)`
  with `b1` CI excluding 0, transition width `Δw < 0.4` (between P=0.1 and P=0.9),
  per-cell collapse fractions ≈1 at low `w` and ≈0 at high `w`.
- **Experiment A falsified** if collapse fraction is flat across `w`.
- **Experiment B:** the interaction term `b3` in
  `P(collapsed) ~ logit(b0 + b1·w + b2·log λ + b3·(w·log λ))` tests whether λ
  shifts `w_crit`. A tilted `w_crit(λ)` boundary = behavioral phase transition; a
  vertical boundary = "centralization dominates, λ is a red herring" (reported as a
  clean negative result, not buried).
