# Internal note: what governs ridge shape across cells

**Status:** hypothesis-generating, NOT a result. Do not put in the AAAI-27 submission.
Grounded in existing raw data only (no new runs). n = 4 cells — too few for a
statistical claim; this is a *direction* for the next paper / the Yamaguchi pitch.
Date: 2026-07-04.

---

## The question that started this

The 2×2 phase-diagram visual makes it obvious that the four `w_crit(λ)` ridges
have *different shapes*. Natural question: does one environmental property predict
ridge shape (and therefore predict when the affine law
`w_crit = φ/(φ+g/c)`, `φ = 1+ρ(λ−1)` holds)?

## What the data actually says (re-fit from raw runs, not the paper's rounded series)

Affine-law fit quality (RMSE), per cell:

| Cell | affine RMSE | verdict |
|---|---|---|
| Supply / PPO | ~0.013 | fits (paper headline) |
| MPE / MAPPO | ~0.010 | fits well (currently under-reported as "no") |
| Supply / MAPPO | ~0.026 | mediocre |
| LBF / MAPPO | ~0.155 | fails badly |

So it is **not** "1 substrate fits, 3 fail." It is **2 fit, 2 degrade** — and the
two that degrade do so for **two different reasons**. That two-reason structure is
the actual finding.

## Reason 1 — LBF: exploration noise is the binding term

- LBF is sparse-reward, hard-exploration.
- Its boundary runs *off the top*: `w_crit → 1.22` at λ=7 (past the w=1 ceiling).
- Fitted `ρ` pins at the grid ceiling (1.5), `g/c ≈ 1.5` — both extreme.
- This lines up with the **entropy dissociation** already in the paper: injected
  exploration noise *moves* the LBF boundary (binding) but *not* MPE's (non-binding).
- **Reading:** the affine law holds when exploration noise is NOT the binding term
  in the SNR; it breaks when sparse-reward exploration dominates. This is the same
  environment axis the dissociation already measures — a *second observable*
  (ridge shape) pointing at the same property. That is what makes it more than
  curve-fitting.

## Reason 2 — Supply/MAPPO: shallow, non-saturating boundary at low λ

This one is NOT a sparsity story — it is a credit-assignment / boundary-sharpness
story, and it is visible directly in the collapse fractions (raw, 700 runs):

Collapse fraction across w, by λ (Supply/MAPPO):

| λ | collapse span (max−min) | floor (min) | fully clears? |
|---|---|---|---|
| 1 | 0.70 | **0.30** | no |
| 2 | 0.90 | 0.10 | no |
| 3 | 1.00 | 0.00 | yes |
| 7 | 0.90 | 0.10 | ~ |

Compare Supply/**PPO**, which saturates fully (floor = 0.00) at every λ.

- At λ=1 under MAPPO, even at w=0.95, ~30% of runs still collapse — the transition
  never completes, so `w_crit` is genuinely ill-defined / smeared, not just noisy.
- The apparent non-monotonic "dip" at λ=2 (0.795 → 0.747 → 0.795) is largely an
  artifact of fitting a crisp midpoint to a mushy transition — but the *mushiness
  itself is real* and is the point.
- **Reading:** the centralized critic smears the boundary at low loss aversion.
  Ridge shape here is set by boundary *sharpness under the critic*, a distinct
  factor from LBF's exploration-noise binding.

## The honest takeaway

Ridge shape is governed by **at least two distinct factors**, not one:

1. whether **exploration noise binds** (sparse reward → affine law breaks; LBF),
2. **boundary sharpness under the learner** (centralized critic smears the
   low-λ boundary → w_crit ill-defined; Supply/MAPPO).

There is **no single unifying law** across the four cells, and forcing one would be
fishing. That is scientifically the correct conclusion, and it is consistent with —
and sharper than — the paper's current "the direction generalizes; the affine law
is substrate-specific."

## Why this is a good next-paper / collaboration thread

- It converts the vague open problem ("what predicts ridge shape?") into a
  **specific, falsifiable, two-factor hypothesis** tied to observables you already
  measure (entropy dissociation; collapse-fraction saturation).
- It is directly in Yamaguchi's wheelhouse — their paper is "environment geometry
  shapes phase structure." Pitch: *you sweep geometry (L, ρ); I sweep reward
  structure (w) and behavioral preference (λ); the open question is what property
  of the environment sets the boundary's shape. Candidate answers: exploration-noise
  binding, and boundary sharpness under the critic.*
- **To actually test it** you would need a deliberately designed environment ladder
  varying reward sparsity (and critic type) — a next-paper experiment, not a
  17-day one. Do NOT attempt before the AAAI deadline.

## Caveats / data hygiene before this ever goes near a paper

- Authoritative per-cell sources: Supply/PPO = `expPPO_lambda_sweep`; Supply/MAPPO
  = `expMAPPO_orgym_figures/mappo_orgym_raw_data.csv`; MPE = `expMAPPO_mpe_figures`
  lambda sweep; LBF = `expMAPPO_lbf_figures/lbf_wcrit_series.csv` (has CIs; the raw
  `mappo_lbf_raw_data.csv` grid starts at w=0.3, above LBF's low-w boundary, so it
  reads all-no-collapse — do not fit w_crit from that file).
- The MPE "affine fits well" result should be re-confirmed on the exact per-λ fit
  before any claim — MPE's collapse polarity/scale differs and my quick grid fitter
  needed per-cell handling.
- Everything above used the paper-consistent / raw series; the intermediate-λ
  numbers for a couple of cells were re-fit from raw here, which is why the
  Supply/MAPPO non-monotonicity is now confirmed real rather than an interpolation
  artifact.
