# Handoff — Experiment A session (the w×λ collapse-surface study)

*Date: 2026-06-03 · Branch: `experiments/w-lambda-collapse-surface` · Maintainer: A. Yadav*

Covers the session that executed [docs/experiment_spec_v1.md](../experiment_spec_v1.md)
through the end of Experiment A. Companion to the project-wide inventory in
[handoff_v1.md](handoff_v1.md).

---

## Goals: what we're trying to build

A single combined contribution out of two previously-weak threads: the
prospect-theory `λ_crit` idea and the continuous-PPO collapse finding. The spine
(per the spec): **cooperative multi-echelon MARL has a collapse phase transition
governed by reward centralization `w`, and loss aversion `λ` may shift where that
transition sits.** The phase transition lives in a 2-D control plane `(w, λ)`, not a
1-D `λ` axis.

The methodological core: stop modeling `Mean_S` as a continuous outcome (it's a
floored bimodal mixture) and model **collapse as a binary event** via logistic
regression, on the learnable **base-stock substrate** (raw-action substrate is
confounded into uselessness and is reserved as a labeled negative control only).

Target outcomes, both pre-committed as publishable:
- **Tilted** `w_crit(λ)` boundary → "loss aversion shifts the collapse boundary."
- **Vertical** boundary → "centralization dominates; λ is a red herring" — reported
  with equal prominence (binding commitment in the Exp B pre-registration).

## Current state: where the work stands right now

- **Day 0 diagnostic: done.** `Mean_S` confirmed bimodal across the 95 existing
  runs (40/95 in [0,10)); collapse definition frozen and pre-registered.
- **Experiment A: COMPLETE and positive.** 220 runs (11 `w` × 20 seeds @150k).
  Sharp collapse transition in `w`, **`w_crit = 0.822`** (bootstrap CI [0.785,
  0.858]), all pre-registered success criteria met. This is the backbone result.
- **Experiment B: pre-registered (while blind to A), not yet run.** Analysis code
  + decision rule committed. A grid-coverage contingency has now **triggered** (see
  Next step).
- **Experiments C, D, E: not started** (basin control, heterogeneous-λ minority,
  published baseline).
- Two spec "hygiene" items cleared: the `phase3_8` stdout-only saving bug is fixed,
  and the placeholder Phase 4 stats file now holds real numbers.

Everything is committed on branch `experiments/w-lambda-collapse-surface` (5 session
commits, `03c3fc6` → `9d100f9`). Not yet merged to `main`; no PR opened.

## Files in flight: active files

Created this session (all committed):
- [docs/collapse_definition.md](../collapse_definition.md) — frozen, pre-registered
  collapse label. **Do not edit**; amendments require a dated v2.
- [docs/expB_preregistration.md](../expB_preregistration.md) — frozen Exp B plan +
  the equal-prominence negative-result commitment. **Do not edit.**
- [experiments/expA_interior_w.py](../../experiments/expA_interior_w.py) — Exp A
  runner (interior-w sweep, one labeled row/run, CLI-parameterized).
- [experiments/analyze_expA.py](../../experiments/analyze_expA.py) — Exp A analysis
  (logistic, `w_crit`, Wilson CIs, bootstrap, sharpness).
- [experiments/analyze_expB.py](../../experiments/analyze_expB.py) — Exp B analysis;
  `decide_outcome()` encodes the frozen §5 rule; produces the headline heatmap.
- [docs/expA_figures/](../expA_figures/) — `expA_raw_data.csv` (220 rows),
  `expA_logistic.png`, `expA_cellfractions.csv`, `mean_S_bimodality.png`.

Present in the tree but **not authored this session** (pre-existing untracked work;
left untouched, not mine to commit): `experiments/phase6_behavioral_claim.py`,
`docs/phase6_figures/`, `commit_new.sh`. The next maintainer should reconcile these
separately — they look like an earlier exploratory cut of the same behavioral claim.

## Changed: what's been touched in this session

| File | Change |
|---|---|
| `docs/collapse_definition.md` | **new** — pre-registered binary collapse label |
| `docs/expB_preregistration.md` | **new** — frozen Exp B analysis + decision rule |
| `experiments/expA_interior_w.py` | **new** — Exp A sweep runner |
| `experiments/analyze_expA.py` | **new** — Exp A logistic analysis |
| `experiments/analyze_expB.py` | **new** — Exp B surface analysis (ready, unrun) |
| `experiments/phase3_8_incentives.py` | **edit** — added per-run `to_csv` (was stdout-only) |
| `docs/phase4_figures/extended_regression_results.txt` | **edit** — placeholder → real η²=0.132, Lost-Sales & Profit regressions |
| `docs/expA_figures/*` | **new** — Exp A data + figures |
| `logs/expA_run.log` | **new** — sweep run log |

`docs/handoff/handoff_v1.md` was produced in a prior session (already tracked).

## Failed attempts: what didn't work and why

- **Logistic fit on 2-level endpoint data (self-tests).** With only `w∈{0,1}` the
  data are quasi-separated (100% vs 10% collapse) → singular Hessian, `LinAlgError`.
  Fixed by adding an **L2-regularized fallback** (`fit_regularized`, α=1e-3) and a
  `nunique<3` guard on the sharpness test. Not a substantive failure — expected
  behavior with no interior points; the full 11-level Exp A data fit cleanly via
  plain MLE.
- **Short-horizon runs do not reproduce collapse.** A 4k-step smoke test gave
  S≈170–182 / profit≈450 / *not collapsed*; collapse only emerges over long
  training (≈20k+). Consequence: the horizon **cannot** be shortened to save
  compute without invalidating the collapse label — all Exp A runs use the full
  150k, matching the calibration data.
- **Minor DataFrame formatting bug** (`{:d}` on a float cell) — trivial cast fix.

No experiment produced a null or contradicted the plan; the "failures" were all
tooling robustness issues, now hardened.

## Successful attempts: what worked and why, and the results

- **Day 0 bimodality check (zero compute).** Pooling the 95 existing runs and
  histogramming `Mean_S` confirmed the spec's core diagnosis: a collapse spike of
  40/95 in [0,10) then a gap. The collapse count is **robust** to the threshold (32
  at S_floor ∈ {5,10,15}) and the AND-with-profit matters (8 runs are S<10 but
  profitable, correctly excluded). This justified the frozen label and the switch to
  binary modeling. Figure: `docs/expA_figures/mean_S_bimodality.png`.
- **Experiment A — the headline success.** 220 runs. Collapse fraction is flat at
  **1.00 for all w ≤ 0.6**, then drops: 0.85 (0.7), 0.55 (0.8), 0.20 (0.9), 0.10
  (1.0). Logistic `P(collapse)~w`: **b1 = −15.78** (p = 5×10⁻⁹, CI [−21.1, −10.5]);
  **w_crit = 0.822** (bootstrap CI [0.785, 0.858]); **transition width Δw = 0.278**
  (< 0.4 → sharp); quadratic term adds nothing (p = 0.21 → cleanly monotone). Every
  pre-registered success criterion met. Interpretation: the cooperative system
  collapses unless reward is *almost fully* centralized (`w ≳ 0.82`). Figure:
  `docs/expA_figures/expA_logistic.png`.
- **Pre-registering Exp B while genuinely blind to A.** The Exp B model, estimand,
  headline figure, and the symmetric TILTED/VERTICAL decision rule (with the
  binding equal-prominence commitment) were committed at `96e0a1b` *before*
  `analyze_expA.py` was ever run on the real data. `analyze_expB.py` encodes the
  rule and was validated end-to-end on existing endpoint data.
- **Hygiene:** `phase3_8` now saves a per-run CSV; the Phase 4 placeholder file now
  reports η² = 0.132 (λ explains 13% of `Mean_S` variance, 87% seed noise),
  Lost-Sales (p=0.005), Profit (p=0.009).

## Next step: the single next thing to try

**Run Experiment B** — the 2-D `(w, λ)` collapse surface, the experiment that
decides the paper's headline claim. Per the pre-registration, the frozen `w` grid
{0, 0.2, 0.4, 0.6, 0.8, 1.0} stays, but because **A returned `w_crit = 0.82`, the
declared grid-coverage contingency is triggered**: add interior `w` levels near the
transition (≈ 0.7, 0.9, 0.95) so the boundary is actually resolved where it lives,
rather than straddled only by 0.8 and 1.0. λ grid {1, 2, 3, 5, 7, 10}, ≥15 seeds.

Command (when ready; ~3–5 h wall, not days — ~540 runs at ~5–6 min/run over 10
procs, comparable to Exp A):

```
python experiments/expA_interior_w.py \
  --ws 0.0,0.2,0.4,0.6,0.7,0.8,0.9,0.95,1.0 --seeds 42-56 \
  --lam <swept per λ-level>   # note: extend the runner to sweep λ, or add an expB runner
```

Implementation note: `expA_interior_w.py` currently holds λ fixed (`--lam`). Exp B
needs a λ sweep too — either add a `--lams` arg that does the outer product, or
write a thin `expB_run.py` reusing `_run_single`. Then analyze with
`python experiments/analyze_expB.py --in docs/expB_figures/expB_raw_data.csv`,
which will print the pre-registered TILTED/VERTICAL verdict and save the headline
heatmap automatically.
