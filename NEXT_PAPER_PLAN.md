# Next paper — plan

**Working title (placeholder):**
*What Sets the Shape of the Coordination-Collapse Boundary? A Sparsity–Sharpness
Account in Cooperative MARL*

**One-sentence pitch:** the first paper measured *where* cooperative-MARL
coordination collapses and showed loss aversion moves that boundary; this paper
explains *why the boundary has the shape it does* — and turns the single-substrate
affine law into a predicted, environment-conditioned family.

**Status:** planning. Depends on the AAAI-27 paper being submitted first (it is the
"instrument" this paper reuses). Date: 2026-07-04.

---

## 1. Why this paper exists (the gap the first paper leaves)

The AAAI paper is honest about three limits, and this paper attacks the one that is
actually *tractable with new experiments*:

- The affine law `w_crit = φ/(φ+g/c)`, `φ = 1+ρ(λ−1)` is **substrate-specific** —
  it fits Supply/PPO (and, under-reported, MPE) but fails on LBF and degrades on
  Supply/MAPPO.
- Re-analysis of existing data (see `RIDGE_SHAPE_NOTE.md`) shows the law degrades
  for **two distinct reasons**, not one:
  1. **Exploration-noise binding** — sparse-reward environments (LBF) where injected
     entropy moves the boundary; the affine form breaks and `w_crit` runs past the
     `w=1` ceiling.
  2. **Boundary sharpness under the learner** — the centralized critic (MAPPO)
     smears the low-λ transition (a ~0.30 collapse floor that never clears),
     making `w_crit` ill-defined at low loss aversion.

The mechanism in paper 1 fixes **sign and form, not magnitude**, and two
pre-registered probes failed to localize `w_crit`. This paper does not promise to
localize magnitude either — it does something more attainable and still valuable:
**predict the boundary's shape (and when the affine law holds) from measurable
environment/learner properties.**

## 2. The central claim to test

> The shape of `w_crit(λ)` — and whether the affine law holds — is governed by
> **(a) whether exploration noise is the binding term** (a reward-sparsity axis) and
> **(b) boundary sharpness under the learner** (a credit-assignment axis).

This is falsifiable, unlike "what predicts ridge shape." Each factor makes a
directional prediction:

- **Sparsity prediction:** as reward sparsity increases (holding everything else
  fixed), the affine law's misfit (RMSE) rises monotonically and the boundary's
  high-λ curvature turns convex / runs off the ceiling.
- **Sharpness prediction:** as the learner's critic becomes more centralized, the
  low-λ transition width Δw grows and `w_crit` becomes less well-identified; the
  affine misfit rises specifically at low λ.

## 3. What must be built — the experiment ladder

The first paper had 4 opportunistic env×algo cells. This paper needs a **designed
ladder** that varies ONE property at a time. That is the whole methodological point.

### Experiment A — Sparsity ladder (the core contribution)
- Take ONE environment family and dial reward sparsity across ≥5 controlled levels
  (e.g. dense shaped reward → increasingly delayed/sparse terminal reward).
- Candidates: a gridworld or foraging family where sparsity is a clean knob, or an
  or-gym variant with tunable reward delay. (Decide the substrate early; it must let
  sparsity move independently of geometry.)
- Hold algorithm, horizon, seeds fixed. Full λ-sweep per sparsity level.
- **Primary output:** affine-law RMSE and ridge curvature as a function of sparsity.
  If they rise monotonically → the binding-noise hypothesis is confirmed on a
  *designed* axis, not inferred from 2 opportunistic points.
- **Tie-in:** repeat the entropy ablation at each sparsity level. The prediction is
  that the entropy effect on `w_crit` grows with sparsity — a second, independent
  confirmation (same move that made the dissociation strong in paper 1).

### Experiment B — Sharpness / critic ladder
- Same environment, vary credit-assignment centralization: independent PPO →
  centralized-critic MAPPO → value-factorized (QMIX) where measurable.
- **Primary output:** low-λ transition width Δw and `w_crit` identifiability as a
  function of critic centralization.
- Confirms factor (b) is real and separable from factor (a).

### Experiment C — The 2×2 (or larger) predicted grid
- Cross sparsity × critic. Show the two factors compose predictably: affine law
  holds only in the dense-reward, sharp-boundary corner; degrades along each axis
  in the predicted direction.
- This is the figure that makes the paper: a **phase-diagram-of-phase-diagrams**,
  where each cell is a `w_crit(λ)` ridge and the meta-axes are the two factors.

## 4. What would make this an ACCEPT (not another borderline)

Paper 1's ceiling was "borderline, single-substrate quantitative core." This paper
is designed to remove that exact criticism:

- **Designed axes, not opportunistic cells.** n ≥ 5 along each factor, so
  "ridge shape ~ property" becomes a fitted, CI'd relationship, not a 4-point
  correlation. This is the single biggest upgrade over paper 1.
- **Two independent confirmations per factor** (ridge shape AND entropy-ablation
  magnitude both tracking sparsity). Dissociation-style evidence is what reviewers
  couldn't dismiss last time.
- **A prediction that stays null where it should** — pick at least one environment
  where the mechanism says the affine law should hold, and confirm it does. Risky
  predictions that survive are the currency.
- **Keep the CI honesty.** Bootstrap CIs on every `w_crit`; report affine misfit
  with uncertainty. Do not swap the honest line-plots for pretty heatmaps without
  error bars (a temptation from paper-1's outreach figure).

## 5. What this paper explicitly does NOT claim

- It does **not** localize the magnitude of `w_crit` from first principles — that
  remains open (paper 1's two failed probes stand).
- It does **not** claim a single universal law. The finding is a *conditioned
  family*: the affine law is the dense-reward, sharp-boundary special case.
- It does **not** re-litigate paper 1's behavioral result — loss aversion moving
  the boundary is assumed and reused, not re-proven.

## 6. Collaboration angle (Yamaguchi)

This is the natural joint project:
- Yamaguchi sweeps **environment geometry** (grid size L, density ρ) and finds a
  three-phase structure driven by kernel drift.
- You sweep **reward structure** (w) and **behavioral preference** (λ).
- Shared open question: *what property of the environment sets the boundary's
  shape?* Your candidate answers (sparsity binding, critic sharpness) plug directly
  into their geometry axes. A combined study — geometry × reward-structure × behavior
  — is a stronger paper than either alone, and both of you already speak the
  statistical-mechanics idiom (their kernel drift ↔ your SNR/Kramers).
- Concrete first step: after AAAI submission + arXiv, send the 2×2 phase figure and
  this hypothesis; propose Experiment A's sparsity ladder as the shared testbed.

## 7. Rough sequencing (not dates)

1. AAAI-27 submitted (July) — freezes the instrument this paper reuses.
2. Post paper 1 to arXiv; open the Yamaguchi conversation.
3. Pick and instrument the sparsity-tunable environment (the critical design
   decision — everything depends on sparsity being a clean, isolated knob).
4. Run Experiment A (sparsity ladder + per-level entropy ablation).
5. Run Experiment B (critic ladder).
6. Compose Experiment C grid; build the phase-diagram-of-phase-diagrams figure.
7. Target venue: AAMAS or AAAI next cycle (or IJCAI if timing fits). If AAAI paper 1
   is rejected, this paper's stronger generalization may make the *combined* story a
   better single submission — consider merging rather than two borderline papers
   (AAAI-27's own guidance warns against thin near-duplicate submissions).

---

*Seed data and the evidence behind the two-factor hypothesis are in
`RIDGE_SHAPE_NOTE.md`. This plan uses existing findings only; every experiment above
requires new runs and must not be attempted before the AAAI-27 deadline.*
