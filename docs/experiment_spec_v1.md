# Experiment Spec v1 — The w×λ Collapse Surface

*A combined credit-assignment × behavioral phase-transition study on the base-stock MARL supply chain.*

**Spine of the paper (one sentence):** Cooperative multi-echelon MARL exhibits a *collapse phase transition* governed by reward centralization `w`, and loss aversion `λ` shifts the location of that transition — so a behavioral bias does not cause collapse on its own, it moves the boundary at which structural collapse occurs.

This reframes the original `λ_crit` idea correctly: the phase transition is real, but it lives in a **2-D control plane (w, λ)**, not a 1-D λ axis. That is novel relative to everything in your lit review (LAIES/SeCA/IMARL study `w`-collapse qualitatively; nobody maps a behavioral×centralization collapse surface).

---

## 0. The thing you must internalize before running anything

Your headline number so far (`Mean S ~ log(λ)`, slope −24.99, R²=0.115) is almost certainly a **line fit through a bimodal mixture**: centralized seeds either collapse (S≈0) or recover (S≈180), and OLS on that two-cluster cloud reports a slope that is really tracking *the fraction of collapsed seeds per λ bin*. So the dependent variable you actually care about is **P(collapse)**, a binary, not `Mean S`, a continuous. Every experiment below is built around modeling collapse as a discrete event. This single change fixes the statistical foundation.

**Define collapse once, globally, and never change it:**

```
collapsed(run) := (Mean_S < S_floor_threshold) AND (Profit <= profit_floor + epsilon)
```

Recommended: `S_floor_threshold = 10`, `profit_floor = -128.1`, `epsilon = 1.0`.
Justify the threshold with a histogram of `Mean_S` across all runs (it should be visibly bimodal with a gap — put the cut in the gap). **Pre-register this definition** (write it in a file, commit it) before you look at the new sweep results, so nobody can accuse you of tuning the cut to get a clean transition.

---

## Experiment A — Interior-w sweep + logistic collapse model

**Purpose.** Establish that collapse is a sharp transition in `w` (reward centralization), and estimate `w_crit`. This is the backbone result.

**What already exists you're reusing.**
- `phase3_8_incentives.py` already sweeps `global_reward_weight ∈ {0, 0.25, 0.5, 1.0}` — it just doesn't save a CSV. **You are mostly adding `df.to_csv(...)` to a script that already runs.**
- `scaled_base_stock_wrapper.py` (the learnable substrate).
- `CPTRewardWrapper` with `global_reward_weight=w`.

**Configuration.**
| Knob | Values | Notes |
|---|---|---|
| Substrate | `scaled_base_stock_wrapper`, range `[0,500]` | the learnable one; this is fixed |
| `w` (= "Alpha") | `{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}` | 11 levels — you need density near the transition, not just endpoints |
| `λ` | `1.0` (no loss aversion) | hold λ fixed here; this experiment isolates `w` |
| α, β curvature | `1.0` | fixed (as in all prior phases) |
| Seeds | ≥ 20 per cell (use your Phase 3.7 20-seed protocol) | binary outcomes need many seeds; 5 is uninterpretable |
| Training | 150k steps, PPO `MlpPolicy`, `n_steps=128`, `batch_size=256` | match prior phases exactly |
| Eval | deterministic, ≥10 episodes, eval-seed varied per episode | more eval episodes than the 3–5 you used; reduces eval noise |

That's 11 × 20 = **220 runs** at minimum. If compute is tight, drop to 7 `w` levels `{0,0.15,0.3,0.45,0.6,0.8,1.0}` × 20 seeds = 140 runs. Do **not** drop seeds below 20 to buy more `w` levels — for a binary outcome, seeds are worth more than grid density.

**Columns to save (one row per run):**
`w, lambda, seed, mean_S, profit, lost_sales, bullwhip, mean_order, collapsed (0/1), final_value_loss, final_explained_variance, train_steps`

**Primary analysis — logistic regression, NOT OLS:**

```
P(collapsed) ~ logit(b0 + b1 * w)
```

Report: `b1` with 95% CI, the fitted curve, and `w_crit` = the `w` at which P(collapse)=0.5, i.e. `w_crit = -b0/b1`, with a CI via the delta method or bootstrap over seeds.

**Secondary analyses (do all three):**
1. **Per-cell collapse fraction plot** with Wilson (not normal-approx) binomial CIs. This is the figure that makes the transition visible and is robust to any model misspecification.
2. **Sharpness test.** Fit logistic with `w` and `w²`; if the quadratic adds nothing and `b1` is large, the transition is sharp. Report the 10%–90% transition width (`Δw` between P=0.1 and P=0.9). A narrow width is your "phase transition" evidence; a wide ramp is just a gradient and you should call it that.
3. **Seed-as-random-effect** (mixed-effects logistic, `w` fixed, seed random) to confirm the `w` effect survives once seed clustering is modeled. This pre-empts the "it's all seed variance" attack.

**Falsification checks built in (run these, report them):**
- *Eval-noise artifact?* Re-evaluate 30 collapsed and 30 recovered runs with 50 episodes each; collapse label must be stable (>95% agreement). If labels flip, your eval is too short.
- *Initialization basin, not w?* This is what Phase 3.96 hinted at. See Experiment C.
- *Value-function failure?* Log `final_value_loss` and `explained_variance`; if collapsed runs simply never fit a critic, the story is "PPO critic fails at low w," which is a *different* (and still publishable) mechanism. You want to know which it is.

**Success criterion (pre-registered):** A monotone logistic fit with `b1` significant (CI excludes 0), a transition width `Δw < 0.4`, and per-cell collapse fractions that are ~0 at high `w` and ~1 at low `w`. If the collapse fraction is flat across `w`, the `w`-transition story is **falsified** — report that honestly; it's still a result.

---

## Experiment B — The 2-D (w, λ) collapse surface

**Purpose.** This is the *combined story* and the actual novel contribution: does loss aversion move `w_crit`? Tests the original behavioral claim correctly, on the substrate where it can show up.

**Configuration.**
| Knob | Values |
|---|---|
| `w` | `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` (6 levels — interior coverage) |
| `λ` | `{1, 2, 3, 5, 7, 10}` (6 levels — your existing PT range) |
| Seeds | ≥ 15 per (w, λ) cell |
| Everything else | identical to Experiment A |

6 × 6 × 15 = **540 runs.** Tight-budget version: 4 `w` × 4 `λ` × 15 = 240.

**Primary analysis — 2-D logistic surface:**

```
P(collapsed) ~ logit(b0 + b1*w + b2*log(λ) + b3*(w * log(λ)))
```

- `b1` (centralization protects against collapse) — expected strong, negative on collapse.
- `b3` **interaction** is the money term: does λ shift `w_crit`? Compute `w_crit(λ) = -(b0 + b2*log λ) / (b1 + b3*log λ)` and plot `w_crit` vs λ with bootstrap CIs.
- **The headline figure of the paper** is a heatmap of P(collapse) over the (w, λ) grid with the `w_crit(λ)` boundary overlaid. If that boundary tilts with λ, you have a behavioral phase transition. If it's vertical (flat in λ), then λ genuinely doesn't matter and your honest finding is "centralization is everything, behavior is a red herring" — which is *also* a clean, defensible, publishable claim (a negative result against the entire behavioral-supply-chain framing).

**This is the experiment that determines which paper you have.** Either outcome is publishable; only a *muddled* outcome (underpowered, OLS-on-mixture) is not.

---

## Experiment C — Basin-of-attraction / initialization control (kills the biggest confound)

**Purpose.** Your single most dangerous confound: maybe "collapse" is just which **basin the policy initializes into**, and `w`/`λ` only correlate with it. Phase 3.96 (`offset_base_stock_wrapper`, `S_init ∈ {0,25,50,80,150,250}`) already built the machinery. You must show the `w`-transition is **not** explained by initialization.

**Configuration.**
| Knob | Values |
|---|---|
| `S_init` (via offset wrapper) | `{0, 50, 80, 150, 250}` |
| `w` | `{0.2, 0.5, 0.8}` (low / mid / high, spanning the transition) |
| `λ` | `1.0` |
| Seeds | ≥ 10 per cell |

**Analysis.** For each `w`, plot converged `S` (last-10k mean) vs `S_init`.
- If converged-S depends on `S_init` → bistable, **initialization-driven**, and your "phase transition in w" is partly an artifact of which basin training fell into. You'd reframe as "w controls basin *width*."
- If converged-S is independent of `S_init` and determined by `w` → the transition is a genuine property of the reward structure, not initialization. **This is the result you want and it directly defends Experiment A.**

Report this as an ablation in the paper. Reviewers *will* ask "is this just initialization?" — answer it before they ask.

---

## Experiment D — Heterogeneous-λ minority (the actual original hypothesis, done right)

**Purpose.** The original design.md claim was that a **minority** of loss-averse agents triggers system collapse. You never tested this on the learnable substrate. Experiments A–B use *homogeneous* λ. This one restores the φ-fraction design.

**What exists.** The averse-subset logic is in `phase2_divergence.py` and `phase1_runner.py` (averse node sets per φ). Port it onto the `scaled_base_stock_wrapper` substrate.

**Configuration.**
| Knob | Values |
|---|---|
| φ (fraction of averse nodes) | `{0, 1/6, 2/6, 3/6, 4/6, 1.0}` (0 to all 6) |
| averse-node λ | `5` (averse), rational nodes λ=1 |
| `w` | fix at the *mid* value near `w_crit` from Exp A (where the system is most sensitive) AND at a high stable value (e.g. 0.8) |
| Seeds | ≥ 15 per cell |

**Analysis.**
```
P(collapsed) ~ logit(b0 + b1*φ)   [run separately at each w level]
```
- **The novel claim, precisely stated:** at a centralization level `w` that is stable under all-rational agents, does adding a *minority* averse fraction (small φ) tip the system into collapse? If P(collapse) jumps at small φ when held at mid-`w`, that's the "minority triggers macroscopic collapse" result — the original vision, finally on a substrate that can express it.
- Also report **which** nodes being averse matters (echelon sensitivity) — you have node-position data; a factory-averse vs retailer-averse contrast is a strong secondary finding.

**Honest prior:** your homogeneous-λ data so far show ≈0 λ main effect, so this may well come back null. That's fine and still informative: "even a behavioral minority cannot move the boundary; centralization dominates." Run it to find out, not to confirm.

---

## Experiment E — Published baseline (non-negotiable for any top venue)

**Purpose.** Every reviewer who reads your lit review will ask: "you cited Beta-policy / hurdle / structured base-stock fixes for exactly this pathology — why didn't you use them?" You need at least one.

**Minimum viable baseline.** A **Beta-distribution policy head** on the *raw* action space (SB3 supports custom distributions, or swap to a Beta via a thin custom `Distribution`). Run all-rational, decentralized (`w=0`), the configuration where your raw-action PPO collapsed.
- If Beta-policy raw-action recovers where Gaussian raw-action collapsed → confirms your "Gaussian pathology" claim *and* shows you know the fix. Strong.
- If Beta-policy *also* collapses at low `w` → even stronger: proves collapse is a **credit-assignment** problem, not an action-parameterization problem, which sharpens your whole thesis.

If a custom Beta head is too much lift, the cheaper baseline is the **structured base-stock action you already have** vs **raw Gaussian**, both at matched `w`, to cleanly separate "action parameterization" from "reward centralization" as causes. You partially have this; just make it a deliberate, saved, head-to-head table.

---

## Statistical discipline checklist (apply to every experiment)

- [ ] Binary outcome → **logistic**, never OLS on the floored continuous.
- [ ] **Wilson** binomial CIs on per-cell collapse fractions.
- [ ] **Seed as random effect** at least once to defend against seed-variance attack.
- [ ] **Bootstrap over seeds** for any `w_crit` / boundary CI.
- [ ] **Pre-register** the collapse definition and success criteria in a committed file before looking at new results.
- [ ] **Save every CSV** — no stdout-only phases. (Your handoff lists 8 phases with no saved table; do not repeat that.)
- [ ] Report **effect size** (transition width Δw, η²/odds ratios) separately from **p-values**.
- [ ] Multiple-comparison honesty: you're fitting several models; say so, and lead with the pre-registered primary.

---

## What you actually have to do, in order

1. **Day 0 (no compute):** Write `collapse_definition.md`, commit it. Pull every existing CSV (`phase4_raw_data.csv`, `master_seed_dataset.csv`) and **re-plot `Mean_S` as a histogram**. Confirm bimodality and place `S_floor_threshold` in the gap. This single plot either confirms or refutes the "OLS-on-mixture" diagnosis and costs nothing.
2. **Fix the saving bug:** add `to_csv` to `phase3_8_incentives.py` and re-run it as-is → that's a cheap first slice of Experiment A's interior-`w` data.
3. **Run Experiment A** (interior-w, 20 seeds). This is the backbone. Get `w_crit`.
4. **Run Experiment C** (basin control) at the same time — it's small and it defends A.
5. **Run Experiment B** (2-D surface). This decides your paper's headline figure and which claim survives.
6. **Run Experiment D** (heterogeneous minority) — restores the original vision; null or not, it's informative.
7. **Run Experiment E** (one baseline) — required armor for ICML/NeurIPS; optional for a workshop.
8. **Re-save the unsaved Phase 4 stats** (η², Lost-Sales, Profit) — trivial, removes a reviewer flag.

**Stop conditions / honesty gates:**
- If Exp A collapse fraction is flat in `w` → no transition; pivot to "why is it flat" (basin/critic), don't force the phase-transition narrative.
- If Exp B boundary is vertical (flat in λ) → the paper is "centralization dominates, behavior doesn't" — a clean negative result against the behavioral framing. Write *that* paper; don't bury it.
- If Exp D is null → state plainly that a behavioral minority does not move the boundary. That contradicts design.md's hypothesis, and saying so is the scientifically correct outcome.

---

## What each result buys you, by venue

- **A alone, clean:** workshop paper / short paper. "A credit-assignment collapse transition in cooperative MEIO MARL."
- **A + B with a tilted boundary:** full conference submission. "Loss aversion shifts the collapse boundary of cooperative supply-chain MARL" — genuinely novel, combines both threads.
- **A + B + D + E with the minority effect real:** the moonshot. "A behavioral minority triggers a credit-assignment phase transition" — field-relevant, defensible, original. High risk: your current data suggest λ may simply not matter, in which case you still have the strong negative result, which a good reviewer respects.

The discipline that makes any of these survive review is the same: **model collapse as a discrete event, sweep the interior, many seeds, pre-register the cut, save everything, and report the null as loudly as the positive.**
