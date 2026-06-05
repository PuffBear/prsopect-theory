# Research Plan: Prospect-Theoretic Phase Boundary in Cooperative MARL
**Project:** Prospect Theory on Deep Learning — Supply Chain MARL  
**Document version:** v2 | Date: 2026-06-04  
**Target outputs:** arXiv preprint (2 weeks) + AAAI 2027 submission (~August 2026)

> **How to use this document:** Every experiment section has two sub-blocks: (1) **Claude Code Instructions** — the exact prompt to paste, and (2) a **Game Tree** — what to do when things go wrong. Read the game tree *before* you run the experiment, not after.

---

## Table of Contents

1. [Where We Stand](#1-where-we-stand)
2. [The Possible Worlds Analysis](#2-the-possible-worlds-analysis)
3. [Literature Kill-Condition Status](#3-literature-kill-condition-status)
4. [The Surviving Novelty Claim](#4-the-surviving-novelty-claim)
5. [What the Data Actually Says](#5-what-the-data-actually-says)
6. [Critical Problems with the Current Story](#6-critical-problems-with-the-current-story)
7. [The 3 Missing Experiments](#7-the-3-missing-experiments)
8. [The 2-Week Execution Plan](#8-the-2-week-execution-plan)
9. [HPC Setup (Ashoka)](#9-hpc-setup-ashoka)
10. [The Paper — Narrative and Structure](#10-the-paper--narrative-and-structure)
11. [The Figure That Makes or Breaks This Paper](#11-the-figure-that-makes-or-breaks-this-paper)
12. [Reviewer Attacks and Pre-emptions](#12-reviewer-attacks-and-pre-emptions)
13. [AAAI 2027 Assessment](#13-aaai-2027-assessment)
14. [Future Work Directions (Post-AAAI)](#14-future-work-directions-post-aaai)

---

## 1. Where We Stand

### Codebase Summary (Branch: `experiments/w-lambda-collapse-surface`)

**Environment:** or-gym `NetworkManagement-v1`, multi-echelon supply chain, 6 agent nodes, Poisson demand μ=20, `num_periods=30`.

**Algorithm:** Continuous PPO (SB3), `MlpPolicy`, `n_steps=128`, `batch_size=256`, SuperSuit vectorization.

**Key wrappers:**
- `CPTRewardWrapper`: blended reward = `(1−w)·local + w·global`, then PT value function `v(x) = x^α` if x≥0, `-λ·(−x)^β` if x<0
- `ScaledBaseStockWrapper`: action [−1,1] → target base-stock S∈[200,700]; order Q = max(0, S−IP)
- `w` = `global_reward_weight` ∈ [0,1] — the central experimental parameter

**Collapse definition (pre-registered, frozen):**  
`collapsed(run) := (mean_S < 10) AND (profit ≤ −128.1 + 1.0)`

### Completed Experiments

| Experiment | Status | Key Result |
|---|---|---|
| Phases 0–5 (PT validation, raw action, base-stock intro) | Complete | PT shaping works; raw continuous action → order≈0 pathology identified and fixed by base-stock wrapper |
| Phase 4 centralized (w=1, λ sweep, 70 runs) | Complete | High variance; some collapse, some reach S≈130–186 |
| Phase 4.5 decentralized (w=0, 25 runs) | Complete | 100% collapse, S=0 |
| Exp A @150k (w sweep, 220 runs) | Complete | w_crit=0.822, CI [0.785, 0.858], Δw=0.278 |
| Exp C (basin control, 144 runs) | Complete | Initialization does not drive collapse |
| Exp 0 (600k convergence, 24 runs) | Complete | Low-w collapse is robust to training horizon |
| 300k w_crit re-estimation (140 runs) | Complete | w_crit=0.666, CI [0.635, 0.694], Δw=0.199 |
| Exp 0b (horizon lock + metastability origin, 104 runs) | Complete | Q1=SLIDING; Q2=INCONCLUSIVE |
| Exp B (w×λ surface, 900 runs) | STOPPED at 20/900 | λ axis unmapped |
| Exp D (heterogeneous-λ minority) | NOT RUN | Behavioral minority claim untested |

---

## 2. The Possible Worlds Analysis

Before fixing the novelty, enumerate every world this research could inhabit. The advisor's framing: "there might be a world where this problem just doesn't exist."

### World 1 (30% probability): Real result, wrong story being told

The w_crit sliding with training horizon is not a failure — **it IS the finding**. You discovered that the collapse-survival boundary is a function of training horizon T. The "critical point" is a curve in (w, T) space, not a scalar. This is a coherent, novel result. Statistical physics calls this **finite-time scaling**: the system is metastable, w controls the energy barrier height for escaping the lazy-agent attractor basin.

**This world is where the paper lives if you reframe correctly.**

### World 2 (25% probability): The project doesn't exist yet — Exp B is the paper

Without the λ axis being quantified, you have a MARL credit assignment paper with a logistic curve. Publishable but not a "Prospect Theory on Deep Learning" paper. The behavioral contribution does not exist without Exp B-Reduced.

### World 3 (20% probability): Novelty is narrower than the lit review concluded

K2 is a **methodological** novelty, not a conceptual one. A NeurIPS reviewer asks: "Why is mapping w_crit interesting beyond this specific environment?" The answer requires either (a) theoretical generality, or (b) the λ modulation — behavioral agents change the phase boundary, which has broader implications. Without (b), the paper has an AAAI/AAMAS ceiling, not NeurIPS.

### World 4 (15% probability): Sliding w_crit kills the central claim

Hostile reviewer: "You didn't find a critical point. You found that with enough training, agents escape the lazy attractor regardless of centralization. Your 'phase boundary' is an artifact of your arbitrary training budget." **You currently have no defense against this.** The metastability reframe IS the defense — but it must be adopted deliberately, not retroactively.

### World 5 (10% probability): The environment is the problem

or-gym's `NetworkManagement-v1` is not a recognized ML benchmark. If the collapse behavior is a pathology of this specific environment (zero-inflated optimal base-stock creating the action-space mismatch identified in `docs/defense.md`), results may not generalize. The null model (Experiment N) partially addresses this.

---

## 3. Literature Kill-Condition Status

| Kill-Condition | Trigger? | Implication |
|---|---|---|
| **K1:** PT/loss-aversion already run in cooperative MARL with λ-effect reported | **TRIGGERED** by Lalmohammed et al. (ICML 2025) — CPT-MADDPG on cooperative coverage tasks | L2 conceptual novelty dead. Must pivot to phase-boundary mapping as the contribution. Note: they modify actor/critic update mechanics; we use PT as reward shaping. Different architectural locus. |
| **K2:** w-transition already quantified with named critical point | **OPEN** — no paper computes a w_crit with CI and transition width for CTDE collapse | **This is the surviving spine.** |
| **K3:** "Behavioral minority shifts a tipping point" already exists | **TRIGGERED** by Mengesha et al. (arXiv 2026) in EGT + Granovetter ABM literature | L3 moonshot dead conceptually. May survive as "first in deep MARL." Low priority. |
| **K4:** Centralization-controls-collapse is folklore / too obvious | **TRIGGERED** by Peng (2023) CURO and Liu (2023) lazy agents literature | L1 unpublishable alone. Behavioral axis is the differentiator. |

**Bottom line from lit review:** The sole surviving avenue is K2 — the quantitative phase-boundary mapping — modulated by the λ behavioral axis. The paper is a statistical mechanics treatment of cooperative MARL collapse, with PT loss aversion as the modulating factor.

### Key Papers to Cite and Distinguish

- **Lalmohammed et al. ICML 2025** (CPT-MADDPG): "They integrate CPT into actor/critic update mechanics; we apply it as a reward transform and measure its effect on the structural coordination boundary — a complementary, not redundant, contribution."
- **Mengesha et al. 2026** (EGT zero-sum norms): "Proves loss aversion shifts equilibria in fixed-strategy EGT; we demonstrate this effect emerges in deep MARL with learned neural policies, where the mechanism is gradient dynamics rather than selection pressure."
- **Weng & Lee 2026** (exploration breaks cooperation): "Identifies the mechanism (shared representation degradation); we quantify the boundary at which this mechanism dominates, and show it is modulated by agent utility curvature."
- **Oroojlooyjadid 2022** (Beer Game DQN): Validates supply chain substrate credibility.
- **Anon 2024** (Risk-Sensitive MARL NAMGs): CPT in non-cooperative games; leaves cooperative setting open.

---

## 4. The Surviving Novelty Claim

> **"Cooperative MARL systems exhibit a training-horizon-dependent metastable coordination attractor. We characterize the collapse-survival boundary as a logistic isocline in (reward centralization w, training horizon T) space. Below the isocline, the lazy-agent basin is absorbing at finite T; above it, coordination emerges. Prospect-theoretic loss aversion λ modulates the depth of this basin — higher λ requires greater centralization to achieve coordination within the same horizon. We establish w_crit(T, λ) as the first quantitative behavioral phase diagram of cooperative MARL coordination collapse in a supply chain network."**

**Self-check against kill conditions:**
- K1 triggered — ✓ acknowledged. Lalmohammed does not map w_crit. Different locus of contribution.
- K2 open — ✓ the (w, T) isocline is unmapped in all reviewed literature.
- K3 triggered — ✓ we're not claiming "first behavioral minority tipping point in any field." We claim first in deep MARL with neural policies and supply chain substrate.
- K4 triggered — ✓ structural collapse is background, not contribution. The boundary mapping is the contribution.

**The contribution is methodological + empirical, not conceptual.** The claim survives because no one has computed the phase diagram, not because no one has thought about it.

---

## 5. What the Data Actually Says

### The Phase Transition is Real

| w | P(collapse) @150k | P(collapse) @300k |
|---|---|---|
| 0.0–0.6 | 1.00 | 1.00 |
| 0.7 | 0.85 | 0.40 |
| 0.8 | 0.55 | 0.00 |
| 0.9 | 0.20 | 0.00 |
| 1.0 | 0.10 | — |

Logistic fit @150k: b1=−15.78 (p=5×10⁻⁹), w_crit=0.822, CI [0.785, 0.858], Δw=0.278  
Logistic fit @300k: b1=−22.05 (p=3.2×10⁻⁷), w_crit=0.666, CI [0.635, 0.694], Δw=0.199

Exp C (basin control) rules out initialization sensitivity. Exp 0 shows low-w collapse is robust to 600k steps. The transition is real.

### The w_crit Slides — And That Is the Finding

Q1 of Exp 0b is formally "SLIDING." This should not be suppressed or treated as a nuisance:

- At 150k, w>0.822 is required for >50% survival probability
- At 300k, w>0.666 suffices
- At 600k, w=0.6 → 35% collapse, w=0.7 → 20% collapse

The isocline in (w, T) space is downward sloping. Agents are trapped in a metastable lazy-agent attractor that becomes easier to escape with more training. w controls the energy barrier height. **This is the metastability result.**

### The λ Effect Is Statistically Fragile

Phase 5 factorial (95 runs): logλ effect on mean_S: OLS coef=−24.99, p=0.0041, R²=0.115. Interaction (logλ×w): p=0.064–0.093. This is marginally significant at n=95 and does not constitute an established behavioral claim. Exp B-Reduced is required.

---

## 6. Critical Problems with the Current Story

### Problem 1: The "finite-time artifact" attack has no defense
If w_crit slides indefinitely with training horizon, a reviewer can argue there is no critical point — just insufficient training. **The metastability reframe is the defense.** You must adopt it explicitly: you're characterizing the horizon-dependent boundary, not claiming a fixed asymptotic critical point.

### Problem 2: The λ behavioral axis is not established
R²=0.115 on 95 runs with marginal interaction p. Without Exp B-Reduced, any behavioral claim is speculative. The paper currently cannot make the claim it wants to make.

### Problem 3: The null model is missing
Is the phase transition driven by the CPT reward transform, or by the credit assignment structure alone? Without running the same w sweep with λ=1, α=β=1 (i.e., no PT distortion whatsoever), you cannot separate these. A reviewer will immediately ask this.

### Problem 4: Collapse definition robustness
The definition `(mean_S < 10) AND (profit ≤ −128.1 + 1.0)` was pre-registered but the second clause is pinned to an environment constant. A threshold robustness check (±20% perturbation of both thresholds, show w_crit is stable) is needed in the appendix.

### Problem 5: Single environment
or-gym `NetworkManagement-v1` is not a recognized ML benchmark. The paper's generalizability claim must be limited to the phase-mapping methodology, not the specific w_crit value. State this explicitly.

---

## 7. The 3 Missing Experiments

Run all 3 simultaneously on HPC. Total: ~620 runs, ~6–8 hours wall time on one CPU node.

---

### Experiment N — Null Model (HIGHEST PRIORITY)

**Purpose:** Isolate the structural (credit assignment) component of collapse from the behavioral (PT reward shaping) component.

**Design:** Same as Exp A @300k, but with the CPT wrapper set to linear (λ=1, α=β=1.0, no probability distortion). Pure ERM reward — no behavioral bias.

| Parameter | Value |
|---|---|
| w values | {0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} |
| Seeds | 42–61 (20 per w level) |
| Steps | 300k |
| PT params | λ=1, α=1, β=1 (linear = no shaping) |
| Runs | 180 |

**Analysis:** Fit logistic to null model P(collapse) vs w. Compare w_crit(null) to w_crit(PT, λ=1) from existing 300k data.

**Interpretation:**
- w_crit(null) ≈ w_crit(PT, λ=1): baseline structural transition is real; PT at λ=1 already approximates no-PT
- w_crit(null) << w_crit(PT, λ=1): PT reward transform creates additional instability even at λ=1 — the action-space/utility-suppression effect is real; behavioral excess is measurable
- The difference w_crit(PT, λ=k) − w_crit(null) is the "behavioral excess" attributable to loss aversion at each λ

#### Claude Code Instructions — Exp N

> **Paste this exactly into Claude Code when you're ready to build the runner script:**

```
I need you to write a new experiment script for my MARL supply chain project.
The codebase is on branch experiments/w-lambda-collapse-surface.

Context:
- Environment: or-gym NetworkManagement-v1 wrapped with MultiAgentNetInvMgmt,
  ScaledBaseStockWrapper (range [200,700]), and CPTRewardWrapper
- Algorithm: SB3 PPO with MlpPolicy, n_steps=128, batch_size=256,
  SuperSuit vectorization (pad_action_space_v0 → pettingzoo_env_to_vec_env_v1
  → concat_vec_envs_v1)
- Reference script: experiments/expA_interior_w.py

Write experiments/exp_null_model.py with these exact specs:
- w values: [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- seeds: range(42, 62) — 20 seeds per w level
- total_timesteps: 300_000
- CRITICAL: set CPTRewardWrapper with lambda_loss=1.0, alpha=1.0, beta=1.0
  (this makes v(x)=x, i.e., linear/no PT shaping — this is the null condition)
- Accept --run_idx argument (0-179), map to (w_idx, seed_idx) via:
    w_list = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seed_list = list(range(42, 62))
    w = w_list[run_idx // 20]
    seed = seed_list[run_idx % 20]
- Output: one CSV row appended to docs/expN_figures/expN_raw_data.csv
  with columns: run_idx, w, seed, mean_S, profit, lost_sales, collapsed
  Use a filelock to make it process-safe (use the filelock library, same
  pattern as expA_interior_w.py)
- collapsed := (mean_S < 10) AND (profit <= -127.1)
- Log final_value_loss and final_explained_variance from the PPO model's
  logger if accessible, otherwise set to NaN

Also write the SLURM script docs/slurm/run_expN.sh:
- job-name: exp_null
- array: 0-179
- cpus-per-task: 1
- mem: 4G
- time: 01:30:00
- output: logs/expN_%A_%a.out
- activate virtualenv at ~/envs/marl (adjust if different)
- run: python experiments/exp_null_model.py --run_idx $SLURM_ARRAY_TASK_ID
```

> **After the script is written, paste this for the analysis:**

```
Write experiments/analyze_expN.py that:
1. Loads docs/expN_figures/expN_raw_data.csv
2. For each w value, computes collapse fraction (collapsed==True) with
   Wilson score 95% CI
3. Fits a logistic regression: collapsed ~ w using statsmodels
   P(collapse|w) = 1 / (1 + exp(b0 + b1*w))
   Extract b1, p-value, pseudo-R2
4. Estimates w_crit = -b0/b1 with bootstrap CI (n=2000 resamples,
   resample at the run level)
5. Also loads docs/expA300k_figures/expA300k_raw_data.csv (existing PT data)
   and runs the same logistic on it
6. Prints side-by-side:
   w_crit(null model): X [CI_lo, CI_hi]
   w_crit(PT λ=1 @300k): X [CI_lo, CI_hi]
   behavioral_excess = w_crit(PT) - w_crit(null): X
7. Saves a figure to docs/expN_figures/expN_vs_PT_comparison.png showing
   both logistic curves on the same axes (null model in gray dashed,
   PT λ=1 in blue solid) with w_crit vertical lines and CI shading
8. Saves results summary to docs/expN_figures/expN_results.md
```

#### Game Tree — Exp N

```
ROOT: Exp N results arrive
│
├── w_crit(null) is similar to w_crit(PT, λ=1) [difference < 0.05]
│   ├── MEANING: The phase transition is structural (credit assignment),
│   │   not an artifact of PT reward shaping. Good — it validates
│   │   that Exp B-Reduced is measuring a real behavioral effect.
│   └── ACTION: Proceed. In the paper, state: "The structural transition
│       persists without any behavioral bias, confirming credit assignment
│       as the baseline driver. Behavioral excess is measured relative
│       to this null baseline."
│
├── w_crit(null) is substantially LOWER than w_crit(PT, λ=1) [difference > 0.1]
│   ├── MEANING: PT shaping at λ=1 is ADDING instability beyond
│   │   structural credit assignment. The CPT wrapper itself suppresses
│   │   coordination even at the lowest λ. This is the action-space
│   │   suppression hypothesis — PT makes all outcomes look bad,
│   │   agents become more conservative.
│   ├── ACTION (Good framing): Reframe the paper. The null model IS
│   │   the baseline. The "behavioral excess" at each λ is now
│   │   w_crit(PT,λ) - w_crit(null). This is a richer decomposition:
│   │   Structural baseline + PT baseline shift + λ-dependent shift.
│   └── ACTION (If bad for story): Check that the null model ran correctly —
│       verify CPT wrapper is genuinely linear (print v(1.0) and v(-1.0)
│       in the script, both should equal their inputs). If confirmed,
│       this is a real finding, not a bug.
│
├── w_crit(null) is HIGHER than w_crit(PT, λ=1) [PT is stabilizing at λ=1]
│   ├── MEANING: PT reward shaping at λ=1 actually helps coordination.
│   │   Counterintuitive but possible — the nonlinear reward scaling
│   │   could be amplifying positive signals.
│   ├── ACTION: First, check for bugs — is the null model actually
│   │   using the correct wrapper? Run a 2-run sanity check manually.
│   │   If confirmed, this is a surprising finding that changes the story:
│   │   "PT reward shaping at moderate λ acts as a coordination stabilizer
│   │   by amplifying reward contrast; only at high λ does loss aversion
│   │   dominate and destabilize." This is publishable but requires
│   │   Exp B-Reduced to confirm the crossover.
│   └── ESCALATE: Flag this to advisor before proceeding. Don't reframe solo.
│
└── Null model fails to produce a logistic transition [flat P(collapse)]
    ├── MEANING: Without PT, agents either always coordinate or never do,
    │   regardless of w. The environment's dynamics dominate completely.
    ├── PROBABILITY: ~5%. Low.
    └── ACTION: Increase w sweep granularity — run w in {0.0, 0.1, ..., 1.0}
        for 10 seeds. Check if the transition exists but is very sharp.
        If still flat, the structural credit assignment story is weaker
        than expected. The PT framing becomes the *only* mechanism —
        which actually strengthens L2 if the behavioral experiment works.
```

---

### Experiment B-Reduced — λ×w Surface

**Purpose:** Determine whether loss aversion λ shifts the phase isocline. The behavioral contribution lives or dies here.

| Parameter | Value |
|---|---|
| w values | {0.5, 0.65, 0.75, 0.85, 0.95} |
| λ values | {1, 3, 7} |
| Seeds | 42–61 (20 per cell) |
| Steps | 300k |
| α, β | 0.88 (standard CPT parameters) |
| Runs | 5 × 3 × 20 = 300 |

**Analysis:** For each λ, fit logistic curve to P(collapse) vs w → estimate w_crit(λ) with bootstrap CI. Test: is w_crit(λ=7) − w_crit(λ=1) > 0 with non-overlapping CIs?

**Interpretation:**
- **CIs separate:** λ modulates the isocline. Behavioral claim confirmed. This is Figure 2.
- **CIs overlap:** Behavioral claim absent. Paper pivots to structural-only framing.

Both outcomes are publishable. The honest null is informative.

#### Claude Code Instructions — Exp B-Reduced

> **Paste this to build the runner:**

```
I need you to write experiments/exp_b_reduced.py for my MARL supply chain project.
Branch: experiments/w-lambda-collapse-surface.
Reference script: experiments/expA_interior_w.py and agents/cpt_wrapper.py.

Specs:
- This is a 5 × 3 × 20 = 300 run experiment
- w_list = [0.5, 0.65, 0.75, 0.85, 0.95]
- lambda_list = [1, 3, 7]
- seeds = list(range(42, 62))
- total_timesteps = 300_000
- alpha = beta = 0.88 (standard CPT; these are fixed throughout)
- Accept --run_idx (0-299). Map as follows:
    n_w = 5, n_lam = 3, n_seed = 20
    lam_idx = run_idx // (n_w * n_seed)
    remainder = run_idx % (n_w * n_seed)
    w_idx = remainder // n_seed
    seed_idx = remainder % n_seed
    lambda_loss = lambda_list[lam_idx]
    w = w_list[w_idx]
    seed = seeds[seed_idx]
- Output: one CSV row appended (filelock) to
  docs/expB_reduced_figures/expB_reduced_raw_data.csv
  Columns: run_idx, w, lambda_loss, seed, mean_S, profit, lost_sales,
           collapsed, final_value_loss, final_explained_variance
- collapsed := (mean_S < 10) AND (profit <= -127.1)
- Same SB3 PPO setup as expA_interior_w.py — MlpPolicy, n_steps=128,
  batch_size=256, SuperSuit vectorization

Also write docs/slurm/run_expB_reduced.sh:
- job-name: exp_b_reduced
- array: 0-299
- cpus-per-task: 1
- mem: 4G
- time: 01:30:00
- output: logs/expB_%A_%a.out
- python experiments/exp_b_reduced.py --run_idx $SLURM_ARRAY_TASK_ID
```

> **After results arrive, paste this for analysis:**

```
Write experiments/analyze_expB_reduced.py that:

1. Loads docs/expB_reduced_figures/expB_reduced_raw_data.csv
2. For each lambda_loss value in [1, 3, 7]:
   a. Subset to runs with that lambda_loss
   b. Compute collapse fraction per w with Wilson score 95% CI
   c. Fit logistic: collapsed ~ w using statsmodels
   d. Estimate w_crit with bootstrap CI (n=2000 resamples at run level)
   e. Store (lambda_loss, w_crit, ci_lo, ci_hi, b1, p_value)
3. Print a summary table:
   lambda | w_crit | CI_lo | CI_hi | b1 | p
4. Conduct a test of whether CIs overlap:
   If w_crit(λ=7) CI_lo > w_crit(λ=1) CI_hi: print "BEHAVIORAL CLAIM CONFIRMED"
   Else: print "CIs OVERLAP — behavioral claim not established at this resolution"
5. Save Figure 2 to docs/expB_reduced_figures/expB_wcrit_by_lambda.png:
   - Three logistic curves on same axes (λ=1 blue, λ=3 orange, λ=7 red)
   - w on x-axis [0.4, 1.0], P(collapse) on y-axis
   - Vertical dashed lines at each w_crit with CI shading
   - Legend, axis labels, seaborn style
6. Save Figure 2b: w_crit vs lambda (3 points with CI error bars)
7. Save docs/expB_reduced_figures/expB_results.md with full statistics
```

#### Game Tree — Exp B-Reduced

```
ROOT: Exp B-Reduced results arrive
│
├── CIs DO NOT overlap — w_crit(λ=7) > w_crit(λ=1) significantly
│   ├── MEANING: Loss aversion shifts the coordination boundary.
│   │   This is the behavioral claim. Paper is complete.
│   ├── ACTION: Proceed to writing. Figure 2 is the centerpiece.
│   │   Frame as: "Higher loss aversion requires greater centralization
│   │   to achieve coordination — a behavioral tax on the credit
│   │   assignment budget."
│   └── BONUS CHECK: Is the shift monotone? w_crit(λ=1) < w_crit(λ=3) < w_crit(λ=7)?
│       If yes — clean result. If λ=3 is out of order — flag as non-monotone,
│       discuss probability distortion (α,β) as a confound, add as limitation.
│
├── CIs overlap — no significant λ effect
│   ├── MEANING: Behavioral claim is absent at λ ∈ {1,3,7}.
│   ├── SUB-BRANCH A: Effect might exist at higher λ
│   │   Check: is there a monotone trend in w_crit even if CIs overlap?
│   │   If point estimates trend upward (w_crit(7) > w_crit(3) > w_crit(1))
│   │   but CIs overlap — the effect is real but underpowered.
│   │   ACTION: Run a quick λ=15 arm: w∈{0.65, 0.75, 0.85} × 20 seeds
│   │   (60 runs, 1hr on HPC). If λ=15 w_crit separates from λ=1,
│   │   include λ=15 and frame the behavioral effect as emerging at
│   │   extreme loss aversion. This is still publishable.
│   │
│   ├── SUB-BRANCH B: Effect genuinely absent at these λ values
│   │   MEANING: Structural credit assignment dominates; behavioral utility
│   │   curvature at plausible λ levels does not alter the coordination boundary.
│   │   ACTION: Pivot paper framing. This is now a null result paper:
│   │   "We establish the structural phase boundary (K2) and provide the
│   │   first test of whether behavioral loss aversion modulates it —
│   │   finding that structural mechanisms dominate at λ ≤ 7."
│   │   This is still publishable at AAAI. Null results in MARL that
│   │   test behavioral claims are informative.
│   │
│   └── SUB-BRANCH C: Check for bugs first before accepting null
│       Verify: does increasing lambda_loss in the CPT wrapper actually
│       change agent behavior? Quick check — compare mean_S distributions
│       for λ=1 vs λ=7 at w=0.75. If they're identical, the wrapper
│       isn't being applied correctly. Run:
│       "In my exp_b_reduced.py, add a debug assertion before training:
│        assert abs(env.lambda_loss - lambda_loss) < 1e-6, 'CPT params not set'"
│
├── Logistic fit fails to converge for some λ
│   ├── CAUSE: Likely because at the tested w values, one λ shows
│   │   all collapse or all survival — the logistic has no information.
│   ├── ACTION: Extend the w grid for that λ. If λ=7 shows 100% collapse
│   │   at all w∈{0.5,...,0.95}, run an extended arm:
│   │   w∈{0.85, 0.90, 0.95, 0.98, 1.0} × 20 seeds × λ=7 = 100 runs.
│   │   Ask Claude Code: "Add a w-extension arm to exp_b_reduced.py for
│   │   lambda=7, w values [0.85, 0.90, 0.95, 0.98, 1.0], 20 seeds each."
│   └── This is actually a positive signal — it means λ=7 needs very
│       high centralization, which would confirm the behavioral shift.
│
└── Partial data (HPC job fails mid-run)
    ├── ACTION: Check how many rows are in expB_reduced_raw_data.csv.
    │   If ≥ 15/20 seeds per cell completed, the analysis is still valid
    │   (Wilson CI will be slightly wider). Run analysis on partial data.
    ├── If < 10/20 seeds for a cell: resubmit only the missing run_idx values.
    │   Ask Claude Code: "Write a script that reads expB_reduced_raw_data.csv,
    │   finds which (w, lambda_loss, seed) combinations are missing, and
    │   prints the corresponding run_idx values to resubmit as a SLURM array."
    └── Never delete partial results — append-only CSV is robust to reruns.
```

---

### Experiment T — Horizon Surface Completion

**Purpose:** Add the 600k isocline point to formally fit w_crit(T) as a curve.

| Parameter | Value |
|---|---|
| w values | {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8} |
| Seeds | 42–61 (20 per w level) |
| Steps | 600k |
| λ | 1, α=β=0.88 |
| Runs | 7 × 20 = 140 |

**Analysis:** Fit logistic at 600k, extract w_crit(600k). Combine with existing 150k and 300k estimates to fit w_crit(T) — test power law or log decay: `w_crit(T) = a·T^{−b} + c`.

#### Claude Code Instructions — Exp T

> **Paste this to build the runner:**

```
Write experiments/exp_T_horizon.py for my MARL supply chain project.
Branch: experiments/w-lambda-collapse-surface.
Reference: experiments/expA_interior_w.py.

Specs:
- w_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
- seeds = list(range(42, 62))  — 20 seeds per w
- total_timesteps = 600_000
- CPTRewardWrapper: lambda_loss=1.0, alpha=0.88, beta=0.88
- Accept --run_idx (0-139):
    w = w_list[run_idx // 20]
    seed = seeds[run_idx % 20]
- Output (filelock append): docs/expT_figures/expT_raw_data.csv
  Columns: run_idx, w, seed, mean_S, profit, lost_sales, collapsed
- collapsed := (mean_S < 10) AND (profit <= -127.1)

SLURM script docs/slurm/run_expT.sh:
- job-name: exp_T_horizon
- array: 0-139
- cpus-per-task: 1
- mem: 4G
- time: 02:30:00   (600k steps, allow extra time)
- output: logs/expT_%A_%a.out
```

> **After results, paste this for the isocline surface analysis:**

```
Write experiments/analyze_expT.py that:

1. Loads docs/expT_figures/expT_raw_data.csv (600k data)
2. Also loads:
   - docs/expA_figures/expA_raw_data.csv (150k, all w, all seeds)
   - docs/expA300k_figures/expA300k_raw_data.csv (300k data)
   Filter both to lambda_loss==1 and only w values in [0.5, 0.8]
3. For each horizon T in [150k, 300k, 600k]:
   Fit logistic, extract w_crit(T) with bootstrap CI
4. Fit w_crit(T) as a function of T using scipy.optimize.curve_fit:
   Model 1: w_crit = a * T^(-b) + c  (power law decay)
   Model 2: w_crit = a * log(T)^(-1) + c  (log decay)
   Report R² for both. Print which fits better.
5. Save Figure 1 to docs/expT_figures/wcrit_vs_horizon.png:
   - x-axis: training horizon T (log scale)
   - y-axis: w_crit estimate
   - 3 data points with CI error bars
   - Best-fit curve overlaid
   - Title: "Coordination Boundary Decreases with Training Horizon"
6. Save Figure 1b: 2D heatmap of P(collapse) in (w, T) space
   using all three horizons' data
7. Save docs/expT_figures/expT_results.md
```

#### Game Tree — Exp T

```
ROOT: Exp T results arrive
│
├── w_crit(600k) is well below w_crit(300k) and fits a clean decay curve
│   ├── MEANING: The isocline is a smooth function of T. The metastability
│   │   story is complete. The (w, T) heatmap is Figure 1.
│   └── ACTION: Fit the curve, report R². If power law fits better than log,
│       note this in the paper (implies the escape rate has a power-law
│       relationship with centralization). This is a bonus theoretical hook.
│
├── w_crit(600k) is similar to w_crit(300k) — curve is flattening
│   ├── MEANING: The isocline may be approaching an asymptote.
│   │   There might be a minimum centralization below which agents
│   │   NEVER coordinate regardless of training time.
│   ├── ACTION (strong framing): This is actually a stronger result.
│   │   Fit w_crit(T) = a·exp(-bT) + c_∞ where c_∞ is the asymptotic
│   │   minimum. If c_∞ > 0 (statistically), you have evidence of a
│   │   structural floor — below w = c_∞, coordination is impossible
│   │   regardless of training. This is a much cleaner critical point claim.
│   └── ACTION: Estimate c_∞ with confidence interval. If CI_lo > 0,
│       report as "irreducible coordination floor." Flag to advisor.
│
├── w_crit(600k) is HIGHER than w_crit(300k) — isocline went up
│   ├── MEANING: More training made things worse at the margin. This
│   │   would be a very strange result. Most likely cause: the 600k runs
│   │   at w=0.65–0.75 have more collapse than 300k runs because some
│   │   recovered seeds at 300k are re-collapsing at 600k (instability).
│   ├── ACTION: First, re-check the collapse definition for 600k runs —
│   │   are you using mean_S over the last 10k steps or the full run?
│   │   Make sure it's the last 10k (converged state). If so, this suggests
│   │   oscillatory behavior — agents coordinate, then collapse again.
│   │   This is related to the "edge of stability" phenomenon in optimization.
│   │   If confirmed: report it as a new finding — "Coordination in the
│   │   transition zone is non-monotone in training horizon, suggesting
│   │   metastability is oscillatory rather than absorbing."
│   └── This is publishable but changes the story significantly. Don't
│       panic — but do flag to advisor before reframing.
│
└── Logistic fit fails at 600k [all collapse or all survive at every w]
    ├── If all survive at 600k: w_crit has passed below 0.5 for all λ.
    │   Add lower w values: run w∈{0.1, 0.2, 0.3, 0.4} × 10 seeds × 600k.
    │   This is 40 runs (~4hr). Ask Claude Code to extend exp_T_horizon.py.
    └── If all collapse at 600k: Something is wrong with the 600k runs.
        Check n_steps — 600k with n_steps=128 should be fine. Verify
        the environment is not hitting a time limit issue. Run 1 seed
        manually and inspect the training curve.
```

---

## 8. The 2-Week Execution Plan

### Days 1–2: M3 Pro — Start Writing Immediately

Do not wait for HPC. These tasks require no new data.

- [ ] **Threshold robustness check** (2 hours). Ask Claude Code:

```
Load docs/expA300k_figures/expA300k_raw_data.csv.
The current collapse definition is: mean_S < 10 AND profit <= -127.1
Run the logistic regression and w_crit bootstrap for 9 variants:
  mean_S thresholds: [8, 10, 12]  × profit thresholds: [-102.5, -127.1, -152.5]
  (±20% on each independently and both together)
For each variant, report w_crit and 95% CI.
Save a table to docs/robustness/threshold_robustness.md.
If all w_crit estimates are within 0.03 of the baseline, write
"Definition is robust to ±20% threshold perturbation."
```

- [ ] **Figure 1 skeleton** — (w, T) heatmap from existing data. Ask Claude Code:

```
Using docs/expA_figures/expA_raw_data.csv (150k) and
docs/expA300k_figures/expA300k_raw_data.csv (300k),
create a heatmap of P(collapse) in (w, T) space.
- x-axis: w values present in both datasets (intersection)
- y-axis: T ∈ {150k, 300k}
- Color: collapse fraction (0=white, 1=dark red)
- Add contour line at P=0.5 (this is the isocline)
- Mark w_crit estimates as points on the isocline with CI bars
Save to docs/figures/fig1_heatmap_draft.png
This is a draft — we will add the 600k row after Exp T completes.
```

- [ ] Write Introduction and Background sections of the paper. These are fixed regardless of new results.

- [ ] Write Methods section (Section 3) up to but NOT including results. Include the pre-registration narrative.

### Days 3–4: HPC Onboarding

- [ ] Get VPN + SSH working. Test: `ssh username@hpc.ashoka.edu.in`
- [ ] Clone repo on HPC: `git clone <repo_url> && git checkout experiments/w-lambda-collapse-surface`
- [ ] Set up virtualenv: `python -m venv ~/envs/marl && source ~/envs/marl/bin/activate && pip install -r requirements.txt`
- [ ] Run sanity check — 3 runs of each experiment manually before submitting arrays:

```
python experiments/exp_null_model.py --run_idx 0
python experiments/exp_b_reduced.py --run_idx 0
python experiments/exp_T_horizon.py --run_idx 0
```

Check: CSV files created, no errors, run time < 35 min.

- [ ] If sanity checks pass: `sbatch docs/slurm/run_expN.sh && sbatch docs/slurm/run_expB_reduced.sh && sbatch docs/slurm/run_expT.sh`
- [ ] Monitor: `squeue -u <username>` every 2 hours.

> **If HPC access is delayed past Day 4:** Run Exp N on M3 Pro locally with 5 seeds per w level instead of 20. This gives enough signal to pre-check the null result while waiting for full HPC runs. Ask Claude Code: "Modify exp_null_model.py to run seeds [42,44,46,48,50] only (5 seeds), output to a separate file docs/expN_figures/expN_pilot_raw_data.csv."

### Days 5–6: Analysis

All three experiments should be done. Run all analysis scripts. Key decision:

**After seeing Exp B-Reduced results → pick your paper version:**

| Exp B-Reduced outcome | Paper version | Lead claim |
|---|---|---|
| CIs separate (λ shifts isocline) | Behavioral + Structural | "Loss aversion shifts the coordination boundary" |
| CIs overlap, monotone trend | Structural + weak behavioral signal | "First phase diagram; behavioral effect emerges at high λ" |
| CIs overlap, flat trend | Structural only | "First phase diagram; behavioral null at λ≤7" |

### Days 7–10: Write Results and Discussion

- [ ] Section 4 (Exp A + T): Structural phase transition + metastability surface
- [ ] Section 5 (Exp B-Reduced + N): Behavioral modulation + null model decomposition  
- [ ] Section 6: Discussion — connect to Weng & Lee 2026, connect to Mengesha 2026, practical implications

### Days 11–12: Related Work + Abstract

- [ ] Related work using kill-condition structure (§3 of this document) as skeleton
- [ ] Abstract: **lead with the phase diagram, not with prospect theory**
- [ ] One crisp paragraph distinguishing from Lalmohammed (ICML 2025)

### Days 13–14: Polish + Submit

- [ ] Advisor review pass
- [ ] arXiv LaTeX formatting (use AAAI 2027 template — submit the same version)
- [ ] Submit to arXiv

---

## 9. HPC Setup (Ashoka)

**Cluster specs:**
- 7x CPU nodes: 2× Xeon Gold 6230R (52 cores/node), 128GB RAM
- 2x GPU nodes: NVIDIA V100 32GB + same CPUs (not needed — PPO is CPU-bound)
- Scheduler: SLURM

**Compute estimate:** PPO @300k steps ≈ 15–30 min/run on 1 core. 52 parallel cores per node. 300-run job = ~3–5 hours. All 3 experiments fit on one CPU node running simultaneously; use 2 nodes if you want them all done in 3 hours.

**Standard SLURM array template:**

```bash
#!/bin/bash
#SBATCH --job-name=exp_b_reduced
#SBATCH --array=0-299
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/expB_%A_%a.out
#SBATCH --error=logs/expB_%A_%a.err

source ~/envs/marl/bin/activate
export OMP_NUM_THREADS=1  # prevent SB3 from spawning extra threads

python experiments/exp_b_reduced.py --run_idx $SLURM_ARRAY_TASK_ID
```

**Checking for failed jobs:**

```bash
# Find which array indices failed (exit code != 0)
sacct -j <JOBID> --format=JobID,ExitCode,State | grep FAILED

# Resubmit only failed indices, e.g. if 5,17,203 failed:
sbatch --array=5,17,203 docs/slurm/run_expB_reduced.sh
```

**HPC game tree:**

```
ROOT: HPC job submitted
│
├── Jobs run normally → wait, check squeue periodically
│
├── Jobs stuck in PENDING for > 6 hours
│   ├── Check queue: sinfo — is the cpu partition full?
│   ├── ACTION: Add --partition=gpu to run on GPU nodes (they have same CPUs)
│   │   You won't use the GPU but you'll get access sooner.
│   └── If still stuck: contact Debargha Ganguly (debargha.ganguly@ashoka.edu.in)
│
├── Jobs running but hitting TIMEOUT
│   ├── Check wall time of failed jobs: sacct -j <JOBID> --format=Elapsed
│   ├── If runs are taking > 90 min (300k steps): increase --time to 02:30:00
│   │   and resubmit failed indices only.
│   └── Alternative: reduce timesteps to 250k for Exp B-Reduced —
│       300k vs 250k doesn't change the collapsed/not-collapsed label for
│       most seeds, only affects the transition zone (w≈0.7).
│
├── OOM (out of memory) errors
│   ├── 4G should be sufficient. If OOM: increase to --mem=8G.
│   └── Also check if SuperSuit vectorization is spawning multiple environments
│       simultaneously — add n_envs=1 if memory is tight.
│
└── CSV file gets corrupted (partial write from two processes)
    ├── This is why we use filelock. If corrupted: delete the CSV,
    │   identify which run_idx values need to be rerun, resubmit.
    └── Ask Claude Code: "Check expB_reduced_raw_data.csv for duplicate
        (w, lambda_loss, seed) rows and report which run_idx to resubmit."
```

---

## 10. The Paper — Narrative and Structure

### Candidate Title

*"The Centralization Threshold: A Prospect-Theoretic Phase Boundary Analysis of Coordination Collapse in Cooperative MARL"*

Alternative if behavioral claim confirmed: *"Loss Aversion Shifts the Coordination Tipping Point: A Phase Diagram of Behavioral Collapse in Multi-Agent Supply Chains"*

### Claim Hierarchy (Ordered by Defensibility)

**Claim 1 — Structural (high confidence, data exists):**  
Cooperative MARL in a multi-echelon supply chain exhibits a sharp logistic phase transition in coordination success as a function of reward centralization w. The critical boundary w_crit is horizon-dependent, constituting a metastable coordination attractor. This is the first quantitative characterization of this boundary as a statistical-mechanics object.

**Claim 2 — Behavioral (pending Exp B-Reduced):**  
Prospect-theoretic loss aversion λ shifts the phase isocline upward. The behavioral excess w_crit(λ) − w_crit(null) increases monotonically with λ, isolating utility curvature from credit assignment structure.

**Claim 3 — Mechanistic (pending Null model):**  
Decomposition via the null model distinguishes structural collapse from behavioral excess — the first causal attribution of a MARL coordination failure to agent utility curvature.

### Paper Structure

```
Abstract
1. Introduction
   - Hook: MARL coordination fails in many practical settings; why?
   - Gap: credit assignment literature identifies the mechanism but not the boundary
   - Contribution: first quantitative phase diagram; λ modulation; metastability result
2. Background
   2.1 MARL credit assignment and the CTDE paradigm
   2.2 Prospect theory and CPT reward shaping in MARL
   2.3 Phase transitions in multi-agent systems
3. Problem Setting
   3.1 Environment (NetworkManagement-v1)
   3.2 CPT reward wrapper
   3.3 Collapse definition and pre-registration
4. Structural Phase Transition (Exp A + Exp T)
   4.1 The logistic transition at 150k and 300k
   4.2 The (w, T) metastability surface
   4.3 Basin robustness (Exp C)
5. Behavioral Modulation (Exp B-Reduced + Exp N)
   5.1 The λ×w surface
   5.2 Null model decomposition: structural vs behavioral excess
   5.3 Critic mechanism diagnostics
6. Discussion
   6.1 Implications for MARL system design
   6.2 Connection to EGT and ABM literature
   6.3 Limitations
7. Conclusion and Future Work
```

#### Claude Code Instructions — Paper Writing

> **For writing Section 4 (paste after analysis scripts have run):**

```
I'm writing Section 4 of my MARL paper. Here is the data:

Exp A @150k logistic: b1=-15.78, p=5e-9, w_crit=0.822, CI=[0.785, 0.858], Δw=0.278
Exp A @300k logistic: b1=-22.05, p=3.2e-7, w_crit=0.666, CI=[0.635, 0.694], Δw=0.199
Exp T @600k logistic: [INSERT RESULTS]
Exp C: corr(S_init, converged_S) within w: -0.11 (w=0.5), -0.38 (w=0.85), -0.36 (w=1.0)

Write Section 4 (Structural Phase Transition) as academic prose (~600 words).
Do NOT use bullet points. Write it as a journal-quality results section.
Structure:
- 4.1: Describe the logistic transition, cite both w_crit estimates with CIs,
  note the steepening slope (Δw decreasing), interpret as sharpening transition
- 4.2: Describe the metastability interpretation — w_crit(T) as a curve,
  not a fixed point. Reference statistical physics (finite-time scaling).
  Describe what the (w, T) surface means operationally.
- 4.3: Basin control — one paragraph on Exp C showing initialization
  doesn't drive collapse.
Be precise with numbers. Do not over-interpret. Stay close to the data.
```

> **For writing the Related Work section:**

```
Write the Related Work section of my MARL paper (~500 words, 4 paragraphs).
Our paper's contribution: first quantitative phase diagram of cooperative MARL
coordination collapse as a function of reward centralization w, with 
prospect-theoretic loss aversion λ as a modulating factor.

Organize into 4 paragraphs:

Paragraph 1 — MARL credit assignment and CTDE collapse:
Cite: Peng (2023) CURO [relative overgeneralization], Liu (2023) lazy agents,
Weng & Lee (2026) exploration breaks cooperation.
Key point: these establish the phenomenon but do not quantify a critical boundary.

Paragraph 2 — PT and behavioral economics in MARL:
Cite: Lalmohammed et al. (ICML 2025) CPT-MADDPG, Anon (2024) risk-sensitive NAMGs,
Qiu (2021) RMIX, Xie (2023) RiskQ.
CRITICAL: Distinguish Lalmohammed architecturally — they modify actor/critic updates,
we use PT as reward shaping and measure its effect on the structural boundary.
These are complementary, not redundant.

Paragraph 3 — Phase transitions in multi-agent systems:
Cite: Mengesha (2026) EGT zero-sum norms, Granovetter threshold models,
Zhang/Li (2026) phase transitions in MARL credit assignment.
Key point: behavioral minorities shifting tipping points is established in EGT/ABM;
we are the first to measure this in deep MARL with neural policies and continuous
state spaces, with a pre-registered quantitative methodology.

Paragraph 4 — Supply chain RL substrate:
Cite: Oroojlooyjadid (2022) Beer Game DQN.
Key point: establishes substrate credibility; collapse in decentralized
supply chain MARL is a known phenomenon.
```

---

## 11. The Figure That Makes or Breaks This Paper

**Figure 2:** Three logistic curves — λ=1, λ=3, λ=7 — plotted on the same axes as P(collapse) vs. w at T=300k, with the fitted w_crit point marked on each curve with bootstrap CI error bars.

If those three w_crit estimates have non-overlapping CIs: behavioral claim confirmed, paper is complete.

If the curves overlap: the figure becomes a null result figure showing λ=1 PT vs. null model (no PT), isolating the structural baseline.

#### Game Tree — Final Paper Decision

```
ROOT: All experiments complete, all figures generated
│
├── Exp B CIs separate AND Exp N shows behavioral excess
│   PAPER: Full behavioral + structural claim.
│   TITLE: "Loss Aversion Shifts the Coordination Tipping Point..."
│   SUBMIT TO: AAAI 2027 (strong submission)
│
├── Exp B CIs separate BUT Exp N shows null ≈ PT at λ=1
│   MEANING: λ=1 PT acts like no-PT, but higher λ shifts the boundary.
│   The "behavioral excess" is visible only above λ=1.
│   PAPER: Same full claim, but reframe behavioral excess calculation.
│   SUBMIT TO: AAAI 2027 (strong submission, same tier)
│
├── Exp B CIs overlap BUT monotone trend AND Exp N shows structural excess
│   PAPER: Structural claim + weak behavioral signal.
│   Add λ=15 extension arm (60 additional runs, fast).
│   If λ=15 separates from λ=1: include and reframe as "behavioral effect
│   emerges at extreme loss aversion."
│   SUBMIT TO: AAAI 2027 (acceptable, slightly weaker)
│
├── Exp B CIs overlap (genuinely flat) AND Exp N shows meaningful excess
│   PAPER: Structural null result. The phase diagram IS the contribution.
│   "Behavioral null: loss aversion at λ≤7 does not shift the coordination
│   boundary, but we characterize the structural boundary for the first time."
│   SUBMIT TO: AAAI 2027 (acceptable; null results are informative)
│   RENAME: "The Centralization Threshold: Phase Boundary Analysis..." (drop behavioral)
│
└── Both experiments show noise / fail to replicate Exp A transition
    PROBABILITY: < 5%.
    ACTION: Do not submit. Diagnose environment. This means there is a
    reproducibility issue with the existing result. Start with:
    "Re-run 20 seeds of the exact Exp A @300k script. Does w_crit=0.666
    replicate?" If not: the result is a seed artifact and the project
    needs to be redesigned.
```

---

## 12. Reviewer Attacks and Pre-emptions

### Attack 1: "w_crit slides — there is no critical point, just a finite-time effect"
**Pre-emption (Section 4.2):** We adopt the metastability framing explicitly. We do not claim an asymptotic critical point. We characterize the horizon-dependent isocline w_crit(T) and show it follows a monotone decreasing curve. For finite operational horizons — which describe real supply chains — the isocline is the relevant quantity.

### Attack 2: "The λ effect is weak / not significant"
**Pre-emption:** If Exp B-Reduced shows significant separation → show figure. If not → report the null honestly: "At λ ∈ {1,3,7}, behavioral excess is not statistically detectable, suggesting structural credit assignment dominates." Do not hide a null. Null results about PT in MARL are informative and publishable.

### Attack 3: "K1 triggered — Lalmohammed already did CPT in cooperative MARL"
**Pre-emption (related work, 1 paragraph):** "Lalmohammed et al. (ICML 2025) integrate CPT into actor and critic update mechanics of MADDPG. Our work is architecturally distinct — CPT acts as a reward transform — and asks a different question: not whether CPT agents can coordinate, but at what centralization level the coordination boundary lies and how loss aversion modulates it. These are complementary."

### Attack 4: "Non-standard environment — why generalize?"
**Pre-emption (Section 1):** Two claims. First, specific w_crit values are substrate-specific and we do not claim numerical transfer. Second, the methodology — logistic phase boundary in (w, T, λ) space with pre-registered collapse definitions and bootstrap CIs — is a general measurement framework. The supply chain is its credibility-validated instantiation.

### Attack 5: "The collapse definition is ad-hoc and post-hoc"
**Pre-emption:** Pre-registered in `docs/collapse_definition.md` before analysis. Show commit hash `03c3fc6`. Threshold robustness in appendix: ±20% perturbation does not change w_crit beyond CI width.

### Attack 6 (most dangerous): "Is the action-space pathology (Gaussian PPO vs zero-inflated optimal) the real driver?"
**Answer:** The base-stock wrapper (Phase 3) was specifically designed to fix this pathology. Phase 1 vs Phase 4 comparison shows agents learn non-zero S values with the wrapper. The persistent collapse in Phase 4/Exp A is a distinct phenomenon from the Phase 1 order≈0 pathology. The null model (Exp N) further isolates behavioral from structural collapse.

---

## 13. AAAI 2027 Assessment

**Deadline:** Abstract mid-August 2026. Paper ~1 week later. ~10 weeks from now.

### If Both Claims 1 and 2 Hold
Strong submission. Phase diagram + behavioral modulation + pre-registration + bootstrap CIs + basin control + null decomposition = rigorous empirical package.  
**Realistic accept probability: ~40–55%**

### If Only Claim 1 Holds (λ null result)
Acceptable. Reframe around metastability and "cost of decentralization." Lead with phase diagram.  
**Realistic accept probability: ~25–35%**

### What Would Make This NeurIPS-Worthy (Future Work)
1. Replication on a second environment (MPE or SMAC)
2. Theoretical grounding: mean-field argument for why the isocline is logistic and decreases with T
3. An adaptive centralization algorithm derived from the phase diagram

---

## 14. Future Work Directions (Post-AAAI)

### Direction A: Adaptive Centralization Scheduling
Design an adaptive w schedule that tracks the isocline: start high (above the boundary), decrease as T increases. Experiment: fixed w vs. isocline-following w on coordination efficiency.

### Direction B: The Phase Diagram as a Design Tool
Before deploying any cooperative MARL system, run a small pilot sweep to estimate the (w, T) phase diagram. Use the isocline to set minimum required centralization for the target operational horizon.

### Direction C: PT as Optimizer (Not Reward Wrapper)
Use the PT value function to reweight the empirical distribution of training losses within the mini-batch — a gradient reweighting scheme rather than reward transform. Different from KTO (LLM alignment) and Lalmohammed (actor/critic updates). ICML-worthy if the theory works.

### Direction D: Theoretical Phase Boundary from Mean-Field
Derive the logistic transition shape analytically. Under mean-field approximation, derive w_crit as a function of λ, α, β, and demand distribution parameters. Turns the empirical results into validation of a theoretical prediction. Journal paper territory.

### Direction E: Heterogeneous-λ Minority (Exp D, deferred)
Run `phase6_behavioral_claim.py` properly: num_averse ∈ {0,...,6}, λ_averse=7, λ_neutral=1, w=1.0, 10 seeds. Does a minority of loss-averse agents drag the system below its own w_crit? If yes — a robustness test of the isocline under heterogeneous populations.

---

## Appendix: Session Log

**Research director analysis date:** 2026-06-04

**Key decisions made:**
1. Reframe w_crit sliding as "metastability" — adopt explicitly in the paper narrative
2. Reduce Exp B from 900 runs to 300 (targeted 5×3×20 design) to fit 2-week timeline
3. Add Experiment N (null model) as highest priority — distinguishes behavioral from structural collapse
4. Reject L3 moonshot — K3 triggered by EGT literature
5. Accept K1 as triggered but argue architectural distinction vs. Lalmohammed
6. Paper spine is K2: first quantitative phase diagram of cooperative MARL collapse
7. arXiv in 2 weeks is achievable: HPC runs ≈ 1 day, writing ≈ 9 days

**Compute allocation:**
- M3 Pro: existing data analysis, threshold robustness, paper writing
- Ashoka HPC CPU queue: Exp N (180 runs), Exp B-Reduced (300 runs), Exp T (140 runs) — all simultaneous

**Sources:**
- [KTO: Model Alignment as Prospect Theoretic Optimization (ICML 2024)](https://arxiv.org/abs/2402.01306)
- [Policy Gradients for CPT in RL (2024)](https://arxiv.org/abs/2410.02605)
- [Prospect Theoretic MARL — CPT-MADDPG (ICML 2025)](https://openreview.net/pdf/4a85065219c0b5124b9f8331410b77459b9d4573.pdf)
- [Cognitive Biases and Zero-Sum Norms (Mengesha 2026)](https://arxiv.org/html/2511.16453v1)
- [Risk-Sensitive MARL in NAMGs (2024)](https://arxiv.org/abs/2402.05906)
- [On Progressive Sharpening, Flat Minima and Generalisation](https://arxiv.org/pdf/2305.14683)
- [A Deep Q-Network for the Beer Game (M&SOM 2022)](https://pubsonline.informs.org/doi/10.1287/msom.2020.0939)