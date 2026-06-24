# `main_aaai.tex` — Story & Structure (current form)

> One-line thesis: **Cooperative-MARL coordination collapses at a sharp,
> measurable boundary `w_crit` in reward centralization; prospect theory moves
> that boundary in two opposite directions; a mean-field signal-to-noise account
> predicts the *signs* in advance; and the loss-aversion direction generalizes
> across environments and algorithms.**

> **Last updated:** 2026-06-23 (Claude Code session). Changes this session:
> (a) the SNR mechanism is **sign-predictive but NOT a quantitative w_crit
> locator** — appendix pre-registered SNR-crossing test negative across two
> sweeps; **per-run prediction empirically ruled out** (early-signature predictor
> AUC=0.51 = chance, 6 observables now tested) — confirms stochastic-escape;
> (b) **Theorems 1–3 added** to the mechanism section (MF criticality / calibration
> gap / **sign invariance** — the *proven* a-priori predictive result), compiling
> clean; (c) **RESUBMISSION REFRAME (post-reject):** demoted "opposite directions"/
> curvature to a **supply-chain-specific** finding; **headline is now loss-aversion
> shift + downstream cascade** (both generalize). New **title**, abstract, intro
> flagship, contributions, conclusion all re-centered. Fixes the R1 weak-reject
> (title sold the claim the generalization section retracts); (d) **high-λ plateau
> (R2) resolved in text**: explicit argument that the plateau at w≈0.85 is
> *behavioral saturation*, not a w=1 ceiling artifact (collapse already 0.00 at
> w=0.95 → headroom to ceiling); (e) 4th-substrate **closed at 3 envs**; cascade
> **supply-chain-only/future work**; LBF **w=0.90 resolver built+gated, not run**.
> (f) **Cascade 2nd-substrate search closed (gym-anm + grid2op tried).** gym-anm
> (power feeder) built fully but has NO w_crit boundary (regulation task,
> local≈global, no tension — like MAgent battle); grid2op scoped ~3 weeks + ~50%
> boundary-gate risk, declined. Cascade stays N=1, but Limitation 8 is **upgraded**
> into a falsifiable, structurally-specified prediction: cascade needs (i) directed
> flow, (ii) local-vs-global tension, (iii) reachable boundary — rare conjunction;
> most benchmarks violate ≥1, itself a finding. Converts the N=1 "ignored promise"
> into a respected, precise claim.

---

## The spine of the argument (read this first)

1. **There is a boundary.** Coordination → collapse is a *sharp phase
   transition* at a critical centralization `w_crit`, not gradual decay. We
   measure it (pre-registered collapse def + bootstrap CIs). *First quantitative
   collapse boundary in deep cooperative MARL.*
2. **The boundary is an instrument.** Once you can measure `w_crit`, you can ask
   what *moves* it.
3. **Prospect theory moves it two ways.** Curvature (probability weighting)
   shifts `w_crit` **left** (−0.10, eases coordination); loss aversion shifts it
   **right** (+0.166 at λ=7, impedes coordination). A full λ=1…7 sweep **rises
   then saturates**, tracked by a one-parameter law (RMSE 0.014).
4. **A mechanism predicts the signs *a priori*.** A mean-field signal-to-noise
   account derives *both* signs before any data (curvature → lower `w_crit`,
   loss aversion → higher) — and the empirical signs match. This is the
   "mechanism-before-generalization" ordering. **Scope (verified this session):**
   the mechanism is *sign-and-form* predictive, **not** a quantitative locator of
   `w_crit`. The pre-registered SNR-crossing test (App) was run on the full
   sweep and came back **negative** — the boundary is reproduced (collapse-frac
   crossing = 0.822) but the measured restoring-drift / SNR crossing does **not**
   land at `w_crit`, and a pre-registered endpoint test falsifies the two-basin
   structure the Langevin coordinate would need. So the paper claims sign + form,
   and the draft already disclaims predicting the numerical `w_crit` value.
5. **The direction generalizes.** Predict-then-confirm: the loss-aversion
   rightward shift replicates across **two further environments** (MPE, LBF) and
   **a second algorithm** (MAPPO). Direction is universal; absolute level and
   curvature sign are substrate-specific.
6. **Boundary conditions, honestly stated.** Two ablations show *where* it
   breaks: coarse action-space discretization erases it (a confirmed mechanism
   prediction, not a failure), and credit assignment matters.
7. **A minority sets the boundary.** A single *downstream* loss-averse node moves
   the system boundary as far as a fully loss-averse population — loss aversion
   cascades upstream. **Scope:** this is intrinsically a *directed-flow* result
   (needs an upstream/downstream axis), so it is demonstrated on the supply chain
   only; cross-substrate generalization is **future work** (no reachable 2nd
   directed-flow MARL env — full search closed this session).

---

## Section-by-section map

| § | Section | What it does | Key result / number |
|---|---------|--------------|---------------------|
| — | **Abstract** | States the boundary + the 2-direction PT decomposition + SNR prediction + generalization | curvature −0.10; loss aversion +0.166; sweep rises-then-saturates |
| 1 | **Introduction** | Frames the 3 research questions | RQ1 existence/location · RQ2 stability vs horizon · RQ3 behavioral decomposition |
| 2 | **Background** | CPT, cooperative MARL, reward centralization `w` | — |
| 3 | **Method** | The 3-way decomposition (null / curvature / loss aversion) — *the key methodological move* | identical env+algo across conditions; only PT params change |
| 4 | **Experimental Setup** | Grids, seeds, collapse label, Firth+bootstrap | supply chain: 1,524 runs, 20 seeds; `tab:experiments` |
| 5 | **Results** | The boundary is real (RQ1), metastable (RQ2), and decomposes (RQ3) | structural null **`w_crit`=0.666** (canonical); λ-sweep `tab:lambda_sweep` |
| 6 | **Mean-Field Mechanism** | Derives `w_crit` as a signal-to-noise crossing; derives both PT signs + the affine loss-aversion law | `fig:meanfield`, `fig:affine_law`; ρ=0.61, RMSE 0.014 |
| 7 | **Generalization** | Predict-then-confirm across environments & algorithms | `fig:generalization` (4 settings) |
| 7.1 | → across environments | MPE Cooperative Navigation (MAPPO) | `w_crit` 0.78→0.89; excess +0.158 [+0.108,+0.212]; **20 seeds, θ=0.144** |
| 7.2 | → third environment | Level-Based Foraging (MAPPO) | null≈0.08, curv +0.18, λ rises to ceiling; affine law does **not** transfer (RMSE 0.15); **960 runs committed** (w=0.90 high-λ resolver built+gated this session but **not yet run** — pulls λ≥5 off the grid ceiling once run) |
| 7.3 | → across algorithms | MAPPO on the supply chain | `w_crit` 0.796→0.889, monotone from λ=2 |
| 8 | **Boundary Conditions: Two Ablations** | Where the boundary breaks (honest scoping) | — |
| 8.1 | → action representation | QMIX/IQL go flat on discrete K=11; **discretized-PPO control also flat** → discretization (not the value-based paradigm) erases it | `tab:ablation_action` truth table; MPE is natively discrete & *retains* it |
| 8.2 | → credit assignment | Independent vs centralized critic | `fig:mpe_floor` |
| 9 | **A Minority Sets the Boundary** | Heterogeneity / network position (supply-chain only; directed-flow result) | one downstream node → **0.853 ≈ 0.851** (full pop.); upstream only 0.726; cascades upstream | `tab:hetero` |
| 10 | **Discussion** | Scope, safety-margin reading, relation to prior work | — |
| 11 | **Conclusion** | — | — |
| App | **Notation · Threshold Robustness · SNR Pre-registered Test · Exploration-Noise Ablation** | supporting robustness | `tab:entropy`, `fig:entropy` — **SNR test status: NEGATIVE** (full 180-run sweep reproduces the boundary but the SNR crossing does not locate `w_crit`; supports sign-and-form, not quantitative prediction — frame as such) |

---

## Figures (10) and Tables (7)

**Figures:** heatmap (w,T metastability) · decomposition (4 logistic curves) ·
mean-field · affine law · **generalization (the headline 4-series `w_crit` vs λ)** ·
MPE λ-curves · MAPPO supply curves · MPE floor · heterogeneity · entropy.

**Tables:** experiments summary · λ-sweep · decomposition · ledger ·
**action-representation ablation (the discretization truth table)** ·
heterogeneity · entropy.

---

## The "three nulls", canonicalized (so reviewers aren't confused)

- **Canonical structural null = `w_crit` 0.666** (risk-neutral, 300k horizon).
- **0.822** = *horizon offset* (same null at 150k — the metastability reference).
- **0.586** = *curvature offset* (−0.080 from canonical; the λ=1 anchor of the
  loss-aversion sweep). It is a **result**, not a third baseline.

---

## What is deliberately NOT in the paper (scope calls)

- **A 4th environment** — RESOLVED this session: **stay at 3** (supply chain, MPE,
  LBF). Four candidates failed for distinct, documented reasons — RWARE
  (sparse-reward exploration wall, robust to every geometry-safe fix incl.
  TWO_STAGE), SMAC (no SC2 binary on arm64, cluster-only), MAgent2 *battle* (no
  incentive tension: trained w=0 ≡ w=1), MAgent2 *adversarial_pursuit*
  (exploration wall). The CNN-MAPPO MAgent machinery is built+gated and parked
  (untracked) for a future tension-having env. 3 environments already answer the
  breadth weak-reject.
- **Generalizing the cascade** — RESOLVED this session: stays **supply-chain-only,
  future work**. The cascade needs a *directed-flow* topology (upstream/downstream);
  no reachable 2nd directed-flow MARL env exists (SUMO-traffic is SMAC-class:
  external binary, not arm64-installable, cluster-only). Full search documented.
- **SNR as a quantitative mechanism** — the pre-registered SNR-crossing test is
  NEGATIVE (above); the mechanism is presented as sign-and-form, not as locating
  `w_crit`. Do **not** tune the analysis band to manufacture a crossing (that path
  was checked and rejected as researcher-DOF / p-hacking).
- **Value-based as a positive result** — QMIX/IQL appear only as the *negative*
  action-representation ablation (they go flat under discretization); they are
  not claimed as a generalization axis.

---

## How the AAAI weak-reject is answered

- **"Single environment dominates"** → MPE + **LBF (new third environment, the
  session's breadth win)** + cross-algorithm MAPPO. `fig:generalization` shows
  all four on one λ axis.
- **"Calibrated, not predictive"** → the mean-field SNR account derives the
  *signs* (and the loss-aversion *functional form*) *before* the sweep; §7 is the
  predict-then-confirm test of those sign predictions. **Be precise:** the claim
  is sign-and-form prediction, NOT numerical `w_crit` prediction — the
  quantitative SNR-crossing test is negative (App), and the draft explicitly
  disclaims predicting the `w_crit` value on any axis. Pitch the mechanism as
  "fixes the signs a priori," not "predicts the boundary."
- **Honesty asks** (monotonicity, nulls, exploratory points) → λ-sweep reported
  as "rises then saturates" (not strictly monotone); single canonical null;
  high-λ LBF points flagged exploratory.
