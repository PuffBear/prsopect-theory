# Value-Based Generalization Plan — QMIX / IQL (EPyMARL)

**Goal of this experiment block:** close the last breadth lever in the AAAI review —
"two policy-gradient learners agree; a value-based method would make the
cross-algorithm claim robust to the *learning paradigm*, not just the critic."

**The claim we are buying.** Completing a 2×2 over **credit-assignment × learning
paradigm**. If `w_crit` rises monotonically with λ in all four cells, the
loss-aversion boundary shift is invariant to both axes of algorithm design.

|                      | Independent            | Centralized                       |
|----------------------|------------------------|-----------------------------------|
| **Policy-gradient**  | PPO ✅ (have)          | MAPPO ✅ (have)                   |
| **Value-based**      | **IQL** (this plan)    | **QMIX** (this plan, monotonic mix) |

This directly serves the *mechanism-before-generalization* ordering: the mean-field
SNR account **derives the sign** (a signal-to-noise property of the update, agnostic
to paradigm). The 2×2 is the predict-then-confirm test of that derived sign — which
is how we answer the "calibrated, not predictive" criticism without hiding from it.

---

## 1. Frozen invariants (identical to the PPO/MAPPO pipeline)

Only the **learner** changes. Everything below is held byte-identical so the
comparison is clean:

- **Reward pipeline:** `CPTRewardWrapper` — blend with weight `w` **then** apply the
  CPT value function `v(x)=x^α (x≥0); −λ(−x)^β (x<0)`. Same wrapper, same order.
- **w-grid (supply chain):** `[0.5, 0.65, 0.75, 0.85, 0.95]` (the `exp_b_reduced` /
  `exp_mappo_orgym` grid).
- **w-grid (MPE):** inherit the **MPE-MAPPO recalibrated** grid (null `w_crit≈0.71`,
  boundary has headroom below 1) — *not* the supply grid. Reuse whatever
  `exp_mappo_mpe_highlambda` used so QMIX/IQL-MPE sit on the same axis as MAPPO-MPE.
- **λ sweep:** full `{1,2,3,4,5,6,7}` — matches MAPPO-orgym and MPE-MAPPO so all
  series land on one λ axis in `generalization_wcrit_vs_lambda.png`.
- **Curvature:** `α=β=0.88`. **Null:** `α=β=1`.
- **Seeds:** `range(42,62)` — 20 per cell.
- **Collapse label:** supply `(mean_S<10) & (profit≤−127.1)`; MPE θ=0.164 occupancy
  (`is_collapsed_mpe_occupancy`). Both are env-outcome labels, independent of the
  learner — they transfer untouched.
- **Estimation:** logistic `P(collapse|w)` → **Firth-penalized** fit → **bootstrap
  (2000)** `w_crit` + 95% CI. Same as Table 2.

## 2. Conditions per cell (the full decomposition)

Each `(learner, env)` cell runs three conditions so it carries its **own** baseline
(a new learner's null boundary ≠ PPO's — it must be re-anchored):

| Condition       | α, β        | λ            | Runs (supply)        |
|-----------------|-------------|--------------|----------------------|
| **Null**        | 1, 1        | 1            | 5 w × 20 = **100**   |
| **Curvature**   | 0.88, 0.88  | 1            | (λ=1 point of sweep) |
| **Loss aversion** | 0.88, 0.88 | 2..7        | 5 w × 7 λ × 20 = **700** |

> The curvature condition **is** the λ=1 point of the 0.88 sweep, so the sweep block
> (700) already contains it. Curvature shift = (λ1@0.88) − null. Loss-aversion excess
> = (λ@0.88) − (λ1@0.88).

**Per-cell total = 800 runs** (700 sweep + 100 null). Note: the "~700" we'd been
quoting was the sweep portion only; the +100 null baseline is what re-anchors each
new learner honestly.

## 3. EPyMARL integration

**Algorithms:** `qmix` (monotonic mixing network = centralized value-based) and `iql`
(independent Q, no mixer = independent value-based). EPyMARL ships both; use the
stock `qmix.yaml` / `iql.yaml` configs with sacred CLI overrides.

**Environment adapters.** Reuse the *existing* PettingZoo parallel stacks; only add a
discrete-action shell and an EPyMARL `MultiAgentEnv` adapter:

- **Supply chain (discretization required — `Box` action).** Pipeline, outer→inner:
  `EPyMARL MultiAgentEnv adapter → Discrete(K)-per-agent wrapper (index → continuous
  base-stock S∈[0,500]) → CPTRewardWrapper(blend w, then CPT) → ScaledBaseStockWrapper
  → NetworkManagement-v1`. Each of the 6 stocking-node agents gets its own
  `Discrete(K)`. **K=11** default (S∈{0,50,…,500}); robustness re-run at **K=21** at one
  λ if boundaries look grid-locked.
- **MPE Cooperative Navigation (natively discrete).** Use `simple_spread` with
  `continuous_actions=False` → `Discrete(5)` per agent (3 agents). Wrap with the same
  `CPTRewardWrapper` + the existing MPE occupancy-collapse eval. No custom
  discretization.

**Iso-sample budget (critical for fairness).** Value-based and policy-gradient differ
in sample efficiency, so we fix **environment interactions**, not gradient steps:
- Supply chain: **300k env steps/run** (the PPO/MAPPO decomposition anchor — the 300k
  boundary).
- MPE: **4M env steps/run** (match `exp_mappo_mpe_highlambda`).
All other value-based knobs (lr, replay buffer size, target-update interval, ε-greedy
anneal) = EPyMARL defaults, **frozen across every cell** (do not let ε vary with λ).

> ε-greedy is the value-based analogue of PPO's entropy coefficient. Keeping it fixed
> isolates the loss-aversion effect; it also ties into the SNR mechanism (exploration
> noise raises `w_crit`) — a future ε-sweep would be the value-based counterpart of the
> entropy ablation. Out of scope here; note it.

**Wiring check (mandatory — mirrors the MAPPO mislabel we already got burned by).**
A `--wiring_check` that asserts, with no training:
1. Action space EPyMARL sees is `Discrete(K)` per agent (supply) / `Discrete(5)` (MPE).
2. The reward written to the replay buffer equals `CPT(blend(raw_reward, w))` — log
   `raw / blended / cpt` for one transition and assert the transform.
3. **QMIX mixer receives the global state; IQL does not.** Assert the
   centralized/independent distinction so we never mislabel the two. Must print
   `WIRING CHECK PASS`.

## 4. Tiers, run counts, order

| Tier | Cells added                          | Runs   | Why                                            |
|------|--------------------------------------|--------|------------------------------------------------|
| **1** | QMIX · supply chain                  | 800    | Answers "value-based?" on the **primary** substrate. Minimum viable for the claim. |
| **2** | IQL · supply chain  +  QMIX · MPE    | +1,600 | Completes the 2×2 on supply; adds the cross-env value-based point. |
| **3** | IQL · MPE                            | +800   | Completes the 2×2 in **both** environments.    |
| (opt) | discretized-PPO control · supply     | +800   | Isolates learner-vs-action-space (see §7).     |

**Grand total ≈ 3,200 runs** (4 cells × 800), +800 if the optional control is run.
Run Tier 1 first; the headline figure is publishable the moment QMIX-supply lands.

## 5. Output schema & analysis

Mirror `exp_mappo_orgym` columns so existing analyzers ingest it:
`run_idx, w, lambda_loss, alpha, beta, seed, algo, condition, mean_S|mean_occupancy,
profit|mean_reward, collapsed`.

Per cell: fit Firth logistic per condition → bootstrap `w_crit`+CI → report
**curvature shift** (λ1@0.88 − null) and **loss-aversion excess** (λ@0.88 − λ1@0.88) →
test monotonicity → append `qmix` / `iql` series to
`generalization_wcrit_vs_lambda.png` (now up to **five comparable series** on one λ
grid: PPO, MAPPO, QMIX, IQL on supply + MPE).

## 6. Pre-registration (set before any run finishes)

- **Prediction (from the mechanism):** `w_crit` monotone non-decreasing in λ.
- **Success:** `w_crit(λ=7) > w_crit(λ=1)` with non-overlapping 95% CIs, monotone from
  some λ — in each cell.
- **Falsification:** flat or decreasing `w_crit(λ)`.
- **Scoped-honest in advance:** as with MPE/MAPPO, we expect the **direction** to
  replicate; the **absolute** `w_crit` and the **curvature sign** may be
  learner/substrate-specific and will be reported as such, not buried.
- **Discretization caveat (pre-written limitation):** "The value-based learners
  require a discrete action space; the supply-chain order quantity was binned into K
  levels. This changes the action representation relative to the continuous PPO/MAPPO
  runs, so the value-based cells test whether the loss-aversion *direction* survives a
  paradigm + action-space change jointly. A discretized-action PPO control (App. X)
  isolates the two." (Include the control only if Tier-3-optional is run; otherwise
  this sentence still pre-empts the reviewer.)

## 7. Risks & decisions

- **Discretized-PPO control:** recommended to **skip for now**, but the limitation
  sentence above is pre-written so a reviewer can't surprise you. Promote to a run only
  if a reviewer pushes or if QMIX/IQL boundaries land suspiciously far from PPO.
- **Grid-locking:** if `w_crit` snaps to grid points at high λ, add `w∈{0.80,0.90}` and
  re-fit (cheap) before adding seeds.
- **Saturation/tail:** 20 seeds is fine for direction; if the λ≥5 CIs are wide, scope
  the saturation claim as exploratory (same caveat as the PPO sweep).
- **MPE null headroom:** confirm the QMIX/IQL-MPE null boundary sits below w=1 (the
  recalibration reason); if it pins at 1, raise the step budget before reading the
  sweep.

## 8. PBS skeleton (per cell)

Mirror `run_mappo_orgym.pbs`: two arrays per cell — sweep `0..699`
(`run_idx → (w_idx, lam_idx, seed_idx)`) and null `0..99`
(`run_idx → (w_idx, seed_idx)`, α=β=1, λ=1). `--wiring_check` before submit;
one 20k-step smoke run before the array; `rsync` results to `docs/`; analyze.
