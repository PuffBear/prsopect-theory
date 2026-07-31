# Experiment A — robustness diagnostics (post-hoc, review-driven)

Three checks raised in review of the Exp A result (`w_crit = 0.822`). None alter the
frozen analysis; they probe whether `0.822` is a system property or a fit artifact.

## 1. Critic diagnostic — are the collapses learning failures? (DONE, zero compute)

Source: `final_value_loss`, `final_explained_variance` columns of
`expA_raw_data.csv` (logged per run).

| group | n | value_loss (median) | explained_var (median) | mean_S (median) | profit (median) |
|---|---|---|---|---|---|
| collapsed @ w=1.0 | 2 | 5363 | 0.092 | 1.6 | −127.7 |
| recovered @ w=1.0 | 18 | 6218 | 0.086 | 95.0 | 197.9 |
| collapsed @ w≤0.6 | 140 | 1761 | 0.165 | 0.0 | −128.1 |
| recovered @ w≥0.7 | 46 | 4796 | 0.136 | 31.9 | 92.6 |

At the transition (w ∈ {0.7, 0.8, 0.9}): collapsed runs have **lower** value loss
(2448 vs 4340) and **higher** explained variance (0.275 vs 0.168) than recovered.

**Conclusion:** collapse is **not** a critic failure. A blown-up critic would show
*high* value loss / *low* explained variance; the collapsed runs show the opposite.
A zero-order policy yields simple, low-variance dynamics that the critic fits easily
— collapse is the **easy, well-fit, low-variance equilibrium/basin attractor**, not
a learning breakdown. This rules out the "PPO critic fails at low w" mechanism and
elevates Experiment C (basin control) to a primary mechanism experiment.

## 2. Horizon-convergence check — is w_crit a 150k-step artifact? (RUNNING)

w = 0.9, same 20 seeds (42–61), 400k steps, paired against Exp A's 150k w=0.9 column
(4/20 = 0.20 collapsed). Output: `expA_horizon_w0.9_400k.csv`.
- If collapse fraction ≈ 0.20 at 400k → transition reflects equilibria; `w_crit` is
  a system property.
- If collapse fraction rises materially → collapse is convergence-rate-dependent and
  `w_crit` must be reported as "critical centralization at 150k steps."

**RESULT (done):** at w=0.9, collapse went **4/20 (150k) → 0/20 (400k)** — paired,
same seeds. All four 150k-collapsed seeds (48, 54, 56, 60) **recovered** by 400k
(median S 0→158, median profit 62→418). Collapse at the transition is therefore
**not a permanent basin; it is slow/metastable convergence** that resolves with more
training. Reconciles with the critic diagnostic: a zero-order policy is trivially
predictable (low value loss) *while the policy is still slowly escaping it*.

**Consequence for the claim:** `w_crit = 0.822` is the critical centralization **at a
150k-step budget**, not an unconditional system constant — the transition moves left
with more training. This is now a required caveat, and it introduces a **known
confound for Experiment B**: if loss aversion `λ` changes convergence *rate*, a tilt
in the 150k boundary `w_crit(λ)` could reflect λ-dependent learning speed rather than
λ-dependent equilibrium structure. B is still validly interpreted as a *fixed-budget*
comparison (all cells share 150k), but the convergence-rate alternative must be
pre-stated and a horizon-robustness check run on B's boundary cells afterward (as
done here for A).

## 3. Censoring at w=1 (mitigation in place)

Residual 10% collapse persists at full centralization (w=1), so the logistic right
tail is partly extrapolated past the observable support and the 0.278 transition
width may be modestly compressed. Mitigation: the Exp B grid is extended with
straddling levels {0.85, 0.95} per pre-registration amendment A1.
