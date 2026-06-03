# Exp 0b — Horizon Lock + Metastability-Origin Spec (pre-registered)

*Committed BEFORE running. The interpretation rules in §4 are frozen at commit time;
not edited after seeing results. Amendments require a dated v2 with the old version
left intact.*

**Branch:** `experiments/w-lambda-collapse-surface`
**Gates:** Experiment B does not launch until this returns and a §4 branch fires.

---

## 1. Why this experiment exists

Two entangled unknowns after the 300k re-estimation:

- **(Q1 — horizon)** Is `w_crit ≈ 0.67` converged, or sliding left past 300k? The
  "300k = plateau" claim rested on 8 seeds at one cell and was undercut by the
  metastability finding.
- **(Q2 — origin of metastability)** Same seed + w + 300k flips collapse under a
  trivial training-procedure perturbation (segmented vs single `learn()`). Two
  readings with OPPOSITE spatial predictions:
  - **Critical-point:** perturbation-sensitivity peaks near `w_crit`, absent in the
    bulk → basin-boundary / critical-slowing-down signature (supports a genuine
    critical point).
  - **Undertraining:** borderline runs just haven't converged → horizon confound
    still alive, `w_crit` budget-dependent.

One experiment discriminates them (different cells flip). Not run to confirm
`w_crit ≈ 0.67`: a leftward 600k slide, or sensitivity at deep cells, each falsifies
a current claim and is reported with equal prominence.

## 2. Design

Same substrate (scaled [0,500]), λ=1, same PPO config, frozen collapse label.
NOTE on the label: for the SCALED substrate the single-reset mean_S mapping
`0.5*(a+1)*500` is the CORRECT mapping (the Exp C bug was applying it to the OFFSET
substrate). converged_S (last-10k logged S) is recorded alongside as a robustness
cross-check; primary label kept identical to the 300k re-estimation for comparability.

**Piece 1 — Horizon lock (Q1):** transition cells to 600k, 20 seeds.
| w | seeds | horizon | compare vs 300k |
|---|---|---|---|
| 0.6 | 42–61 | 600k | 0.80 |
| 0.7 | 42–61 | 600k | 0.40 |

**Piece 2 — Perturbation-origin controls (Q2):** segmented-vs-single at transition
AND deep cells, shared seeds 42–49, both arms run in-script for a controlled compare.
| w | regime | seeds | arms |
|---|---|---|---|
| 0.3 | deep-collapse control | 42–49 | segmented(150k+150k) vs single(300k) |
| 0.6 | transition | 42–49 | segmented vs single |
| 0.7 | transition | 42–49 | segmented vs single |
| 0.8 | deep-recovery control | 42–49 | segmented vs single |

Parallelism kept low (≤5-way) + thread-limited workers (OMP/MKL=1) to avoid the
load-46 thrash that slowed Exp A.

**Row metrics:** `w, seed, arm, horizon, collapsed, converged_S, profit,
final_value_loss, explained_variance` (+ `flipped_vs_pair` computed in analysis).

## 3. Quantities computed (mechanical)

1. 600k collapse fractions @ w=0.6,0.7 with Wilson CIs; Δ vs 300k.
2. Perturbation-flip rate per cell (fraction of 8 seeds whose label differs between
   arms) at w ∈ {0.3,0.6,0.7,0.8}, Wilson CI per cell.
3. Flip-rate spatial profile: transition (0.6,0.7) vs deep (0.3,0.8), difference with
   bootstrap CI.
4. value_loss / explained_variance of flipped vs non-flipped runs.

## 4. FROZEN interpretation rules

### Q1 — horizon (quantity 1)
- **STABLE** if both 600k fractions are within their 300k Wilson CIs (0.6→[~0.58,0.92],
  0.7→[~0.22,0.61]) AND neither drops >0.15 absolute.
- **SLIDING** if either drops >0.15 absolute at 600k.
- **AMBIGUOUS** otherwise → treat as SLIDING for decisions (conservative).

### Q2 — metastability origin (quantity 3, spatial profile)
- **CRITICAL-SIGNATURE** if transition flip-rate (0.6/0.7) exceeds deep flip-rate
  (0.3/0.8) by ≥0.25 absolute AND deep-cell flip-rate ≤0.15.
- **GENERIC-NONDETERMINISM** if deep-cell flip-rate >0.15.
- **INCONCLUSIVE** if transition high but deep borderline (0.15–0.25) → more seeds.

### Combined branch → action
| Q1 | Q2 | Action |
|---|---|---|
| STABLE | CRITICAL-SIGNATURE | Lock `w_crit≈0.67`. Promote metastability to a result (boundary sensitivity peaks at criticality). Run **B @300k** straddling 0.67. |
| STABLE | GENERIC-NONDET | Lock `w_crit≈0.67`. Metastability = measurement caveat. Run **B @300k**. |
| SLIDING | CRITICAL-SIGNATURE | Do NOT lock 0.67. Extend horizon (Piece 1 @1.2M) before B. Criticality result stands independent of location. |
| SLIDING | GENERIC-NONDET | Worst case. **Do not run B.** Escalate: (a) asymptotic-horizon collapse def, or (b) pivot to **Exp D** (relative, matched-horizon). |
| AMBIG/INCONCLUSIVE | any | +12 seeds at ambiguous cells (→32), re-evaluate SAME rules. No B on ambiguous data. |

**B runs only in the two STABLE rows.** Every other branch gates B behind more work.

## 5. Written down regardless of outcome
- `w_crit(horizon)` as a function (0.82@150k, 0.67@300k, 600k transition-cell values),
  with only the asymptotic claim §4-Q1 licenses.
- The flip spatial profile, null or positive, reported equally.
- Seeds 43,47,49 tracked individually through both arms (appendix worked example).

## 6. Self-check before accepting
- Read Q1/Q2 from frozen thresholds, not by eyeballing.
- Deep-cell flip-rate genuinely low before claiming CRITICAL-SIGNATURE.
- About to run B? Confirm STABLE row, else B does not launch.
- Written `w_crit` claim matches exactly what Q1 licensed.
