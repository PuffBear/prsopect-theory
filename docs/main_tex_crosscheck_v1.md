# main.tex Cross-Check v1

*Verification of `latex/main.tex` against the actual experiment CSVs, the codebase,
and `latex/references.tex`. Date: 2026-06-05.*

Every number below was recomputed directly from the raw `docs/exp*/*_raw_data.csv`
files using the committed analysis pipeline (`fit_logistic` + 2000-sample run-level
`bootstrap_wcrit` from `experiments/analyze_expN.py`; `transition_width` from
`analyze_expA.py`). "Paper" = value as written in main.tex. "Actual" = recomputed.

## 0. Verdict

The draft is **highly accurate**: all headline `w_crit` values, CIs, slopes,
p-values, transition widths, collapse fractions, correlations, and the 1,124 run
count reproduce **exactly** from the data. Citations are internally consistent (all
20 `\cite` keys are defined). The methodology (collapse definition, logistic
estimation, three-way decomposition, environment, algorithm) matches the code.

**One material issue** (M1, below) is *not* in the author's `[I1]`–`[I7]` note list
and should be addressed before submission: the metastability heatmap mixes
`α=β=1` (150k, 300k) with `α=β=0.88` (600k), confounding training horizon with the
curvature effect the paper itself measures. The remaining items are minor (a typo,
a 0.001 rounding, a dual-baseline ambiguity, one uncited bib entry).

---

## 1. Number verification (all cross-checked against CSVs)

| Claim in main.tex | Paper | Actual (recomputed) | Status |
|---|---|---|---|
| Exp A @150k w_crit | 0.822 [0.785, 0.858] | 0.822 [0.785, 0.858] | ✅ exact |
| Exp A @150k slope b1 | −15.78 | −15.78 | ✅ |
| Exp A @150k p | 5×10⁻⁹ | 5.0×10⁻⁹ | ✅ |
| Exp A @150k Δw | 0.278 | 0.278 | ✅ |
| Exp A @300k w_crit | 0.666 [0.635, 0.694] | 0.666 [0.635, 0.694] | ✅ exact |
| Exp A @300k slope b1 | −22.05 | −22.05 | ✅ |
| Exp A @300k p | 3.2×10⁻⁷ | 3.2×10⁻⁷ | ✅ |
| Exp A @300k Δw | 0.199 | 0.199 | ✅ |
| Exp N null w_crit | 0.685 [0.657, 0.710] | 0.685 [0.657, 0.710] | ✅ exact |
| Null vs Exp A@300k diff | −0.020 | 0.685−0.666 = −0.019→−0.020 | ✅ |
| Exp B λ=1 w_crit | 0.586 [0.539, 0.624] | 0.586 [0.539, 0.624] | ✅ exact |
| Exp B λ=3 w_crit | 0.774 [0.751, 0.798] | 0.774 [0.751, 0.798] | ✅ exact |
| Exp B λ=7 w_crit | 0.851 [0.845, 0.858] | 0.851 [0.845, 0.858] | ✅ exact |
| λ=7 collapse fractions | 1.00,1.00,1.00,0.55,0.00 | 1.00,1.00,1.00,0.55,0.00 | ✅ |
| λ=7 vs λ=1 CI gap | 0.845−0.624 = 0.221 | 0.221 | ✅ |
| Curvature excess (λ=1) | −0.100 [−0.172, −0.033] | −0.100 [−0.172, −0.033] | ✅ exact |
| Loss-aversion excess λ=3 | +0.089 [+0.041, +0.141] | +0.089 [+0.041, +0.141] | ✅ exact |
| Loss-aversion excess λ=7 | +0.166 [+0.135, +0.202] | +0.166 [+0.135, +0.202] | ✅ exact |
| Exp C corr (w=0.5,0.85,1.0) | −0.11, −0.38, −0.36 | −0.11, −0.38, −0.36 | ✅ exact |
| Heatmap w=0.5: 150k/300k/600k | 1.00 / 0.95 / 0.35 | 1.00 / 0.95 / 0.35 | ✅ (see M1) |
| Heatmap w=0.8: 150k/300k | 0.55 / 0.00 | 0.55 / 0.00 | ✅ |
| Total run count | 1,124 | 220+140+144+180+300+140 = 1124 | ✅ exact |
| Threshold robustness max Δw_crit | 0.016 | 0.015 | ⚠ minor (see m4) |
| λ=7 increment "0.851 − 0.585" | +0.266 | 0.851−0.586 = 0.265 | ⚠ typo (see m3) |
| "18.5 pp" extra centralization | 18.5 | 0.851−0.666 = 0.185 | ✅ but baseline ambiguity (see m2) |

---

## 2. Issues found, ranked

### M1 — MATERIAL: the metastability heatmap confounds horizon with curvature (NOT in [I1]–[I7])
`Fig:heatmap` and §R2 build the (w, T) collapse surface from **three different
substrates**:
- 150k row = Exp A @150k → **α=β=1** (`exp_T_horizon.py` docstring: "Companion data
  at 150k: expA_raw_data.csv filtered to lambda==1").
- 300k row = Exp A @300k → **α=β=1**.
- 600k row = Exp T (`exp_T_horizon.py`, `ALPHA=BETA=0.88`) → **α=β=0.88**.

So as T goes 150k→300k→600k, the curvature parameter silently changes 1→1→0.88.
The paper's own central result is that `α=β=0.88` shifts `w_crit` **left by 0.100**.
Therefore part of the apparent 300k→600k leftward slide is the curvature effect, not
horizon. This is confirmed directly: at 600k,

| w | Exp T (α=β=0.88) | Exp 0b (α=β=1) |
|---|---|---|
| 0.6 | 0.20 | 0.35 |
| 0.7 | 0.10 | 0.20 |

The α=β=1 600k data (Exp 0b, `docs/exp0b_figures/exp0b_raw_data.csv`, horizon=600000)
shows ~0.15 *higher* collapse at matched w — i.e. a *less* extreme slide than Exp T
implies. **Fix options:** (a) rebuild the 600k row from the α=β=1 data we already
have (Exp 0b: w=0.6→0.35, w=0.7→0.20) so the heatmap holds α,β fixed; or (b) keep
Exp T but state explicitly that the 600k row carries curvature and the
horizon-only slide is bounded by the α=β=1 Exp 0b numbers. The metastability claim
("boundary slides left with T") still holds qualitatively under either fix — Exp 0b
α=β=1 also slides (w=0.6: 0.80@300k → 0.35@600k) — but the magnitude in the current
figure is curvature-inflated. The Table 1 label "T … α,β=0.88" is also misleading
since two of its three plotted horizons are α=β=1.

### m2 — MINOR: two different "null" baselines used interchangeably
The decomposition uses **Exp N = 0.685** as the null (Table 2, §R6). The Discussion
(§D1) states risk-neutral agents need "w > 0.666" (= Exp A @300k) and computes
"18.5 percentage points" as 0.851 − 0.666. Both 0.685 and 0.666 are α=β=1, λ=1
conditions and agree within 0.020 (the reproducibility check, §R4), so this is
defensible — but the paper alternates between them without flagging which is "the"
null. Pick one baseline for the headline pp figure (using 0.685 gives 16.6 pp; using
0.666 gives 18.5 pp) and state it. This is exactly the author's note `[I1]` extended
to the Discussion.

### m3 — MINOR: typo in the λ=7 increment arithmetic (§R6)
Text: "the loss aversion increment … is +0.266 (= 0.851 − 0.585)". The λ=1 w_crit is
**0.586**, not 0.585; 0.851 − 0.586 = **0.265**. Net shift to +0.166 is unaffected
(−0.100 + 0.265 ≈ 0.166). Fix "0.585" → "0.586" and "+0.266" → "+0.265" (or round
consistently).

### m4 — MINOR: threshold-robustness max deviation is 0.015, not 0.016
§Appendix A and §M2 state "maximum Δw_crit = 0.016". Recomputing the six estimable
variants (S̄ ∈ {8,10,12} × π̄ ∈ {−102.5, −127.1}; the three π̄ ≤ −152.5 variants are
degenerate, as the paper says) gives point-fit deviations of {0.000, 0.015} → **max
0.015**. The paper's 0.016 may come from bootstrap-median rather than point w_crit.
Either way the conclusion ("robust, ≪ CI width") holds; update 0.016 → 0.015 or note
the estimator.

### m5 — MINOR: one defined-but-uncited bib entry
`terry2021pettingzoo` is defined in `references.tex` but never `\cited` in main.tex,
so it will **not** appear in the compiled bibliography (plainnat only prints cited
keys). Either add `\citep{terry2021pettingzoo}` at the env/vectorization mention
(§E1–E2, alongside SuperSuit) or delete the entry. All other 20 entries are cited.

### Already-handled by the author ([I1]–[I7], verified correct)
- `[I3]` λ=7 quasi-separation: footnote suppresses slope/p, reports bootstrap w_crit
  only. ✅ Correct — the data are 1,1,1,0.55,0, near-separated.
- `[I2]` Exp T 600k w_crit = −0.60 unphysical: paper reports "w_crit < 0.50" bound,
  never the −0.60. ✅ Verified (`expT_results.md` shows −0.6046; main.tex avoids it).
- `[I4]` curvature CI [−0.172, −0.033] excludes 0 from above; "significant (neg.)"
  in Table 2 despite the script's `significant` column saying "no" (it tests
  CI_lo > 0). ✅ Paper's reading is correct; the script flag is one-sided.
- `[I7]` 1,124 excludes Phases 0–5. ✅ Six-experiment sum = 1124 exactly.
- `[I6]` node_1 scripted pull = 20, agents are nodes 2–6. ✅ Matches
  `DiagnosticWrapper` (`target_qty["node_1"]=20`).

---

## 3. Methodology verification (against the code)

| Paper statement | Code reality | Status |
|---|---|---|
| Collapse: 1[S̄<10 ∧ π̄≤−127.1] (Eq. 4), pre-registered commit 03c3fc6 | `is_collapsed`: `(mean_S<10) and (profit<=-128.1+1.0=-127.1)`; commit 03c3fc6 is the collapse_definition.md pre-registration | ✅ |
| CPT applied after blending: v((1−w)r_local + w·r_global) (Eq. 2–3) | `cpt_wrapper.py`: `blended=(1-w)*local+w*global`, then PT `v(blended/scale)`; reward_scale=1.0 | ✅ |
| v(x)=x^α (x≥0), −λ(−x)^β (x<0) (Eq. 1) | `cpt_wrapper.py` exact | ✅ |
| Null α=β=1,λ=1; Curvature α=β=0.88,λ=1; LA α=β=0.88,λ∈{3,7} | Exp N uses `_build_env(...,lam=1)` (α=β=1); Exp B-Reduced `ALPHA=BETA=0.88` | ✅ |
| Logistic P(collapse\|w)=σ(b0+b1w), w_crit=−b0/b1, 2000 bootstrap @ run level | `analyze_expN.fit_logistic` + `bootstrap_wcrit(n_boot=2000)` | ✅ |
| Wilson 95% CI for per-w fractions | `wilson_ci` | ✅ |
| Decomposition: curvature = w_crit(B,λ=1)−w_crit(N); increment = w_crit(B,λ)−w_crit(B,λ=1) | matches `analyze_behavioral_excess.py` (excess vs null) | ✅ (see note) |
| Substrate S ∈ [0,500], Q=max(0,S−IP) | `ScaledBaseStockWrapper` low=0,high=500 in `_build_env` | ✅ |
| node_1 fixed demand-pull; nodes 2–6 trained | `DiagnosticWrapper(scripted_nodes=["node_1"])`, pull=20 | ✅ |
| PPO MlpPolicy, n_steps=128, batch=256, SB3+SuperSuit; eval 10 ep, 20 seeds | matches all runners | ✅ |
| Env NetworkManagement-v1, Poisson μ=20, 30 periods | `or_gym_network.py` (μ=20 retail edge, num_periods=30) | ✅ |

**Note on the decomposition wording (§M4):** the paper states the curvature effect is
`w_crit(B,λ=1) − w_crit(N)`. The CSV-reported `behavioral_excess(λ=1)` in
`analyze_behavioral_excess.py` is identical to this (excess vs the Exp N null), so the
−0.100 is correctly the curvature term. The loss-aversion *increment* the text uses
(+0.265 at λ=7) is `w_crit(B,λ) − w_crit(B,λ=1)`, computed in §R6, not a column in the
script — it is arithmetic on the table and is correct (modulo the m3 typo).

---

## 4. Experimentation verification

| Exp | Table 1 design | CSV reality | Status |
|---|---|---|---|
| A @150k | w∈{0,0.1,…,1.0}, λ1, α,β=1, 20 seeds, 220 | `expA_raw_data.csv` 220 rows, 11 w × 20 | ✅ |
| A @300k | w∈{0.2,…,0.8}, λ1, α,β=1, 20, 140 | `expA300k_raw_data.csv` 140, 7 w × 20 | ✅ |
| C | w∈{0.5,0.85,1.0}, λ1, α,β=1, 12, 144 | `expC_raw_data.csv` 144, 3 w × 4 S_init × 12 | ✅ |
| N | w∈{0,0.2,…,1.0}, λ1, α,β=1, 20, 180 | `expN_raw_data.csv` 180, 9 w × 20 | ✅ |
| B-Red. | w∈{0.5,0.65,0.75,0.85,0.95}, λ∈{1,3,7}, α,β=0.88, 20, 300 | `expB_reduced_raw_data.csv` 300, 5 w × 3 λ × 20 | ✅ |
| T | w∈{0.5,…,0.80}, λ1, α,β=0.88, 20, 140 | `expT_raw_data.csv` 140, 7 w × 20, **600k only** | ✅ rows; ⚠ see M1 (table row implies T spans horizons; T is 600k-only, its 150k/300k cells in Fig 1 are Exp A α=β=1) |

All six pre-registration / design facts in §E match the runners. Run-count total 1,124
verified.

---

## 5. Action checklist before submission

1. **M1 (do first):** fix the metastability heatmap α,β confound — rebuild the 600k
   row from α=β=1 data (Exp 0b) OR annotate that 600k carries curvature; correct the
   Table 1 "T" row description.
2. m2: choose and state one null baseline for the "pp" figure (0.685 vs 0.666).
3. m3: "0.585" → "0.586", "+0.266" → "+0.265".
4. m4: "0.016" → "0.015" (or note bootstrap-median estimator).
5. m5: `\citep{terry2021pettingzoo}` somewhere in §E, or remove the entry.
6. `[I5]`: fix `\graphicspath`/figure paths for Overleaf (currently
   `docs/expT_figures/...` relative to project root).

Files produced: this doc, and `latex/references_table.csv` (all 21 bib entries with
cited/uncited status and per-section roles).
