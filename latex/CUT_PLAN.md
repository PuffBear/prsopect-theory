# AAAI Cut Plan — `main_aaai_trimmed.tex`

> **Ground truth (compiled 2026-06-24):** the file compiles clean to **17 pp total**.
> Body runs to **page 14** (Conclusion starts p.14); references occupy pp.14–17.
> **AAAI limit = 7 pp body** (references + supplementary are free/unlimited).
> **Real deficit: body must shrink from ~13.5 pp → 7 pp, i.e. cut ~6.5 pp of body.**
> This is NOT a half-page trim. It requires Tier 1 (relocate) + Tier 2 (compress) + Intro tightening.

## The one-sentence claim (the spine — protect at all costs)
Coordination collapse in cooperative MARL is a **measurable phase boundary `w_crit`**;
**loss aversion shifts it right** (impeding coordination), a **direction a mean-field
signal-to-noise argument fixes a priori** (Thm: sign invariance) and that **replicates
across 3 environments and 2 algorithms**.

Everything is judged against that sentence: does this block support it, or is it bonus/scaffolding?

## Page budget (compiled)
| Block | Pages | Verdict |
|---|---|---|
| Introduction | 1–3 (~2.5) | Tighten (Tier 2) — too long for AAAI |
| Background+Method+Setup | 3–4 (~1.5) | Keep, light trim |
| Results (boundary, null, behavioral) | 4–6 (~2) | Keep core; relocate baselines table + 600k aside |
| **Mechanism (incl. 3 theorems)** | **6–9 (~3)** | **Biggest target.** Keep Thm 3 + signs in body; relocate Thms 1–2, Langevin/Kramers derivation, affine-law derivation detail |
| **Generalization (MPE/LBF/MAPPO)** | **9–11 (~2.5)** | Keep headline + `fig:generalization`; compress LBF & MAPPO-on-substrate hard |
| Ablations (action, credit) | 11–12 (~1) | Relocate both to appendix; leave 2-sentence pointer |
| Minority sets boundary | 12 (~1) | Demote to 2 Discussion paragraphs; push `fig:hetero`+table to appendix |
| Discussion | 13–14 (~1.5) | Trim limitations prose |
| Conclusion | 14 | Keep, tighten |

---

## TIER 1 — Relocate to appendix now (no rewrite, pure move). Est. saving ~3–3.5 pp
1. **Theorems 1 (Mean-field criticality) & 2 (Calibration gap)** + their proofs → appendix.
   Keep in body: one sentence each stating what they establish, pointing to appendix.
2. **Full Langevin / Fokker–Planck / Kramers derivation** (`Gradient flow and its noise`,
   `Sharpness and metastability` math) → appendix. Keep the SNR-crossing intuition + `fig:meanfield` in body.
3. **Affine-law derivation detail** (`Derived form of the loss-aversion law` — the
   differentiation-of-E[v(r)] block) → appendix. Keep the law eq. (`eq:wcrit_lambda`),
   the RMSE 0.014 result, and `fig:affine_law` in body.
4. **Both ablations** (`Boundary Conditions: Two Ablations` — action-rep + credit-assignment,
   incl. `tab:ablation_action`, `fig:mpe_floor`) → appendix. Leave one body sentence:
   "Two ablations (App.) confirm the boundary needs a continuous policy gradient and a
   credit-assignment-capable learner."
5. **`tab:baselines`** (the recurring-w_crit reconciliation table) → appendix. Keep one
   sentence naming the canonical null (0.666) in body.
6. **"Note on the 600k row" paragraph** (curvature-deflation caveat) → appendix footnote.
7. **`tab:ledger`** (what-the-mechanism-claims) → appendix (it's a meta-table; body keeps the prose).

## TIER 2 — Compress in place (only as needed after Tier 1). Est. saving ~2–2.5 pp
1. **Introduction** 2.5 → ~1.5 pp: the six folded-in subsections (`Coordination collapse…`,
   `Why prospect theory and MARL…`, `Approach…`, `Why the supply chain…`, `Research questions`,
   `Findings and contributions`) are essay-length. Merge related-work-style paragraphs into
   one tight "Related work and gap" paragraph; convert RQ1–RQ4 description list to inline prose
   or shorten; trim the findings recap (it duplicates the abstract).
2. **Generalization**: LBF (`sec:lbf`) and MAPPO-on-substrate (`sec:appendix_crossalgo`) each
   ~1 para; lean on `fig:generalization`. Drop `fig:mpe_lambda_curves` and `fig:mappo_curves`
   from body (subsumed by combined figure) → appendix.
3. **Discussion** limitations: the five labeled limitation paragraphs → compress to one
   dense paragraph; the cascade-substrate-search detail → appendix.

## TIER 3 — Protect, do not touch
- Abstract (tighten to ≤200 words only if needed)
- Boundary exists + metastability (`sec:results_structural`)
- Loss-aversion sweep + affine law result + `tab:lambda_sweep` + `fig:decomposition`
- **Theorem 3 (sign invariance)** — statement + short proof sketch stays in body (it IS the spine)
- Generalization headline + `fig:generalization`
- The decomposition table `tab:decomposition`

## Demotions (single-substrate → not load-bearing for the AAAI claim)
- **Curvature opposite-sign**: keep `tab:decomposition` + one Discussion paragraph; the
  detailed enumerated treatment is bonus. (Seed of the follow-up paper.)
- **Minority/cascade**: 2 Discussion paragraphs + appendix figure/table. (Seed of the follow-up paper.)

## Consistency pass (after cuts, before submit)
- One baseline number everywhere (0.666 canonical vs 0.685 Exp-N — pick one, footnote the other).
- One definition of "significant."
- Remove scratch `.tex` from build (`k.tex`, `v.tex`, `d.tex`, `m_chk.tex`, `_nomt`).
- Swap to official AAAI-2027 `.sty` when released (deferred — not out yet).
