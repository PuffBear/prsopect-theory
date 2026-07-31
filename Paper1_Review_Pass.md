# Paper 1 — Reviewer 1 & Reviewer 2 Checklist Pass

**Source reviewed:** `AuthorKit27/AnonymousSubmission2027_2.tex` (current main paper — title matches the plan: *Measuring Coordination-Collapse Boundaries in Reward-Mixed Cooperative MARL*) and `AuthorKit27/SupplementaryMaterial2027_2.tex`, cross-checked against `references.bib`, the compiled `.bbl`, and `AnonymousSubmission2027_2.pdf`.

**Read this first — one blocking finding:** the checked-in `AnonymousSubmission2027_2.pdf` and its `.bbl` are **stale relative to the current `.tex`**. The `.tex` was last saved 17:15, the `.bbl` was last generated 16:57. Concretely:

- Two in-text citations, `\citep{efron1993bootstrap}` and `\citep{firth1993}` (Method section, bootstrap-CI sentence), are **not in the compiled `.bbl`** — they'd render as broken/missing citations in the actual PDF a reviewer opens.
- The stale PDF's Method section literally has different, shorter wording than the current `.tex` in that paragraph — the PDF was compiled from an earlier draft of that paragraph.
- I recompiled from the current `.tex` in a sandbox (using a substitute Times-like font, since `newtxtext` isn't installed here — so treat this as approximate, not authoritative) and got **8 content pages + 2 reference pages = 10 total**, vs. the checked-in PDF's 7+2=9. That's a real signal the paper has grown past 7 content pages since the last real compile, not a font artifact — font substitution could shift the exact break point by a line or two, but not by a full page.

**Action needed before anything else in this pass is trustworthy:** rerun `pdflatex → bibtex → pdflatex → pdflatex` on `AnonymousSubmission2027_2.tex` in a real AAAI-toolchain environment (Overleaf or local with `newtxtext` installed) and re-verify the page count. Everything below is based on the `.tex` source (which is authoritative for content), not the stale PDF.

---

## 0. Reviewer 1 [Cowork] items — status

| Item | Status | Where |
|---|---|---|
| Reverse-outline check | Done | §1 below |
| Cross-check $w_{\mathrm{crit}}$ consistency | Done | §3 below |
| Reference formatting/hygiene pass | Done | §5 below |
| Page count fits 7+≤2 | **Not currently confirmed** — see blocking finding above | §5 below |

---

## 1. Paragraph-by-paragraph reverse outline

Canonical spine: **define $w$-mixing → collapse label → boundary-fit method → primary result (loss aversion shifts $w_{\mathrm{crit}}$ right) → robustness (horizon slide, cross-env/algo) → scoped secondary findings → limits.**

Legend: **R**etain / **S**horten / **M**erge / **Mv**ove-to-supplement. Paragraphs are anchored by their opening words for lookup.

### Abstract
| ¶ | Purpose | Central claim | Spine bin | Action |
|---|---|---|---|---|
| Abstract (1 ¶) | Self-contained summary | Instrument + rightward shift | All bins, compressed | R — but see the flag below |

### Introduction
| ¶ | Purpose | Central claim | Spine bin | Action |
|---|---|---|---|---|
| "A team of learning agents can fail…" | Motivate collapse, define $w$ dial, pose the twofold question | Collapse is real; where it happens is unmeasured | define-$w$-mixing + framing | **R**, minor S (dense clause chain) |
| "Pinning down this number…" | Scope $w_{\mathrm{crit}}$, preview 4 env-algo combos, horizon re-measurement, broader PT framing | $w_{\mathrm{crit}}$ is not universal; instrument generalizes beyond supply chains | Previews Method/Results/robustness | **S** — this paragraph pre-states content that Method and Generalization say again in full; trim to ~4 sentences |
| "We build the instrument before using it…" | Design-trustworthiness principles + contributions list | (i) instrument (ii) replicated shift (iii) scoped secondary findings | boundary-fit method + primary result + secondary findings | **R** — essential anchor, does triple duty on purpose but that's appropriate for a contributions paragraph |

### Related Work (3 ¶s: coordination/credit-assignment, phase transitions, behavioral preferences)
None of these map onto the 7-bin spine — they're positioning, not instrument/result content. That's expected and correct for a Related Work section; **flagging per your instruction, not recommending deletion.** All three: **R**, with rw-P2 (phase transitions) as the first candidate to trim by a sentence or two if page budget gets tight — it's the least load-bearing of the three (Related Work generally is deprioritized under the page-allocation proposal in §4).

### Method
| ¶ | Purpose | Spine bin | Action |
|---|---|---|---|
| "We study reward-mixed cooperative MARL…" ($x_i(w)$ def) | Define $w$-mixing | **core** | **R** |
| "The boundary estimate follows four steps…" | Boundary-fit method, high-level preview | **core** | **R** |
| "The reward transform is prospect-theoretic…" (Eq. 1) | Define the PT transform applied to $w$-mixing | **core** | **R** |
| "A collapse label is a pre-specified binary…" | Collapse label, per-substrate | **core** | **R** |
| "For each condition we fit a logistic model…" (stats, Firth, decomposition baselines, substrate/algo, seeds, reading guide) | Boundary-fit method, full detail | **core**, but overloaded | **S** — move seed-count/compute-stack specifics to a one-line pointer (Supplement already has the full inventory); this paragraph currently carries method mechanics *and* substrate justification *and* algorithm roster *and* a reading guide in one block |

### Results
| ¶ | Purpose | Spine bin | Action |
|---|---|---|---|
| "A sharp boundary exists at fixed training budget" | Primary result: location | **core** | **R** |
| "The boundary is finite-horizon and metastable" | Robustness: horizon | **core** | **R** |
| "The horizon slide is not a supply-chain artifact" | Robustness: cross-env | **core** | **R**, could trim the basin-control clause (already detailed in Supplement) |
| "Loss aversion raises the boundary; curvature lowers it" | **PRIMARY RESULT — headline** | **core** | **R in full** — this is the single most important paragraph in the paper |
| "Two effect sizes answer two questions" | Primary result, effect-size decomposition | **core**, detail | **S** — largely restates Table 2's numbers in prose; let the table carry the numbers, keep only the two-questions framing sentence |

### Mechanism
| ¶ | Purpose | Spine bin | Action |
|---|---|---|---|
| Mean-field/Kramers framing + SNR definition | Scoped secondary finding (mechanism intuition) | secondary | **R**, trim hedging |
| Entropy-dissociation prediction | secondary | secondary | **R** |
| "The sign result rests on a two-part local premise…" | Restates evidential-status hedges already given | secondary | **M** into the paragraph above — redundant framing |
| Proposition + full proof | Formalizes the sign result | secondary | **Mv** proof — it is reproduced **verbatim** in the Supplement (§4.5 there). Keep the proposition statement in main, cut the derivation to "(proof in Supplement)" |
| "The curvature result is scoped more narrowly" | Restates opposite-sign SNR claim | secondary | **M** — this exact claim is stated 5 times across the paper (see §2); keep once in Results, cross-reference here |
| One-parameter saturating summary (affine law) | secondary (ledger explicitly calls this "calibrated/fitted," future-work-adjacent) | secondary | **S** — move leave-one-out bound detail and $\rho$-interior-vs-binds-on-LBF mechanics to a Supplement pointer; keep the headline RMSE + one caveat sentence |

### Generalization
| ¶ | Purpose | Spine bin | Action |
|---|---|---|---|
| Setup, 4 combos | Robustness cross-env/algo | **core** | **R** |
| Transfer prerequisite (coordination floor, MAPPO recovery, MPE wrapper) | Robustness + scope | **core** + scope | **R**, minor S (wrapper justification duplicates Supplement) |
| Generalization numbers | **Robustness — headline** | **core** | **R in full** |
| "Scope condition: action representation" | Limits (coarse-discretization scope condition) | **limits**, but located mid-paper | **Flag** — this is genuinely a limits-bin paragraph sitting in the Generalization section rather than in "Scope and Future Work." Not wrong, but worth a deliberate check: **the Discussion's "Scope and Future Work" list currently doesn't mention this limitation at all** (see below) |

### Minority/Cascade section (explicitly supply-chain-only by the paper's own framing)
| ¶ | Purpose | Spine bin | Action |
|---|---|---|---|
| Setup | secondary | secondary | **R** |
| Dose-response result | secondary result | secondary | **R** — see overclaim note on the "converging" phrasing in §4 |
| Position result + Mengesha contrast | secondary | secondary | **R** — see overclaim note on the "sign is a property of the dynamics" sentence in §4 |

### Discussion
| ¶ | Purpose | Spine bin | Action |
|---|---|---|---|
| Decomposition novelty + operational-number restatement + positioning | Mixes primary-result restatement + positioning | mixed | **S** — cut the "$w{=}0.852$ against $w{=}0.685$… $+0.167$" restatement (duplicate of Table 2); keep the novelty framing and positioning sentence |
| "Operational interpretation" | Design guidance (not strictly in the 7-bin spine — an application of results) | secondary/applied | **R** — distinct value-add, keep |
| "Scope and Future Work" | **LIMITS — core** | **core** | **R**, but **add or cross-reference the action-representation/coarse-discretization limit** — currently orphaned from the paper's designated limits section (see flag above) |

### Conclusion (1 ¶)
Wraps primary result + robustness + limits; mirrors the abstract's three-part structure (rightward/loss-aversion, leftward/horizon, cascade/downstream-node). **R.** One note: giving the single-substrate cascade finding equal sentence-level billing with the two multiply-replicated findings, in the paper's most-skimmed paragraph, is a framing risk — see §4.

### Ethical Statement (1 ¶)
Outside the 7-bin spine (required section). **R** — its dual-use language overlaps with the Discussion's "Operational interpretation" paragraph, which is expected/appropriate redundancy for a required ethics statement, not a flaw.

---

## 2. Repetition report

**"Substrate-specific" qualifications.** 6 literal occurrences of the string (lines 123, 479, 555, 597, 684, 734 in the main `.tex`), plus a larger family of synonyms ("single-substrate," "supply-chain-only," "supply-chain PPO substrate," "does not transfer," "primary substrate," "SC PPO only"). Assessment: **this is the paper's core epistemic discipline, not filler** — it's exactly the scoping language the plan asks for, and it's applied consistently everywhere the paper draws a line between what generalizes and what doesn't. Keep all of it.

**Definition of $w_{\mathrm{crit}}$.** Defined informally in the Abstract and Introduction (necessary — different audience/altitude), then formally **twice** in Method within ~40 lines: once as a one-line preview ("fit a monotone binary model… reporting the midpoint $w_{\mathrm{crit}}$") and once in full statistical detail ("$w_{\mathrm{crit}}=-b_0/b_1$, the $w$ at which fitted collapse probability is $0.5$"). This reads as intentional preview-then-detail, but it's close enough together that a reviewer could read it as accidental duplication. Low-cost fix: a one-clause bridge ("formally, ...") at the second instance would signal intent.

**Claims about centralization.** "Centralization helps" functions as a **deliberate rhetorical bookend**: it closes the Abstract, characterizes prior work in Related Work, and opens the Conclusion. This is an effective structural device, not redundancy — recommend keeping it as-is.

**Loss-aversion vs. curvature distinction.** This is the paper's real repetition finding. The full "loss aversion and curvature move $w_{\mathrm{crit}}$ in opposite directions" claim is stated **essentially in full at least five times**: the Method's justification for a curvature-only baseline, the Results headline paragraph (correctly, once, in full), the Mechanism section's "curvature is scoped more narrowly" paragraph, the Discussion's opening paragraph, and — separately — the Supplement's "Curvature is scoped to the primary substrate" subsection restates it again in nearly identical wording to the main-text Mechanism paragraph. **This is the single highest-value consolidation target for the page-budget problem**: state it fully once (Results), and have Mechanism/Discussion reference it in a half-sentence instead of re-deriving the sign logic each time. See §4 for the page-allocation implication.

---

## 3. Consistency table — every reported $w_{\mathrm{crit}}$

All values below were cross-checked between Abstract, body text, main-paper tables/figures, and the Supplement's tables. **No numeric contradictions were found** — every value that appears in more than one place matches exactly. The notable findings are about *disambiguation*, not error.

| $w_{\mathrm{crit}}$ | 95% CI | Env | Algo | Horizon | Seeds/cell | Reported in | Role |
|---|---|---|---|---|---|---|---|
| 0.666 | [0.635, 0.694] | SC | PPO | 300k | 20 | Abstract, Results, Supp (horizons + baselines tables) | **Headline structural null** |
| 0.822 | [0.785, 0.858] | SC | PPO | 150k | 20 | Results, Supp horizons table | Null, shorter horizon |
| 0.622 | [0.603, 0.641] | SC | PPO | 450k | 20 | Results, Supp horizons table | Horizon point |
| 0.573 | [0.547, 0.598] | SC | PPO | 900k | 20 | Results, Supp horizons table | Horizon point |
| 0.685 | [0.656, 0.710]* | SC | PPO | 300k | 20 | Results Table 2, Supp baselines table, Supp entropy table (ent=0 row) | Linear baseline, **independently reproduced** (Exp N) |
| 0.586 | [0.539, 0.624] (point); also reported as a **difference** CI [−0.172, −0.033] vs. the null | SC | PPO | 300k | 20 | Results Table 2, Supp λ-sweep table, Supp baselines table | Curvature baseline (λ=1, α=β=0.88) |
| 0.774 | [0.751, 0.798] | SC | PPO | 300k | 20 | Results Table 2, Supp λ-sweep | λ=3 loss-aversion (the abstract's headline effect point) |
| 0.852 | [0.841, 0.864] | SC | PPO | 300k | 20 | Results Table 2, Supp λ-sweep, Discussion, Conclusion | λ=7 loss-aversion (plateau) |
| 0.715/0.806/0.825/0.835 | (Supp only) | SC | PPO | 300k | 20 | Supp λ-sweep table (main gives the series inline without CIs) | λ=2,4,5,6 |
| 0.796 | [0.742, 0.860] | SC | MAPPO | matched | 20 | Generalization, Table 4, Supp cross-algorithm §| λ=1 |
| 0.889 | [0.856, 0.917] | SC | MAPPO | matched | 20 | Generalization, Table 4, Supp | λ=7 |
| 0.78 → 0.89 | +0.158 [0.108,0.212]@λ7; +0.101 [0.055,0.150]@λ3 | MPE | MAPPO | 2M (calibration horizon) | 20 | Generalization, Table 4, Supp MPE §| λ=1→7 endpoints (λ=1 here is the 0.88-curvature-baseline endpoint, **not** the 0.74 linear null — see note below) |
| 0.74 | — | MPE | MAPPO | — | — | Generalization only | Linear (no-transform) null, wrapper-calibration target |
| 0.26 → 0.83 | — | LBF | MAPPO | — | 20 | Generalization, Table 4, Supp LBF §| λ=1→4 endpoints (λ=1 here is the 0.88-curvature-baseline endpoint, **not** the ≈0.08 linear null) |
| ≈0.08 | — | LBF | MAPPO | — | — | Supp only | Linear null (not in main text at all) |
| 0.82 → 0.68 | non-overlapping | MPE | MAPPO | 2M→8M | — | Results, Supp | Horizon-slide transfer |
| 0.33 → 0.20 | non-overlapping | LBF | MAPPO | 1M→4M | — | Results, Supp | Horizon-slide transfer |
| 0.685→0.690→0.715→">1" | — | SC | PPO | 300k | 20 | Supp entropy table only (main narrates, no table) | Entropy coef. 0, 0.001, 0.01, 0.05 |
| 0.13 → 0.37 | non-overlapping | LBF | MAPPO | — | — | Supp only | Entropy dissociation |
| flat (+0.05) | overlapping | MPE | MAPPO | — | — | Supp only, pooled over 1,320 runs | Entropy dissociation (null) |
| 0.586 → 0.856 | — | SC | PPO | 300k | — | Results (minority §) only | Single-node dose-response, λ=1→7 |
| 0.856 / 0.800 / 0.797 / 0.726 / 0.673 | — | SC | PPO | 300k | — | Results (minority §) only | Position sweep, λ=7 fixed |

\* The 0.685 CI shown is from the Supplement's entropy table (the ent=0 replicate of the same condition); the main-text decomposition table doesn't restate a CI for 0.685 itself, only for the differences against it.

### Two disambiguation notes worth your attention (not errors)

1. **The paper deliberately reports two "λ=1" values per substrate** — a true linear null ($\alpha{=}\beta{=}1$) and a curvature-baseline ($\alpha{=}\beta{=}0.88,\lambda{=}1$) that anchors each λ-sweep. This pattern is correct and consistent (SC: 0.685 vs 0.586; MPE: 0.74 vs 0.78; LBF: ≈0.08 vs 0.26), but the Supplement's disambiguation table (`tab:baselines`, "the recurring $w_{\mathrm{crit}}$ values") **only covers the supply chain**. MPE and LBF get the same two-null structure without an equivalent explicit callout, which is a minor clarity gap if a reader jumps straight to the Generalization section.
2. **"Converging" and "matching" language slightly outruns the numbers** in two places — flagged in §4, not repeated here.

### One item that doesn't match the plan document itself

Your Stage P1-D notes say the Supplement "reports the complete experimental apparatus, including 1,524 runs." The Supplement's own experiment-inventory table sums to **2,364** runs for the primary supply-chain apparatus alone (I verified the row sum: 220+140+144+180+300+400+160+160+140+400+120 = 2,364, and it matches the table's stated total exactly), before adding the 1,760 MPE + 1,600 LBF + 700 cross-algorithm runs on top. Either the plan text is describing an earlier state of the experiment log, or there's a number to reconcile — worth a quick check on your end since it's your own planning document, not something I'd change in the paper.

---

## 4. Sentences that sound stronger than the evidence

1. **"converging on the fully loss-averse value ($0.852$) by $\lambda{=}7$"** (Minority section, dose-response). The single-node dose-response actually reaches **0.856** at λ=7, not 0.852 — a 0.004 gap the sentence glosses over with "converging on." Likely within noise, but the phrasing implies exact agreement where the reported numbers differ.

2. **"matching the supply chain's $+0.089$"** (Generalization, MPE result). MPE's λ=3 excess is $+0.101$ — a 0.012 gap from the supply chain's $+0.089$. Defensible because 0.089 falls inside MPE's own CI $[0.055,0.150]$, but "matching" reads as tighter agreement than "consistent with, within CI" would.

3. **"LBF thus confirms the prediction"** (Mechanism, entropy dissociation) sits two sentences before **"This is indirect support only"** in the same paragraph. Both are defensible individually, but "confirms" and "indirect support only" in the same breath creates a legibility problem for a reviewer skimming for overclaim.

4. **"so an audit of reward transforms should target that node, not the mean"** (Minority section) is prescriptive operational advice drawn from a single substrate that the paper's own claim ledger elsewhere labels "measured only on SC PPO; future work for other directed-flow domains." The advice may well be right, but it's stated with more confidence than the paper's own scoping elsewhere grants it.

5. **"which suggests the sign is a property of the dynamics rather than of the preference structure"** (Minority section, contrasting with Mengesha et al.'s non-monotone evolutionary-model result). This is a fairly broad claim about what governs monotonicity in general, built from a contrast between two papers using different model classes (deep MARL vs. evolutionary/quantal-response). Worth softening to something like "is at least consistent with" rather than a general "suggests."

6. **Conclusion's equal billing.** "…rightward under loss-averse reward transforms, leftward with a longer training horizon, and, on a directed supply network, largely with the reward design of the downstream node nearest demand." The first two are replicated across four environment-algorithm combinations; the third is explicitly single-substrate everywhere else in the paper. Presenting all three in one sentence, in the paper's most-read paragraph, gives the cascade finding more rhetorical weight than its evidence bucket ("measured only on SC PPO; future work for other directed-flow domains" per the claim ledger) supports. Not a factual error — a framing risk.

---

## 5. Reference formatting/hygiene pass

**Unused entries in `references.bib`** (safe to drop — cited nowhere in either `.tex`): `barfuss2025collective`, `jiang2018dgn`, `samvelyan2019smac`, `strogatz1994nonlinear`, `zhao2025maca`. Harmless if left (BibTeX only pulls cited entries into the `.bbl`), but worth cleaning for hygiene.

**Real entry-type bug:** `tampuu2017multiagent` is typed as `@inproceedings` with `booktitle={PLoS ONE}` — but PLoS ONE is a journal, not a conference. BibTeX itself throws a warning on this ("can't use both volume and number fields in tampuu2017multiagent") because `@inproceedings` doesn't expect both fields the way `@article` does. Fix: retype as `@article` with `journal={PLoS ONE}`.

**Stale `.bbl` (the blocking finding from the top of this doc):** `efron1993bootstrap` and `firth1993` are cited in the current `.tex` but absent from the compiled `.bbl`. This is the most important fix in this whole pass — rerun the LaTeX/BibTeX cycle before anything else.

**No missing citations otherwise, no duplicate keys.** The `references.bib` header comments (documenting corrected author lists, corrected arXiv IDs, a removed duplicate `lowe2017maddpg` key) show a genuinely careful hygiene history already — the bib file itself is in good shape apart from the two items above.

**Style consistency (minor, cosmetic):** arXiv preprints are entered two different ways — some as `@article` with `journal={arXiv preprint arXiv:XXXX}` (e.g. `yamaguchi2025`, `dmarl2025`, `mengesha2025`, `weng2026`), others as `@misc` with `eprint`/`archivePrefix`/`primaryClass` (e.g. `schulman2017ppo`, `hubbs2020orgym`, `ethayarajh2024kto`). Both render fine under the AAAI style, so this isn't user-visible, but it's inconsistent house style within the file.

**Page count:** see the blocking finding at the top — cannot confirm 7+≤2 holds until the `.tex` is recompiled with the real AAAI toolchain. My sandbox recompile (approximate font) landed at 8 content + 2 reference pages, which is a real warning sign given the paper's growth since the last authoritative compile, not a font artifact.

---

## 6. Proposed seven-page content allocation

This assumes the consolidation moves below are taken — they target the redundancy found in §2 and the overclaim-adjacent restatements in §4, and are chosen specifically because none of them touch a scientific claim, only where/how many times it's stated.

| Page | Content | Trim relative to current |
|---|---|---|
| 1 | Title, Abstract, Intro through the twofold-question paragraph | Shorten intro-P2 (the "pinning down this number" paragraph) to ~4 sentences — cuts previewed content that Method/Generalization state in full anyway |
| 2 | Intro contributions paragraph, Related Work (all 3 ¶s) | Trim phase-transitions paragraph by 1–2 sentences if still tight |
| 3 | Method (all 5 ¶s) | Shorten the stats/decomposition paragraph — move seed-count and compute-stack detail to a one-line Supplement pointer |
| 4 | Results (Fig. 1, Fig. 2, Table 2, all 5 result paragraphs) | Shorten "Two effect sizes" paragraph — let Table 2 carry the numbers instead of restating them in prose |
| 5 | Mechanism section (Fig. 3, Fig. 4, Proposition) | **Biggest single saving**: cut the full proof to "(proof in Supplement §4.5)" since it's verbatim-duplicated there; merge the "two-part premise" paragraph into the SNR-framing paragraph; compress the affine-law paragraph to headline RMSE + one caveat |
| 6 | Generalization (Table 3 ablation, Fig. 5, Table 4) | Compress Table 3 — it is **near-verbatim duplicated** in the Supplement (`tab:ablation_action`); keep a compact one-line summary in main with a pointer, or keep the table but drop the caption's repeated explanatory clause |
| 7 | Minority/Cascade section, Discussion, Conclusion, Ethical Statement | Cut the Discussion's "$w{=}0.852$ against $w{=}0.685$… $+0.167$" restatement (duplicate of Table 2); consider adding one clause to "Scope and Future Work" naming the action-representation/coarse-discretization limit, which is currently missing from that list entirely |

Estimated recovered margin from these five moves: roughly 0.6–0.8 of a page — enough to absorb the overflow the sandbox recompile suggests, with room to spare. None of these changes drop a claim, a number, a figure, or a table; they remove *restatement* of things said correctly once elsewhere, and relocate a verbatim-duplicated proof and a verbatim-duplicated table to the Supplement where the full version already lives.

---

## Summary of what needs your attention

1. **Recompile the PDF from the current `.tex` with the real AAAI toolchain and rerun BibTeX** — this fixes the two missing citations and gives you a trustworthy page count. Do this before the freeze, not after.
2. Retype `tampuu2017multiagent` as `@article`.
3. Drop the 5 unused `.bib` entries (optional, hygiene only).
4. Decide on the 5 consolidation moves in §6 if the recompile confirms you're over budget.
5. Your own [Agriya] item — "confirm abstract carries exactly one headline number" — the abstract currently carries the boundary point estimate (0.666) plus two effect sizes (+0.089, +0.188), so worth a look when you do that pass.
6. Consider a one-clause fix to "Scope and Future Work" so the action-representation limit isn't orphaned from the paper's own limits section.
