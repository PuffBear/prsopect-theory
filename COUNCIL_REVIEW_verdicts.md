# Council Verdicts on the 62 Pointers

*Input: your review of `AnonymousSubmission2027_2`, four independent reviewer responses (R1–R4), and the paper + supplement themselves. For each pointer the "council" reports two things you asked for:*

- **Advantage?** — does acting on the pointer materially improve the paper (not just polish)?
- **AAAI-agreeable?** — would a competent AAAI reviewer nod along, or push back?

**Verdict legend**

- ✅ **ADOPT** — real advantage; a reviewer agrees.
- 🟡 **ADOPT (minor)** — helps readability/polish; low stakes; reviewer indifferent-to-positive.
- 🔵 **ADOPT w/ MODIFICATION** — your instinct is right but the reviewers correct the *execution* (don't do it literally as written).
- ⚪ **TASTE / OPTIONAL** — no material advantage either way; house-style call.
- ❌ **REJECT / MISDIAGNOSED** — the pointer is wrong, or acting on it would not help (or would hurt).

A one-line **council note** flags where the four reviewers *disagreed*, because those are the only places your judgment is actually needed.

---

## Meta-verdict (where all four converge)

All four reviewers independently reached the **same top-level diagnosis**, and it is worth stating before the line items because ~40 of your 62 pointers are symptoms of it:

> The paper's *content* is exciting; its *delivery* is tiring. The single root cause behind most of your pointers is (a) **terms used before they are defined** (α/β/λ, curvature, MPE, LBF, CTDE, the collapse labels, b₀/b₁, κ_T) and (b) **the four-step protocol restated 3–5 times** (abstract → intro → method opening → method boundary paragraph → conclusion). Fix ordering + de-duplicate + lead with the *number* not the *protocol*, and the paper flips from "careful, modest, accept" to "memorable."

R1 and R4 additionally surfaced **seven correctness-level problems that are more reviewer-lethal than anything in your list** (see the dedicated section below). The council's strongest recommendation: those seven come before any of the 62.

---

## Abstract (1–5) — *your instruction: read last; verdicts here for completeness*

**1. "Reusable boundary instrument" sentence too surface-level.** 🔵 ADOPT w/ MODIFICATION.
Council: you're uneasy for the wrong reason. It's not too abstract — it's a *procedure list with no result*. All four say: replace with your own pointer-5 thesis sentence and immediately give a CI'd number (`wcrit = 0.666 [0.635, 0.694]`). An abstract with a CI reads as a *measurement* paper; a four-verb protocol reads as a *methods proposal*. Reviewer strongly agrees.

**2. "Loss-averse transforms" undefined.** ✅ ADOPT. Unanimous. Rewrite to self-define: "a reward transform that weights losses λ× more than equal gains." Reviewer agrees — undefined jargon in an abstract is a real ding.

**3. "+0.089 net of curvature (+0.188 in isolation)".** ✅ ADOPT — **and it is worse than you thought.** R1 and R4 both catch that the labels are **inverted**: +0.089 is the deployment shift *with curvature's −0.100 still in it*; +0.188 is the curvature-*removed* number. "Net of X" conventionally means "after removing X," so the abstract calls the contaminated number clean and vice-versa. This is a headline effect size, in paragraph 1, checkable against Table 1 in thirty seconds. **Council priority: P0.** A reviewer who catches it stops trusting the paper.

**4. Drop "supply-chain" from the abstract.** 🔵 SPLIT — the one place the reviewers genuinely disagree.
- You + R1 + R3: cut it; it's scope-hedging, and hedging is a body activity.
- R2: *keep* it — "primary substrate" alone is insider language; "primary supply-chain environment" is clearer to a cold reader.
Council ruling: **R2 wins on the noun, you win on the hedge.** Name the environment once ("supply-chain") so the reader knows what it is, but delete the *"substrate-specific unless replicated"* hedge from the abstract. Also: "substrate" is opaque throughout — "environment/testbed/setting" reads better (R2).

**5. Prefer the pointer-5 thesis line.** ✅ ADOPT. Unanimous — that *is* the thesis sentence; move it to position two, right after the collapse setup. Plus: cut the abstract to ~150 words (R1/R2/R3 all say it's over-dense at ~230).

---

## Introduction (6–15)

**6. Fact-check the citations.** ✅ ADOPT — highest-risk item in the intro. R1 and R2 both found the *same* three misattributions:
- "more centralization helps (Akella 2025; Liu 2023)" — neither varies a centralization dial; Akella is a single-author non-archival preprint, Liu is sparse-reward lazy agents. Overclaim.
- The mean-field grouping mis-assigns work (Hu et al. is large-population repeated games, *not* networked; only Leonte is networked).
- **Mengesha et al. (2025): the direction you cite is wrong** — their result is *non-monotone* and partly *opposite* to yours (strong loss aversion can lock in defection). You cite it as a clean monotone tipping-point shift, and again in Discussion. R1/R2 both flag; R3 flags it as unverifiable-from-here. Council: this is a genuine mischaracterization a knowledgeable reviewer will catch. Cite it as a *contrast*, or drop it.

**7. Remove x_i(w) equation from the abstract (duplicated in intro).** ✅ ADOPT — but R1 flips *which* copy to cut: keep the **intro** instance (that's where the reader learns what w means and gets the "w=0 → own reward only" gloss), cut it from the **abstract**. Unanimous on the cut.

**8. Citations for "lazy-agent / credit-assignment mitigate without locating a boundary."** ✅ ADOPT. You already own all six needed keys — Liu, Shi&Peng for lazy/RO; Sunehag, Rashid, Foerster, Son for credit assignment. R3 notes they're technically earned two sentences earlier, so it's belt-and-suspenders, but low cost and removes doubt for a skimming reviewer.

**9. Citations for "closest phase-diagram studies."** ✅ ADOPT. Cite Yamaguchi + Capannoli inline here (both already in Related Work). R1 adds: place Weng & Lee here too (it's your closest competitor — see item on P0/refs). Your "none estimates a CI'd crossing" claim survives *only because of the CI clause* — say so explicitly rather than let a reviewer discover it.

**10. Specify which environment-algorithm combos wcrit applies to.** ✅ ADOPT — high value. All four love this. R1's framing is the strongest: turn it into a stated **applicability envelope** ("the instrument needs a reward-mixing dial, a continuous scalar action channel, and a learner that coordinates at high w"), which converts your two *negative* ablations (discretization kills it; independent PPO floors out) from apologies into a scope statement. Preview the four cells (SC-PPO, SC-MAPPO, MPE-MAPPO, LBF-MAPPO) here.

**11. "Nothing else to check."** ⚪ — reviewers supplied 12–15 below.

**12. Name the DAG / concrete collapse early.** ✅ ADOPT. R1: the single-node cascade is your most striking result and the intro doesn't hint the substrate has a topology — plant it with one clause. R3: give a concrete instance of "collapse" (nodes that stop ordering) within the first paragraph instead of two pages of abstraction. Strong agreement.

**13. Motivate why location matters / horizon-dependence early.** ✅ ADOPT. R1: frame the horizon slide as a *second measurement*, not a caveat — it's the thing that makes a naive reader suspicious ("if wcrit moves with training time, is it real?"), so meet it head-on. R3 concurs (merge "field knows direction" with "why location matters").

**14. "We differentiate below" placeholder / contribution (iv) is an apology.** ✅ ADOPT. R1 + R3: item (iv) is the only contribution phrased as a disclaimer and it's your *best* result — either promote the cascade to a real contribution or cut the hedge. R3: the "differentiate below" clause adds nothing; either cut or do the differentiation now.

**15. Simplify/rank the contributions.** ✅ ADOPT. R2 gives the cleanest 3-item restructure (Measurement / Replicated finding / Scoped supply-chain analysis). Contribution (iii)'s "checked... premise" language is opaque cold (R3) — see the P0 "checked" item.

---

## Related Work (16–17)

**16. Do we need bold paragraph headings?** ❌ REJECT (keep them). **Unanimous, all four.** AAAI style permits bold run-in headings, they're standard in this literature, and removing them makes the section a wall. R2's only tweak: shorten the first heading to "Coordination failures and credit assignment" (don't project your term "collapse" backward onto cited work).

**17. Related work otherwise fine.** 🔵 ADOPT w/ MODIFICATION. You said it's good; the reviewers found two real issues you missed: (a) "A thin bridge connects..." is a **gap-assertion topic sentence** that reads as defensive (R1) — state what exists, let the reader see the gap; (b) **Ethayarajh/KTO is your strongest motivation and it's parenthetical** (R1) — KTO ships a Kahneman–Tversky loss-aversion coefficient in a production alignment objective *right now*; that deserves a sentence. Also the mean-field grouping needs splitting (R1/R2, see #6).

---

## Preliminaries and Method (18–34)

**18. Expand CTDE.** ✅ ADOPT. Unanimous. But R2 adds a sharper point: **check that your PPO setup is actually CTDE.** Independent PPO through SB3 is *not* CTDE (no centralized critic). Don't define the whole paper as "settings under CTDE" if the primary learner is independent PPO. This is a real technical correction, not just an expansion.

**19. Add w; and isn't this redundant with the abstract?** ✅ ADOPT. Add `w`, compress, and your instinct is right — keep it *here*, thin it in the abstract.

**20. "One dial at a time" sentence: heavy, α/β/λ undefined, redundant.** ✅ ADOPT. Unanimous — this is a genuine ordering bug (α,β,λ used a paragraph before definition) *and* redundant with the four-step protocol. Split it; move after the transform definition. (Minor: it's λ, not "gamma" — your note said gamma.)

**21. Four-step protocol repeats the previous paragraph + abstract.** ✅ ADOPT. Unanimous: state the protocol **once, canonically, here**; reduce the intro to *why* the protocol has this shape (an argument), and the conclusion to one clause.

**22. "Linear utility."** ✅ ADOPT. Replace with "α=β=1, λ=1 leaves the reward unchanged, v(x)=x." R1/R2: "linear utility" is an econ term of art doing no work.

**23. "Probability weighting."** ✅ ADOPT — and reframe as a *differentiator, not a disclaimer*: full CPT reweights probabilities; you omit that channel *so the only thing varying between matched runs is the scalar value transform* (this is precisely how you differ from Lalmohammed). R2 adds: since you drop probability weighting, call it a "prospect-theoretic value transform," not "CPT."

**24 + 31. "Curvature" never defined; used as a noun for ~15 pages.** ✅ ADOPT — **council flags this as the single highest-leverage readability fix** (R1, R2, R3 all independently call 20/22/24/31 "one bug"). Define once, concretely, at the transform (α,β = exponent/diminishing sensitivity; λ = separate loss multiplier), then state *why you examine it*: because it's present in every λ-condition, so unmeasured it contaminates attribution — and on this substrate it moves the boundary the *opposite* way, so a naive "prospect theory as one knob" reader under-reports the λ effect by half. That two-sentence argument *is* the justification for the whole decomposition.

**25. Is the Lalmohammed difference clean?** 🔵 ADOPT w/ MODIFICATION. The difference is *methodologically* clean (all four agree) but the *wording overclaims*. R1/R2: "confounding any shift with a change to the optimizer" criticizes their design for a goal they didn't have — reads uncharitable; and they're a workshop paper, so resting your central differentiation on non-archival work is fragile. Widen to also contrast Prashanth 2016 / Ghaemi 2024, and frame *positively* ("we keep the optimizer bit-identical"). R3 found it clean; council sides with R1/R2 on softening the causal verb.

**26. Indicator notation 1[·].** ❌ REJECT (notation is fine). All four: standard, no explanation needed. **But the council redirects your worry:** the real problem underneath is (a) `−127.1` is a suspiciously tuned magic number sitting ~1.0 above the dataset floor of −128.1 (R1) — one sentence ("collapsed runs pile up at the lazy attractor's deterministic profit floor ≈ −128, so the profit clause is a redundancy check on S̄<10") converts a red flag to a strength; and (b) define S̄, π̄ and the evaluation window at first use (R1/R2). R4 adds: state *how* the threshold was set pre-registration.

**27. MPE/LBF collapse labels opaque; environments not yet introduced.** ✅ ADOPT — real ordering bug (labels precede environment naming). R2's small comparison table (setting / learner / local incentive / team objective / collapse observable) is the cleanest fix. Underneath: **the LBF label is never actually stated as a number, anywhere** (R1) — see the P0 section; that's the more serious half.

**28. Pre-registration/commit language reads like explaining to an examiner.** 🔵 ADOPT w/ MODIFICATION. All four: pre-registration statements are *standard and expected* in 2024–26 ML, *especially* for a paper whose contribution is pre-specification — **keep it**, but trim to one sentence and move anonymization mechanics to a footnote. R1's addition is important: the honesty is only worth something *because* two probes failed — say "including two probes that returned negative results (Supp §6)" right here.

**29. Define b₀, b₁.** ✅ ADOPT. Unanimous, one clause ("intercept b₀, slope b₁; wcrit = −b₀/b₁ is the w where fitted collapse prob = 0.5"). R2 adds: note b₁ is expected negative.

**30. Cite Firth.** ✅ ADOPT. Unanimous — Firth (1993, *Biometrika* 80(1):27–38). R1/R2 add a *bigger* gap: **the bootstrap is undocumented** (which bootstrap? BCa/percentile? how many resamples?) — for a paper whose contribution is confidence intervals, that's a real hole, larger than the Firth citation. Also distinguish "bootstrap" (uncertainty) from "Firth" (point-estimate correction) — they're not the same procedure (R2).

**32. PPO sentence implies all three environments use PPO.** ✅ ADOPT — real correctness/clarity issue (R3 confirms from the tex: the PPO sentence directly follows the MPE/LBF sentence). Underneath, R1's deeper catch: **the MAPPO implementation is never named anywhere** — half your results run on an unnamed learner. That's a straight reproducibility hole; scope the sentence *and* name the MAPPO impl.

**33. 1,524-run inventory sentence is redundant.** 🔵 SPLIT. R3 pushes back — it's an *inventory manifest*, not a protocol repeat, so it's not redundant with the four-step protocol. But R1 + R4 found something worse: **the accounting doesn't close** — the Supplement table sums to 1,524 but contains *no rows* for the 450k/900k horizons, the finer λ≥5 grid, or the entropy ablation, all of which the paper reports. Council ruling: trim the prose (your instinct) **and** fix the inventory so every reported estimate maps to an inventoried run (R4's P0-5). This is a "referee does arithmetic and it fails" item.

**34. Clarity on primary vs. transfer results.** ✅ ADOPT. Unanimous — end the Method with a one/two-sentence reading guide (primary = SC-PPO; MPE/LBF/MAPPO = transfer tests of what travels). A three-row table is even cleaner (R2). Highest-value structural fix in the section.

---

## The Boundary and Its Reward-Transform Shift (35–40)

**35. Move Figure 1 up.** ✅ ADOPT (trivial). Pure LaTeX float — `[t]`/`[h]`. Unanimous.

**36. Four horizons but Fig 1 shows only 150k/300k.** ❌ MISDIAGNOSED (but a caption fix is warranted). R1 (and R2) checked: panel (a) shows *two representative* fitted curves for legibility; panel (b) plots *all four* wcrit + CIs (that's where the log-linear fit lives). Nothing is missing. So the "bug" you flagged isn't a data bug. **But** the caption should say "(a) two representative horizons; (b) all four" so a reviewer doesn't have your reaction. Council: fix the caption, not the data. (R2 also: caption "20 seeds per horizon" should read "per w-cell at each horizon.")

**37. Horizon slide deserves a figure (generalizability).** ✅ ADOPT. All four agree the generalized slide is under-supported visually — a reviewer will suspect the MPE/LBF endpoints were cherry-picked. Show the points + intervals.

**38. Proposed 4-panel Figure 1 (a: SC metastability; b: SC-MAPPO; c: MPE; d: LBF).** 🔵 ADOPT w/ MAJOR MODIFICATION — **do not build it as specified.** R1, R2, R3 independently checked the supplement and found: **your proposed panel (b), SC-MAPPO metastability, does not exist — there is no SC-MAPPO horizon sweep in the submitted material.** And MPE/LBF have only *two* horizon points each (can show direction, not log-linearity). A 2×2 with 4, 0, 2, 2 points invites "why?" and the honest answer is "we didn't run it" — worse than not making the figure.
- Council ruling: either **(A)** run SC-MAPPO at 3–4 horizons + add a third point to MPE and LBF (R1: "the highest-value experiment left in the paper" — it makes metastability a genuine cross-environment result), or **(B)** keep Fig 1 = (a) sharp boundary + (b) four-point log-linear fit, and add a small 3-panel direction-only transfer figure. R2 also warns: a *tall vertical* 4-panel strip prints too small in two-column AAAI — use 2×2. Your format instinct (single figure, sparse caption) is right; the *data behind panel (b)* is the blocker.

**39. "Two effect sizes answer two questions" vs Table 1 confusion.** ✅ ADOPT — and there are **numeric inconsistencies** underneath (R1): text says "+0.166" but Table 1 says "+0.167"; the reconciliation `−0.100+0.265≈0.166` should be 0.167; Discussion then uses +0.167. Pick one, recompute from unrounded wcrit, make table = text. Also Table 1's "vs. curv." column has **no ✓ marks** while the text calls it "the cleaner estimate" — implicitly says your cleaner estimate isn't significant. Name the comparators explicitly ("vs. reproduced identity" / "vs. curvature-matched," R2). This is a real "table and prose disagree" catch, not just presentation.

**40. Figure 2 dotted wcrit lines invisible.** ✅ ADOPT (can't verify pixels from source, but all reviewers accept the premise). Fix: drop the dotted verticals, use a rug of wcrit ticks on the x-axis, or thin to λ∈{1,3,5,7} (seven overlapping sigmoids is past the readability limit anyway). Note you wrote "figure 20" — you mean Figure 2.

---

## Mechanism (41–48)

**41. Is Equation 2 100% correct?** 🔵 It's a *definition*, so it can't be "wrong," but the framing overclaims (R1, R2 both). Issues: S = F/√Σ is not dimensionless (units t^−1/2); Kramers escape depends on the *barrier*, not one drift/√variance ratio; and — R1's key point — **the log-linear-in-T slide does not follow from Eq. 2; it follows from the barrier picture (Supp Eq. 11):** τ ≍ τ₀ e^{ΔU}, set τ=T ⟹ ΔU = ln(T/τ₀) ⟹ wcrit = a − b·ln T = your fitted `2.44 − 0.138 ln T`. **You are showing the weaker argument and hiding the stronger one.** Council: lead the main text with the barrier (it *predicts* the observed functional form); demote S to "a monotone proxy for the barrier." Genuinely strengthens the section. Don't call Eq. 2 a consequence of mean-field theory — it's a modeling choice (R2).

**42. κ_T undefined before use.** ✅ ADOPT. Unanimous — one clause ("κ_T a horizon-dependent critical value; longer horizons need less escape SNR"). R2 caveat: once κ_T "absorbs everything unobserved," S ≈ κ_T risks non-falsifiability — call it an organizing picture, not a predictive equation.

**43. Does collapse mean τ_{L→C} > T?** ✅ Your reading is correct (all four confirm). But R2 adds a genuine precision bug: the *individual run* collapses when the **random** first-passage time exceeds T; the paper writes it with the **mean** τ. And the 50% boundary is the **median** first-passage = T, not the mean (under exponential Kramers escape, median = τ ln 2). Worth fixing in main + supplement.

**44. "Increasing w shortens escape; loss aversion lengthens it" — how do we know?** ✅ ADOPT — **this is a P0-level honesty issue** (R1, R2, R4 all converge). The w-half is *measured* (Supp Eq. 4 finite-difference probe). The λ-half is **not measured by anything** — and Supp §6 reports *two* pre-registered probes that failed to localize it. As written, the premise is "checked," then Proposition 1 concludes what the premise assumed — a circle. Council: split the two halves explicitly; call the λ-half *assumed, consistent with data, tested indirectly by the entropy dissociation, not independently verified.* R4's P0-3 patch does exactly this and is the cleanest wording. **Do not call it "checked."**

**45. Entropy dissociation correct? Figure 3 good?** 🔵 ADOPT w/ MODIFICATION — **R1 found a real problem you'll want to know about.** In Supp Table 7 the supply-chain arm rises on *overlapping* CIs (0.685→0.715) yet you call MPE "flat" *because its CIs overlap.* So the same evidentiary standard yields "rises" for SC and "flat" for MPE — only LBF (+0.24, non-overlapping) is clean. Reframe as "LBF confirms; supply chain directionally consistent but within noise; MPE null," or get more SC seeds. Also **Figure 3 plots the censored ent=0.05 SC point as a value+error bar when the table says it's censored** (collapse at all w) — plot it as an arrow/censored bound, not a point (R1, R2). The MPE *null itself* is solid and well-powered (R1) — keep it, it's good.

**46. Is Proposition 1's math correct?** ✅ YES — all four verified the chain-rule + implicit-function-theorem algebra is clean. Three caveats: (a) all content is in premise (i), a strong structural assumption (τ depends on (w,λ) only through scalar S); (b) main-text and supplement versions have *different hypotheses* — make them identical (R1); (c) own it: "Proposition 1 is elementary; its content is the empirical premise, not a theorem" (R1). R2 suggests re-proving directly on the survival probability q(w,λ;T) to dodge the mean-vs-random issue from #43 — cleaner and matches the fitted object.

**47. Claim-ledger table → paragraph?** 🔵 SPLIT. R1: you have **three** near-duplicate ledgers (main Table 2, Supp Table 5, Supp Table 8) — keep *exactly one*, a 3-row table, scannable; don't convert to prose (a paragraph of scope-hedging "reads as pleading"). R2/R3: a paragraph is fine and saves space. Council ruling: your instinct (cut visual weight) is right; **the higher-value move is deleting the two duplicate ledgers regardless of table-vs-prose.** One place, once.

**48. "A theory-bounded law, tested out of sample" — sensical? Fig 4 good?** 🔵 ADOPT w/ STRONG MODIFICATION — **R1's most consequential structural recommendation.** The paragraph reads fine; the *claim* oversells. R1's findings: (a) the "out-of-sample test" barely tests — calibration residuals (λ=3,7) are the *same size* as held-out residuals (~0.015), and RMSE 0.014 is *below* your CI half-width (~0.02), so the law is indistinguishable from noise around several other functional forms — **compare against a logistic/power-law/linear-interp baseline** or the claim is empty; (b) "theory-bounded" bought nothing on SC (ρ=0.61 is interior; the bound never binds) — it only does work on LBF where it saturates and misfits, which is your *actual* "assertion → measured result" moment. **Council's headline recommendation across all reviewers: demote the affine law to the supplement.** It's a 7-point, 1-parameter, single-substrate curve fit that explicitly doesn't transfer, and it costs a figure + subsection + ~a column. That space buys pointer 10's scope paragraph, the curvature definition (24/31), and the Figure 1 you want. R2/R3 agree it's over-framed (rename to "a one-parameter saturating summary"); R1 goes further to cut. A reviewer's summary sentence will not mention the affine law.

---

## The Instrument Travels; So Does the Direction (49–52)

**49. "Two further environments and a second algorithm" → name the four cells.** ✅ ADOPT. Unanimous, and it's more than phrasing (R1): that phrase describes a 2×2 you *didn't* run (independent PPO floors out on shared-reward tasks) — you say so two paragraphs later. Name the four cells up front and state the design is *not fully crossed*, so the reader understands the design instead of discovering the gap as a concession.

**50. Critique the transfer-boundary paragraph (you couldn't name the issue).** ✅ ADOPT — **R1 named it for you: you tuned the environment until the instrument gave a reading.** "Its scale is chosen so the no-transform baseline boundary is interior" + the herding wrapper is yours (local_ratio=0) → a reviewer reads garden-of-forking-paths. It's *defensible* (you need an interior baseline), but reframe from buried passive-voice concession to *stated method*: "MPE's wcrit is a property of our wrapper, not of MPE; we report the calibration explicitly." Three reviewers add: **LBF is your stronger transfer** (native public-goods structure, no wrapper needed) — lead the section with LBF, not MPE (R1). R2/R4: state the tuned interior number (wcrit=0.74) in the main text, and add a herding-scale sensitivity analysis to preempt "engineered to manufacture a boundary."

**51. Inconsistent λ endpoints (λ=4 here, λ=7 there).** ✅ ADOPT — **and it has a legitimate cause you should state, not hide** (R1, R3, R4 all converge). Each setting saturates at a different λ: LBF's boundary extrapolates past w=1 for λ≥5, so λ=4 is its *last interior point*; SC/MPE stay interior to λ=7. You're reporting "last interior λ," which is *correct* — but presented as if arbitrary. Fix: report **every setting at λ=3** (apples-to-apples) *and* each setting's last-interior endpoint with λ labeled. R4 flags a concrete table error: Table 4 says MPE 0.78→0.89 (raw Δ=0.11) but labels it "+0.158 excess" (vs a different baseline) — must be made explicit. R3 ranks this "one of the more important fixes" because a reviewer could wrongly cry cherry-picking.

**52. Scope condition: action representation (you were fatigued).** ✅ ADOPT — **council's read: this is the strongest ablation in the paper and it's written apologetically** (R1). The logic is a clean single-variable control (QMIX/IQL flat → PPO on the *same* discrete shell also flat → therefore discretization, not paradigm). Three fixes: (a) **it's misplaced** — it's a scope *condition of the instrument*, belongs in the Method applicability statement (pointer 10), not buried as a transfer-section negative; (b) promote the MPE counter-control ("discrete labels but retains a boundary") — it's a positive control against your own scope condition; (c) preempt the obvious reviewer question with a **resolution sweep** (K∈{5,11,21,51}) if you have it — R2/R3 both say the K=11-only result is a single-resolution observation, and "why not finer bins?" is the first thing a skeptic asks. Also be honest that "four combinations" are *all policy-gradient* (PPO + centralized-critic MAPPO) — a tough reviewer calls that PPO→PPO-with-a-different-critic (R1); preempt it.

---

## A Single Downstream Node Can Nearly Set the Boundary (53–57a)

**53. Why only supply chain? Reasoning up front.** ✅ ADOPT. Unanimous — state it as the *first* sentence of the section (needs a directed-flow DAG; MPE/LBF don't have one), and R3 adds: introduce the DAG back in the Method substrate paragraph so the reader isn't wondering for a paragraph.

**54. Figure 6 hard to read.** ✅ ADOPT. Unanimous and specific: node IDs (n2,n3…) mean nothing — **label bars by topological role** ("feeds retail," "sole supplier to dominant node," "split-path"); the roles *are* the finding, so the legend disappears. R2's stronger version: make panel (a) a **DAG diagram** annotated with each node's wcrit — "a bar chart without topology is the wrong visual form for a topology claim."

**55. This is a supply-chain-only question — say so?** ✅ ADOPT — as a **strength, not a limitation** (R1/R3 both): reframe from "doesn't transfer" to "a falsifiable directed-flow *prediction*" for traffic networks, power grids, multi-echelon logistics. "Predictions are interesting; limitations are boring. Same fact."

**56. Dose-response looks like the affine law.** ✅ ADOPT — **your sharpest observation, and R1 says you didn't follow it far enough.** Single-node sweep (0.586→0.856) ≈ homogeneous sweep (0.586→0.852): same start, same end, both monotone. Reading B (which a reviewer *will* write): if one node reproduces the full effect, your homogeneous λ-sweep is **confounded** — it can't distinguish "λ acts on the team" from "λ acts on n2, other four irrelevant," which would make the affine law *n2's law with four redundant agents attached.* The test is sitting in your data: **overlay the two dose-response curves.** If indistinguishable, the affine law's real meaning sharpens to "the boundary is set by the node whose output most directly reaches demand" — a much stronger claim. R2/R3 concur: fit the same form to both, compare ρ. Right now Fig 6b hides whether the curves coincide. Council: highest-value *new analysis* in the minority section.

**57a. "One node nearly matches... saturation of a continuous law, not a special case."** ✅ ADOPT (sharpen). All four: the striking sentence is followed by an abstract one the reader can't use. Bring the operational reading here (from Discussion): "the team's collapse boundary is not an average over agents — it is set, nearly entirely, by whichever agent's output reaches demand most directly; audit *that* node, not the mean." R3 explains the cryptic phrase concretely (spread-thin vs concentrated λ = two points on one dose-response surface).

---

## Discussion (57b–60)

**57b. "Curvature and loss aversion opposite effects... to our knowledge novel."** 🔵 ADOPT w/ MODIFICATION — **strong statement, and dangerous for a precise reason** (R1): the "novel" finding is the one that **doesn't replicate** — curvature is null on MPE and *opposite-signed* on LBF. So you're claiming novelty for a single-substrate effect your own Table 2 contradicts; a reviewer quotes Table 2 back at you. Fix: attach the novelty to the **decomposition method**, not the finding — "we're not aware of prior work that *separates* curvature from loss aversion in cooperative MARL; doing so reveals a single 'prospect-theoretic' condition can under-report the loss-aversion effect by half." That claim survives all three substrates (R1). R2/R3 concur: scope to "first empirical *boundary-level* decomposition."

**58. "Operational interpretation" — you like it.** ✅ ADOPT (keep + strengthen). All four call it the best paragraph — it's the only place the reader learns *what to do with the number*. R1: add the concrete design action ("if w is capped at 0.8, your λ budget is bounded by where wcrit crosses 0.8 — here λ≈4") and move a compressed version into the *intro* (it answers "why read this" better than anything currently there). R2: cut the "adversarially... degrade coordination" dual-use drama from here (keep it in the Ethics statement).

**59. "Across environments and a second algorithm" → "across environment-algorithm combinations."** ✅ ADOPT. Unanimous — more precise, and avoids implying a full 3×2 cross when independent PPO fails on MPE/LBF. R1's honest-framing caveat: "four combinations, all policy-gradient" is the accurate version.

**60. The (iii) "bracket to calibration" paragraph is opaque.** ✅ ADOPT. Unanimous — the three future-work items are compressed into metaphors ("localizing observable," "bracket to calibration") that were never set up. Split into three plain sentences (R2/R3 give clean rewrites: infer effective λ from behavior to connect to real RLHF-style pipelines). **Delete "None undercuts the central, replicated claim"** — all three who mention it call it a defensive tic ("if the limitations section needs a bodyguard, the limitations are stated wrong," R1).

---

## Conclusion (61)

**61. Stronger ending.** ✅ ADOPT — unanimous and emphatic. Your current conclusion **restates the abstract** verbatim/near-verbatim; the reader's last impression is a summary they've read twice. End on the one idea you want in their head tomorrow — the *operational inversion* ("reward centralization has a *location*, and you can measure yours... The numbers are ours. The instrument is anyone's," R1; R2's and R4's drafts are similar). Don't re-scope in the conclusion.

---

## References (62)

**62. Correction/hallucination sweep.** ✅ ADOPT — high value; R1 and R2 did the actual cross-check and converge on the same findings:

- 🔴 **Hu, Leung, Leung 2019 — wrong metadata.** Real paper is *Shuyue Hu, Chin-wing Leung, Ho-fung Leung, "Modelling the Dynamics of Multiagent Q-Learning in Repeated Symmetric Games: A Mean Field Theoretic Approach," NeurIPS 2019.* Your entry has the wrong first-author initial, the two Leungs swapped, and a reconstructed title — R1 notes this is "the exact signature of an LLM-generated citation," and one bad entry makes a reviewer suspect all of them. **Fix first.** Also the attached claim is wrong (it's large-population repeated games, not networked).
- 🔴 **Mengesha et al. 2025 — direction mischaracterized** (see #6). Their result is non-monotone/partly opposite. Cite as contrast or drop.
- 🟡 **Weng & Lee 2026 — mis-filed.** It's parameter-sharing DQN + representation failure, *not* "decentralized exploration," and it's your closest competitor ("failure boundaries that shift with a control parameter" ≈ your one-line summary). Move to the Phase Transitions paragraph and differentiate deliberately (R1).
- **Missing citations:** Firth (1993); a bootstrap citation (Efron 1979 / Efron & Tibshirani 1993) + which bootstrap + resample count; LBF origin (Albrecht & Ramamoorthy 2013) alongside Christianos.
- **λ≈3 "commonly used" is imprecise** — the canonical Tversky–Kahneman value is λ≈2.25, not 3 (R2/R4). Reframe λ=3 as "a moderate value near the canonical range" (R4's P1-7 adds a clean bracketing sentence: 2.25 lies between your λ=2 and λ=3 grid points).
- **Cleanup:** Rabin → journal cite (*Econometrica* 68(5):1281–1292); consistent arXiv formatting; PettingZoo has a NeurIPS venue; the Juhan/Jaan Aru "duplicate" is correct (two people) — leave it.
- **R3's standing caveat:** the very recent arXiv IDs (Yamaguchi, Weng, Mengesha, Capannoli, Hussain, Barfuss-2025, Zhao, Hammond) can only be verified by *you* opening each URL. R1/R2 verified several as real; but you must personally confirm title/authors/abstract for the ones central to your framing before submission. **This is non-optional given how load-bearing Yamaguchi and Capannoli are.**

---

## The seven items your pointers missed (R1 + R4 — council ranks these ABOVE the 62)

These are correctness-level, referee-quotable, and none is in your list. The council's unanimous recommendation is to fix these *first*.

1. **Inverted abstract decomposition** (your #3, but the *labels* are backwards, not just underexplained). P0.
2. **Run-accounting gap** — 450k/900k/finer-grid/entropy runs are reported in the paper but absent from the Supplement inventory that sums to 1,524; and §2 references "the 600k row of the main-paper heatmap" — *there is no heatmap in the main paper* (leftover text from a prior draft). Your headline log-linear fit rests on runs the inventory doesn't contain. **R1 calls this the single most reviewer-lethal item.**
3. **Entropy dissociation uses two evidentiary standards** (SC "rises" on overlapping CIs; MPE "flat" on overlapping CIs) — your #45.
4. **Collapse label is nearly a point mass** — −127.1 sits ~1.0 above the −128.1 floor; either explain the redundancy or a reviewer reads it as tuned.
5. **LBF collapse threshold is never stated as a number** — anywhere. Your whole contribution is *pre-specified labels*, and one of three is a paraphrase.
6. **MAPPO implementation never named** — half the results run on an unnamed learner (SB3 has no MAPPO).
7. **"Checked premise" is inferred from the result it explains** — the λ-half of the SNR premise is a circle unless you split it and admit the two failed probes (your #44).

Plus R4's directly-usable finds: **"p<10⁻³" is contradicted by your own Table 3** (λ=7 is p=1.5×10⁻³) — a one-word falsifiable claim a referee quotes; and **"non-overlapping by 0.221"** should be ~0.217 from the rounded CIs — verify or soften.

---

## Exact passages that read as extremely AI-generated

You asked specifically for sentences/paragraphs that look "extremely AI generated." These are the strongest tells in the actual submission text — the pattern is **triads, "X, not Y" / "X rather than only Y" tails, colon-reveals, metaphor topic-sentences, and terse reassurance closers.** In roughly descending order of how strongly they signal it:

1. **Related Work, opening of "Behavioral preferences":** *"A thin bridge connects human preference structure to collective learning dynamics."* — Metaphor-as-topic-sentence stating your own gap. R1 independently flagged this one. The single most AI-sounding line in the paper.
2. **Section title:** *"The Instrument Travels; So Does the Direction."* — Semicolon-callback title; almost no human author titles a results section this way.
3. **Discussion / Future Work triple:** *"(i) From one substrate to many. ... (ii) From calibrated to predictive. ... (iii) From bracket to calibration."* — The parallel "From X to Y" heading triple is a very strong LLM structural signature (and (iii) is even semantically muddled, per #60).
4. **Intro:** *"We therefore build the ruler first, and treat it as the paper's central object. Our protocol is simple."* — "the ruler" metaphor + "Our protocol is simple" declarative.
5. **Intro:** *"Turn w down and an agent's private performance drowns out its contribution; cooperating earns no learnable signal and the lazy policy wins."* — Imperative opener + parallel clauses + punchy "the lazy policy wins."
6. **Intro:** *"That is precisely what makes it useful: it tells a designer where their own system crosses into collapse."* — "That is precisely what makes it useful:" colon-reveal is a hallmark cadence.
7. **Generalization opener:** *"The first thing that must generalize is the instrument... It can:"* — Setup-then-colon-reveal ("It can:") is very LLM.
8. **Minority section:** *"One node, biased hard enough, nearly matches the entire network's midpoint requirement; the homogeneous result is the saturation of a continuous law, not a special case."* — Comma-delayed qualifier + "not a special case" triadic tail.
9. **Discussion closer:** *"None undercuts the central, replicated claim."* — Terse reassurance closer (also flagged in #60).
10. **Abstract + Conclusion mirror:** *"...a measurement protocol for reporting where a cooperative learning system crosses into collapse, rather than only asserting that centralization helps."* / *"prospect-theoretic reward transformation is its first application."* — "X rather than only Y" and "is its first application" are recurring LLM tails; their near-verbatim mirroring across abstract and conclusion is itself a tell.
11. **Method:** *"We use 'phase boundary' in this operational sense, an empirical crossing in learning dynamics, not thermodynamic criticality."* — appositive-plus-negation ("X, not Y") rhythm, used repeatedly throughout the paper (it's the paper's dominant sentence shape — the *repetition* of this exact cadence across sections is the giveaway more than any single instance).

Practical note: none of these is *wrong*, and an AAAI reviewer rarely writes "this sounds AI-generated" in a review. The risk is subtler — the uniform cadence (especially the "X, not Y" appositive, which appears dozens of times) makes the prose feel *machine-even*, which compounds the "impressive but flat" reaction below. Breaking up that one sentence shape in maybe 15–20 places would do more for perceived voice than any single rewrite.

---

## Your actual question: will the paper excite the reviewer, or bore them?

**All four reviewers give the same answer, and the council endorses it:** the *result* is exciting; the *current reading experience* is flat, and the fix is subtraction, not addition.

R1 put it most sharply: a reviewer will be **"impressed and bored at the same time, and will write a lukewarm accept."** The paper is organized around *not being wrong* instead of around *being interesting*, and that has a measurable cost — count the hedges (abstract "substrate-specific unless replicated," the claim ledger appearing *three* times, "we claim nothing about it," "not a tested generalization," "None undercuts the central, replicated claim"). A reviewer reading that many hedges doesn't think "rigorous," they think "this author is expecting an attack" — and defensiveness reads as weakness even when the work is strong.

Meanwhile the things that would actually excite a reader are **underplayed or buried**:
- *"One node sets the boundary"* — your headline — is on page 6, in a section that opens by apologizing for being single-substrate.
- *"Loss aversion and curvature push opposite ways, so reporting prospect theory as one knob halves your measured effect"* — a real warning to a real community — is a subordinate clause.
- *`wcrit = 2.44 − 0.138 ln T`* (each compute doubling buys ~0.10 of centralization) — remarkable — is a caveat paragraph.
- *KTO ships a loss-aversion coefficient in a production alignment objective* — your entire motivation — is a parenthetical.

**So: not boring, but flat.** The reader's honest reaction to the current draft is "careful work, hard to argue with, I'm not sure what happened."

What flips it (council consensus ordering):
1. **Fix the seven correctness items** (esp. the inverted abstract and the run-accounting gap). These are the difference between "rigorous" and "sloppy in a way that makes me doubt the rigor."
2. **Consolidate all scoping into one claim ledger and delete every other hedge.** Highest-leverage change in the paper — costs nothing, weakens no claim, converts the tone from anxious to authoritative.
3. **Restructure around the number, not the protocol.** Open with `wcrit = 0.666 [0.635, 0.694]`, then "here's how, and here's what moves it." Nobody gets excited about an instrument; they get excited about a reading.
4. **Buy the space by cutting the affine law to the supplement** (#48) — it funds pointer 10's scope paragraph, the curvature definition, and the fixed Figure 1.
5. **Run the two cheap experiments that turn predictions into results:** SC-MAPPO at 3–4 horizons (makes the metastability figure real, #38) and the single-node-vs-homogeneous overlay (turns #56 from a confound into your sharpest finding).

Do those and the reviewer's summary moves from *"careful measurement paper, modest contribution, accept"* to *"they found that one agent's reward design sets the whole team's collapse point, and they measured it properly"* — which, as R1 notes, is a sentence people cite.

---

### One-line index of the rare disagreements (where you actually have to choose)

- **#4** (drop "supply-chain" from abstract): R2 dissents — keep the *noun*, cut the *hedge*.
- **#33** (inventory redundant): R3 dissents — it's a manifest, but the accounting is broken (fix, don't just trim).
- **#36** (Fig 1 missing curves): you misread — panel (b) has all four; fix the caption only.
- **#38** (4-panel Fig 1): the SC-MAPPO panel has no data — modify or run it; don't build as specified.
- **#47** (ledger → paragraph): table-vs-prose is taste; deleting the two *duplicate* ledgers is the real win.
- **#48** (affine law): R1 says cut to supplement; R2/R3 say keep-but-rename. Council leans cut.

Everything else: the four reviewers agree with each other and, mostly, with you.
