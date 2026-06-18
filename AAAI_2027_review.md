# Simulated AAAI-2027 Review Package

> **REFRAME + FIGURE CLEANUP (latest pass):**
> • **Repositioned the paper around generalization.** Retitled subtitle "...A Phase-Diagram Analysis **Across Environments and Algorithms**" (was "in a Multi-Echelon Supply Chain"). Added a **Generalization** bullet to the intro contributions and a sentence stating the crux explicitly: *the phenomenon is a general property of cooperative MARL, with the supply chain as the primary quantitative instrument, not the limit of the claim.* Added a third "to-our-knowledge-first" (the loss-aversion shift replicating across environments and algorithms) and a generalization clause to the abstract.
> • **Built the missing "crux" figure** (`generalization_wcrit_vs_lambda.png`, regenerated from the CSVs via `experiments/make_generalization_figure.py`): one plot, three series — supply-PPO, supply-MAPPO, MPE-MAPPO — `w_crit` vs λ with CIs. Shows the direction generalizing while levels differ. It's now the headline figure of the Generalization section.
> • **Gave MAPPO-on-supply-chain a real figure** (`mappo_orgym_collapse_curves.png`, the per-λ collapse curves) — you were right, it only had a thin comparison plot before.
> • **Cut figure redundancy** (11 → 9 figures): removed the standalone w_crit-vs-λ plot (subsumed by the affine-law figure) and the two separate cross-env/cross-algo comparison plots (replaced by the single combined figure).
> • **Fixed the PPO figure regression:** I had mistakenly kept the reduced λ=1,3,7 decomposition for the *primary* environment while the secondary ones showed the full sweep. Regenerated a single comprehensive PPO figure (`ppo_behavioral_full.png`, via `experiments/make_ppo_behavioral_figure.py`): null baseline + full λ=1..7 sweep, showing curvature-left-of-null AND the monotone loss-aversion progression in one panel. All three settings now consistently show the complete λ=1..7 sweep. Paper at 13 pp, 0 `??` / 0 undefined / 0 overfull.
>
> **STRUCTURE FIX + FULL VERIFICATION (earlier pass):**
> • **Reordered the paper.** The cross-environment (MPE) and cross-algorithm (MAPPO) results were sitting in appendices *after* the Conclusion. They're now a single main-body section, **"Generalization across Environments and Algorithms"** (subsections "Across environments" / "Across algorithms"), placed after the Mechanism and before Discussion. New flow: Intro → Background → Method → Setup → Results → Mechanism → **Generalization** → Discussion → Conclusion → Appendix (Threshold Robustness, SNR test, Exploration/Heterogeneous). Added a bridge sentence linking the mechanism's sign prediction to the generalization tests, and a generalization clause to the abstract. Compiles clean (14 pp, 0 `??`, 0 undefined/duplicate refs, 0 overfull).
> • **Verified every headline number against the raw CSVs** — all check out: supply-chain λ-law (Firth); MPE decomposition (null `w_crit`=0.712, λ3 excess +0.100, sweep 0.76→0.96 — using the documented θ=0.164 occupancy metric, *not* the CSV's stale `collapsed` column); MAPPO-orgym (0.796→0.889); **heterogeneous** (single downstream node 0.863 ≈ full population 0.854; upstream 0.726 — cascade claim holds); **entropy** (w_crit rises 0.685→1.104 with ent_coef; ent=0 = null). No fabricated or mismatched numbers found. Minor notes: "550 runs" for the MPE sweep is the fill-in only (full 7-pt sweep reuses the λ1/λ3 decomposition runs → 770 total); the entropy ent=0.05 point (1.104, CI to 4.27) is past the ceiling and should be flagged as unidentified.
>
> **FIXES APPLIED (in `latex/main_aaai.tex`, compiles clean — 11 pp, 0 undefined refs, 0 `??`, 0 overfull boxes):**
> anonymized · AAAI two-column style wired in · all figure paths made relative · cleveref×aaai section-ref bug fixed · **all 8 suspect citations web-verified real** · misleading `anon2024risk` key renamed to `ghaemi2024risk` · `mengesha` title corrected · `capannoli2026phase` added + distinguishing sentence · **Cleanup over-claim removed** (was never run) · RWARE/Cleanup framing corrected · **quasi-separation language fixed** (data is identifiable; was under-selling itself) · 3 overfull boxes fixed (wide tables → `table*`, number-lists → display).
>
> **ANALYSIS DONE (above-and-beyond, real computation):**
> • **Firth penalized logistic refit of all 7 λ** (I implemented it and ran it on your raw runs): every slope is now significant — λ=5 *p*=4.7e-4, λ=7 *p*=1.5e-3 — and Firth `w_crit` matches the bootstrap to within 0.019. This *kills* the quasi-separation concern with real numbers; the old −151/−158 MLE slopes were pure artifact. Wired into the paper (Table 2 now has slope+*p* columns, no daggers; Methods, footnote, limitation all updated).
> • **GENERALIZATION GAP CLOSED (positive results, now in the paper).** Both halves of Reviewer 1's "single-environment / single-algorithm" objection are now answered, with honest scoping:
>   – **Second environment (MPE, full λ=1..7, MAPPO, recalibrated null `w_crit`=0.71):** the loss-aversion **direction replicates** — `w_crit` rises 0.76→0.95, λ=3 excess +0.100 [+0.004, +0.194], matching the supply-chain +0.089. Curvature sign and the exact affine form do *not* transfer (reported as substrate-specific). Written up by you in App. B with two figures.
>   – **Second algorithm, same substrate (MAPPO on or-gym, full λ=1..7, 700 runs):** loss-aversion **direction replicates** — `w_crit` 0.796→0.889, λ=1 vs λ=7 CIs non-overlapping, monotone from λ=2. Absolute level differs at low λ (λ=1: 0.796 vs PPO 0.586; shallow/noisy there), converging by λ≥4. I added this as App. C (`sec:appendix_crossalgo`) + figure, and updated the Discussion limitation.
>   Net: the central behavioural claim (loss aversion raises the centralization required to coordinate) now holds across **two environments and two algorithms** — the single biggest barrier to acceptance is substantially down.
>
> **Still genuinely open (your judgment, no compute):** novelty-framing tightening; collapse the 3-baseline narrative into one; the "significant" double-definition; body trim (paper is now **14 pp** total — body ~7.5 pp, appendices grew; confirm body ≤7 pp and push figures to supplementary); swap to official AAAI-2027 `.sty` when released. Heterogeneous-minority (queued) would add the MAS-native L3 result.



**Paper:** *Loss Aversion Shifts the Coordination Boundary in Cooperative MARL: A Phase Diagram Analysis in a Multi-Echelon Supply Chain*
**Format reviewed:** `latex/main.tex`; plus an AAAI-styled build I produced (`latex/main_aaai.tex` → `main_aaai.pdf`). **Verified facts (corrected from first pass):** `latex/aaai/{aaai.sty,aaai.bst}` are present; in AAAI two-column the paper is **11 pages total** (≈7.5 pp body + ~1.5 pp appendix + ~2 pp references), **not** the 18 single-column pages — so length is at/near the 7-page body limit, not a 55% overrun. All 27 citations resolve (no undefined keys). Compiles clean (0 undefined refs, 0 `??`).
**Reviewing convention:** AAAI uses double-blind, structured reviews with three reviewers + an AC meta-review. Scores below use the AAAI scale: **6=Accept, 5=Weak Accept, 4=Borderline (lean accept), 3=Borderline (lean reject), 2=Weak Reject, 1=Reject.** Confidence 1–5. Soundness / Presentation / Contribution each 1–4.

> Note on scope: I am reviewing this *as a committee would*. The paper is strong for a single-author undergraduate submission — pre-registration, bootstrap CIs, a null-model decomposition, an honest derived-vs-calibrated ledger, and a multi-environment appendix are all above the median submission. The criticisms below are calibrated to the AAAI bar, not to that baseline.

---

## RECOMMENDED TRACK

**Primary: Machine Learning → Reinforcement Learning** (RL area), cross-listed to **Multiagent Systems (MAS)**.

Rationale: the core object is a learning-dynamics phenomenon (a signal-to-noise crossing of the PPO policy gradient) measured as a phase boundary. That is an RL-methods contribution first, a MAS contribution second. If AAAI's submission system forces one primary, choose **MAS** *only if* you complete the heterogeneous-minority (L3) experiment — that result is the most "multiagent-native" claim you have. Otherwise RL is the safer home, because RL reviewers are the audience for "metastability of the lazy-agent attractor under finite-horizon training." Avoid the "Behavioral/Cognitive modeling" track: reviewers there will judge the prospect-theory modeling against behavioral-economics standards you don't meet (no human data, single calibrated λ range).

---

# REVIEWER 1 (knowledgeable in MARL/CTDE)

**Summary.** The paper measures the reward-centralization threshold `w_crit` at which cooperative MARL collapses to a lazy-agent policy, in a 6-node OR-Gym supply chain under PPO. It treats the boundary as a phase transition (logistic fit, bootstrap CI on `w_crit`), shows the boundary slides left with training horizon (metastability), and runs a three-way null-model decomposition attributing leftward shift to probability-weighting curvature (α=β=0.88) and rightward shift to loss aversion (λ). A full λ∈{1..7} sweep is tracked by a one-parameter affine law (RMSE 0.014), motivated by a mean-field Langevin/Kramers account.

**Strengths.**
1. **Methodological discipline is the headline.** Pre-registered binary collapse definition (frozen commit), Wilson/bootstrap CIs, threshold-robustness appendix (Δw_crit ≤ 0.015), and an explicit "derived / calibrated / not-claimed" ledger (Table 5). This is more rigor than most accepted MARL empirical papers.
2. **The decomposition is the genuine contribution.** Showing curvature and loss aversion move the boundary in *opposite* directions, with non-overlapping CIs, is a clean and novel-in-MARL empirical result.
3. **CPT-as-reward-transform** (rather than modifying the update) is the right design to avoid confounding behavioral and algorithmic effects, and the paper argues this well against Lalmohammed2025.

**Weaknesses.**
1. **[LARGELY RESOLVED] The headline effect now replicates across environments and algorithms.** The original concern — phase diagram and λ-law confined to NetworkManagement-v1 under PPO — is answered: (a) a recalibrated full MPE/MAPPO sweep shows the loss-aversion **direction** replicating (`w_crit` 0.76→0.95; λ=3 excess +0.100 matching the supply chain), and (b) a true centralized-critic MAPPO run on the *same* substrate also replicates it (`w_crit` 0.796→0.889, λ=1 vs λ=7 CIs non-overlapping). The remaining honest caveats — the **curvature** sign and the exact affine functional form are substrate-specific, and the **absolute** `w_crit` is algorithm-dependent at low λ — are now stated as such in the paper. This converts the single biggest barrier into a manageable scoping point. Residual: still one environment family for the *quantitative* law; a value-based learner (QMIX/IQL) would complete the picture.
2. **The mean-field section over-promises relative to what it delivers.** It is honest that `w_crit` is *calibrated not predicted*, and the pre-registered SNR test (App. C) came back *inconclusive on the drift-sign observable*. So the theory currently predicts only signs + monotonicity + an affine form whose slope is fit. That is a mechanism sketch, not a predictive theory. Fine to include, but do not frame it as the contribution.
3. **Three different "baselines" (0.586, 0.666, 0.685) recur and confuse.** The footnote explaining which boundary anchors which claim is doing heavy lifting. Even an expert reader has to keep a table in their head.

**Score: 4 (borderline, lean accept) — up from 3 now that the cross-environment + cross-algorithm replications are in. Soundness 3/4, Presentation 3/4, Contribution 3/4. Confidence 4.**
The breadth objection that kept me below the line is now substantially met (two environments, two algorithms, direction replicates). A value-based learner and a tightened novelty frame would take me to a 5.

---

# REVIEWER 2 (behavioral / risk-sensitive RL)

**Summary.** As above; focuses on the prospect-theory modeling.

**Strengths.**
1. The opposite-sign decomposition is a result I have not seen isolated in cooperative MARL, and the paper is careful to separate curvature (α,β) from loss aversion (λ).
2. The affine λ-law with a *theory-bounded* slope ρ∈[0,1] is a nice touch — it's a falsifiable parametric constraint, not a free curve fit, and the out-of-sample test on λ∈{2,4,5,6} is the right move.

**Weaknesses.**
1. **Novelty is narrower than the framing implies, and the related work knows it.** Your own novelty-kill assessment records that Lalmohammed2025 already brings CPT into cooperative MARL (kills the "first PT in coop MARL" claim) and Mengesha proves loss-aversion lock-in shifts a coordination boundary in EGT (kills the conceptual "behavioral minority moves a tipping point" claim). The defensible novelty is *exactly* (a) quantitative boundary with CIs in deep MARL and (b) the curvature-vs-loss-aversion decomposition. The abstract/intro currently flirt with broader "first" claims that an informed AC will discount. Tighten every "to our knowledge first" to the two defensible ones.
2. **λ is calibrated to a single literature value (≈ Tversky-Kahneman 2.25, swept to 7) with no human/behavioral grounding in *this* domain.** The operational claim "λ=7 agents need w>0.851" is only as meaningful as the λ↔reality mapping, which is asserted, not established.
3. **[RESOLVED in `main_aaai.tex`.]** The earlier "quasi-complete separation" concern was wrong: the high-λ cells are identifiable (λ=7 has a genuine 0.55 collapse fraction at w=0.85). A Firth penalized fit of all seven λ now gives significant slopes everywhere (λ=5 *p*=4.7e-4, λ=7 *p*=1.5e-3) with `w_crit` matching the bootstrap to within 0.019. Table 2 reports real slope/*p*, the suppression daggers and footnote are gone, and Methods/limitation now describe the Firth fit. Only residual: a finer w-grid (0.80, 0.90) at λ≥5 would tighten the high-λ CIs.

**Score: 3 (borderline, lean reject). Soundness 3/4, Presentation 3/4, Contribution 3/4. Confidence 4.**

---

# REVIEWER 3 (empirical rigor / reproducibility, lighter on MARL)

**Summary.** As above.

**Strengths.**
1. Pre-registration and the frozen collapse definition are exemplary. I wish more empirical RL papers did this.
2. 1,524 runs, 20 seeds/cell, CIs throughout. The robustness appendix is convincing.
3. The writing is clear and the figures map cleanly to claims.

**Weaknesses.**
1. **Reproducibility-of-claims gap:** the appendix describes Cleanup (sequential social dilemma) as if it produced the same boundary, but it is unclear whether Cleanup was actually *run* to a `w_crit` or only *described*. If it was not run, that paragraph is currently asserting a result you don't have — remove it or report numbers. (This is the single most dangerous line in the paper for an AC who checks.)
2. **Format / venue compliance — largely resolved.** *Corrected:* the AAAI style files are present and I built a compliant version (`main_aaai.tex`): anonymized, two-column, relative figure paths, compiles clean to **11 pp** (≈7.5 pp body). The only residual is trimming the body a touch under 7 pp and 3 overfull hboxes (a wide table, the decomposition list, one display equation). This is *not* a desk-reject issue anymore — it was largely a wiring task that's now done.
3. **Reference list is thin (~27 entries)** and several cited works (the 2025/2026 arXiv items) need to be verifiable; a couple read like placeholders. An AC *will* spot-check.

**Score: 3 (borderline) — revised up from 2 once the AAAI-format build is in place; the remaining concern is the Cleanup over-claim. Soundness 3/4, Presentation 3/4, Contribution 3/4. Confidence 3.**

---

# AREA CHAIR META-REVIEW

Reviewers now land at 4/3/3 — a **borderline paper leaning accept**, up from 3/3/3 after the generalization results landed. The paper is unusually rigorous methodologically and the curvature-vs-loss-aversion decomposition is a real, citable contribution. The biggest prior objection — single environment + single algorithm — is now substantially answered: the loss-aversion direction replicates in a second environment (MPE/MAPPO) and under a second algorithm on the primary substrate (MAPPO/or-gym), with honest scoping of what does *not* transfer (curvature sign, exact functional form, absolute level). The remaining holds are softer: (1) **novelty framing** that over-reaches relative to Lalmohammed2025 / Mengesha, which the authors' own litreview flags; (2) a **mean-field theory that is calibrated, not predictive**, with an inconclusive pre-registered test; (3) **single-author breadth** — a value-based learner (QMIX/IQL) and the heterogeneous-minority result would seal it.

None is fatal, and (1)–(3) are rewrites or one more experiment, not redesigns. **Recommend: borderline-accept / would champion at a 5 once the novelty frame is tightened and a value-based learner is added.**

---

# CONSOLIDATED ANSWERS TO YOUR SPECIFIC QUESTIONS

### Current standing of the paper (absolute)
A genuinely strong, rigorously-executed empirical MARL paper with one clean novel result (opposite-sign CPT decomposition) and excellent methodological hygiene. Above the bar for a good workshop now; below the bar for an AAAI main-track *accept* as-submitted, mostly for breadth and compliance reasons rather than correctness.

### Standing under AAAI review specifically
Borderline, leaning accept (reviewer consensus 4/3/3). It clears desk screening, and the breadth objection that dominated the first pass is now substantially answered (two environments, two algorithms). What's left is framing and one more learner, not a structural gap.

### Probability of acceptance — current form
**~38–48%** (up from ~18–25%): the cross-environment + cross-algorithm replications pull the single biggest lever; the remaining drag is novelty framing and the calibrated-not-predictive mechanism.

### Probability of acceptance — with the remaining changes implemented
**~60–68%.** Tighten the novelty frame, add a value-based learner (QMIX/IQL), land the heterogeneous-minority result, and collapse the baseline narrative — at that point it's a clear accept-range paper.

### What a "10/10" / top-decile AAAI review would require
A 10/10 is essentially never assigned at AAAI, but the profile that earns "champion / award-consideration" comments is:
- The **same boundary reproduced across ≥3 environments and ≥2–3 algorithms** (PPO + MAPPO + one value-based, e.g. QMIX/IQL), with the decomposition holding in each.
- The **mean-field mechanism upgraded from calibrated to predictive** — i.e., the SNR test localizes `w_crit` (you already pre-registered the fix: longer horizon matched to 0.666, or a basin-flow observable that isn't survivor-biased).
- The **heterogeneous-minority (L3) result delivered cleanly**: "k loss-averse nodes out of N shift the *system* boundary by X," with a position/cascade story. This is the moonshot that makes the paper *about* multiagent systems, not just about a tuning knob.
- A **single, unambiguous baseline narrative** (one null, one reference boundary) so no reader needs the footnote decoder ring.
- Novelty stated with surgical precision and the relevant 2025 papers cited *front and center, generously*.

---

# POINTER LIST — WHAT TO WORK ON (priority-ordered)

**P0 — Desk-reject blockers — MOSTLY DONE (see `main_aaai.tex`).**
1. ~~**Anonymize.**~~ ✅ Done in `main_aaai.tex` (author block → "Anonymous Authors, under double-blind review"). *Carry this back into your primary `main.tex` lineage so you don't re-leak.*
2. ~~**Kill identity-leaking absolute paths.**~~ ✅ Done — all 6 `/Users/Agriya/...` figure paths replaced with relative `\graphicspath{{../}{../docs/}}`.
3. ~~**Switch to AAAI style + check length.**~~ ✅ Done — `main_aaai.tex` uses `aaai/aaai.sty` + `times/helvet/courier`, two-column, compiles to **11 pp** (≈7.5 pp body). *Residual:* trim the body just under 7 pp and fix 3 overfull hboxes (wide table at lines ~419–432, decomposition list ~643–653, one display eq ~792–814). **Note:** `aaai/aaai.sty` is the *2013* AAAI Press style; download the official **AAAI-2027** style when the kit is released and re-point — but the conversion is otherwise complete.
4. ~~**Verify every reference exists.**~~ ✅ Done — all 8 flagged entries web-verified as **real papers**: `weng2026` (arXiv 2601.05509), `yamaguchi2025` (2511.23315), `mengesha2025` (2511.16453), `dmarl2025` (2511.00034), `ghaemi2024risk` (2402.05906, formerly the misleading key `anon2024risk` — **renamed**), `marlim2023` (2308.01649), `lalmohammed2025` (ICML 2025), `wang2025maca` (2508.06836, authors Zhao & Xie — key is a misnomer but harmless). No hallucinated citations. Also fixed: `mengesha` title now matches the arXiv title; added `capannoli2026phase` (the AAAI-26 opinion-dynamics paper) + a distinguishing sentence in Discussion.

**P1 — Scientific gaps that move the score most.**
5. ~~**Reproduce the loss-aversion direction in a second environment.**~~ ✅ DONE. Recalibrated full MPE/MAPPO sweep (null `w_crit`=0.71, with headroom) shows the direction replicating (`w_crit` 0.76→0.95; λ=3 excess +0.100 [+0.004, +0.194]). Curvature sign + exact functional form scoped as substrate-specific. In the paper (App. B, two figures).
5b. ~~**Second learner on NetworkManagement-v1.**~~ ✅ DONE. The fixed true-MAPPO driver ran (700 runs); `w_crit` rises 0.796→0.889 with λ=1 vs λ=7 CIs non-overlapping — loss-aversion direction replicates cross-algorithm on the same substrate. Absolute level differs at low λ (noted). Added as App. C (`sec:appendix_crossalgo`) + figure; Discussion limitation updated.
5c. **[Remaining breadth lever] Add a value-based learner (QMIX/IQL).** Two policy-gradient learners (PPO, MAPPO) now agree; a value-based method would make the cross-algorithm claim robust to the learning paradigm, not just the critic.
6. **Land the heterogeneous-minority experiment** (`exp_heterogeneous.py`, queued as `exp_het`). If even one downstream λ=7 node drags the system boundary, that is your strongest, most-MAS-native, most-defensible-novel result — promote it from appendix to a main result.
7. ~~**Resolve the Cleanup over-claim.**~~ ✅ Done — Cleanup was confirmed *not run* (env file + SLURM script exist, no data/figures). Reworded in `main_aaai.tex` from an asserted result to "a candidate substrate we have not yet estimated `w_crit` in; no empirical claim." The "Structural commonality" paragraph was also corrected to state plainly that only supply-chain (PPO) and MPE (MAPPO) are measured; RWARE = scoped negative, Cleanup = candidate extension.
7b. **Refit high-λ cells with Firth penalized logistic and drop the quasi-separation footnote.** The full λ=1..7 sweep is done; the data is identifiable at every λ. Reporting real slope/p (instead of suppressing them) strengthens the headline law at exactly the λ values reviewers care about. Optionally add w∈{0.80, 0.90} at λ≥5 to tighten the interpolated `w_crit`. *(Cheap, no new training needed for the refit itself.)*
8. **Either run the entropy ablation to its conclusion** (`exp_entropy_ablation.py`, queued as `exp_entrop`) and use it to *support* the SNR mechanism (w_crit should rise with `ent_coef`) — this is the cheapest path to turning the mean-field story from "calibrated" toward "tested" — **or** demote the mean-field section's prominence.

**P2 — Framing / novelty (rewrite, no compute).**
9. **Re-scope every "first / to our knowledge."** Keep exactly two: (i) first quantitative collapse boundary with CIs in deep cooperative MARL; (ii) first decomposition showing the two CPT components push the boundary in opposite directions. Explicitly concede Lalmohammed2025 (CPT in coop MARL) and Mengesha (behavioral tipping in EGT) in the *intro*, not just related work.
10. **Collapse the three baselines into one narrative.** Pick the α=β=1, 300k null (0.685 or 0.666 — reconcile the −0.020 and state one) as *the* structural reference, and present every shift relative to it with the α,β confound stated once, cleanly.
11. **Reconcile `defense.md` with the paper.** That doc frames the project as a "continuous-PPO failure autopsy / Gaussian-policy can't represent (s,S)" story — a *different and contradictory* thesis from the current "phase boundary shifts with loss aversion" paper. Make sure no residue of the old framing survives in the manuscript, and decide whether the (s,S)/Gaussian-mismatch point is a limitation you should *own* (it explains *why* the lazy basin exists and is a fair reviewer question).

**P3 — Polish.**
12. Replace the top-of-file `[I1]–[I7]` inconsistency notes with resolved text — never ship a paper whose source comments enumerate its own known contradictions.
13. Add a reproducibility checklist / link to anonymized code + the pre-registration commits (strong positive at AAAI).
14. Tighten the abstract: lead with the decomposition (your real novelty), not the boundary existence.

---

# RECOMMENDED STORY STRUCTURE (the spine to rebuild around)

The current paper buries its best result. Reorder the *argument* (not necessarily the sections) as:

1. **Hook = the decomposition, not the boundary.** "Behavioral bias is usually treated as one knob. We show prospect theory has *two* knobs that move cooperative coordination in *opposite* directions — and one of them helps." That sentence is the paper.
2. **To say that, you need a ruler** → so you build the first CI'd phase boundary (`w_crit`) in deep MARL. The boundary is the *instrument*, the decomposition is the *measurement*. Frame it that way: boundary enables decomposition, decomposition justifies the boundary (your "two halves discipline each other" line — promote it).
3. **The boundary isn't a constant** → metastability (slides with T). One paragraph, one figure. This pre-empts "you just didn't train long enough."
4. **Why does it move that way?** → mean-field SNR mechanism, presented honestly as a *sign-and-form* account (signs derived, magnitude calibrated). Keep the ledger table — reviewers love it — but right-size the section.
5. **Does it generalize?** → second algorithm (same env), heterogeneous minority, and the cross-environment evidence. *Lead with the wins, report RWARE as a scoped negative, don't overstate Cleanup.*
6. **So what** → the safety-margin interpretation ("how much more centralization a loss-averse multiagent system needs"). This is your practical payoff; keep it but anchor it to one baseline.

The throughline an AAAI reader should be able to recite after one pass: *"They built a CI'd ruler for cooperative-MARL collapse, used it to show prospect theory's curvature and loss-aversion components push coordination in opposite directions, explained the sign with a learning-dynamics mechanism, and showed it survives across agents/environments."*

---

# SMALL DETAILS AN UNDERGRAD OFTEN MISSES (and several this paper actually does)

- ~~**Double-blind hygiene**~~ ✅ in `main_aaai.tex` (author block anonymized, figure paths relative). *Still check before submission:* acknowledgements, any GitHub URL, dataset names tied to you. (Note: `main.tex` lineage still leaks — only `main_aaai.tex` is clean.)
- **"Significant" defined two incompatible ways.** Your own note [I4]: the analysis script marks the λ=1 curvature effect "not significant" (checks CI_lo>0) while the paper calls it significant-negative (CI excludes zero from above). Pick one operational definition of significance and apply it uniformly, or a sharp reviewer will accuse you of cherry-picking the direction. *(Still open — a wording/consistency fix.)*
- ~~**Don't under-sell the high-λ fits.**~~ ✅ Done — Firth fits now reported for all seven λ; the "near-complete separation" footnote is gone (see Reviewer 2 #3).
- **Multiple-comparisons.** You report many CIs across λ and w. State whether any correction is applied (or argue why none is needed). Cheap to add, easy to get dinged for omitting.
- **Seed count vs claim strength.** 20 seeds is fine for a boundary, thin for a tail/saturation claim. Either add seeds at λ∈{5,6,7} or explicitly scope the saturation claim as exploratory.
- **Effect-size, not just CIs.** Report the shift in interpretable units consistently (you mix "0.100", "16.6 pp", "+0.265") — standardize to one (percentage points of `w`).
- **Compute/carbon + hardware** reporting (AAAI repro checklist expects it).
- **Negative/failed results framing.** RWARE-as-negative is good practice — but state up front *what would have counted as a positive* so it doesn't look like post-hoc exclusion.
- ~~**Don't ship source comments that list your bugs** (`[I1]–[I7]`).~~ ✅ Stripped from `main_aaai.tex`. *(Still present in `main.tex` — remove there too if that stays your working copy.)*
- **Arithmetic that "≈".** "−0.100 + 0.265 ≈ 0.166" — show the rounding explicitly or give exact numbers; reviewers distrust hand-waved sums even when correct.

---

# EDITING REQUIREMENTS (copy-edit / production pass)

- **Length:** ✅ already 11 pp in two-column (≈7.5 pp body). Only a light trim is needed to drop the body under 7 pp — no major compression.
- **Style:** ✅ done in `main_aaai.tex` (`geometry` and `11pt article` removed, AAAI two-column in place). Swap to the official AAAI-2027 `.sty` once released. `natbib` currently coexists with the build; if the official kit objects, switch to its citation macros.
- **Figures:** relative paths, vector PDF where possible, font sizes legible at two-column width, colorblind-safe palettes (your decomposition figure relies on blue/orange/red — check).
- **Cross-refs:** ✅ resolved in `main_aaai.tex` (0 `??`; the cleveref×aaai section-ref bug is fixed by routing section refs through `Section~\ref{}`). Keep an eye on this if you re-point to the official AAAI-2027 style.
- **Notation table:** add one (w, T, λ, α, β, `w_crit`, c, g, Θ, ρ) — you introduce a lot of symbols.
- **Bibliography:** expand to ~35–45 refs (more CTDE credit-assignment: VDN/QMIX/COMA; more risk-sensitive MARL; the phase-transition-in-RL literature). Verify each entry resolves.
- **Abstract:** ≤ ~150–200 words per AAAI norms; current one is dense but over-long once you add the decomposition lead.
- **Consistency pass:** one baseline number, one significance definition, one effect-size unit, no leftover `_nomt`/`v.tex`/`k.tex` scratch artifacts in the submission.

---

*Bottom line: reject-in-current-form but a credible accept after a focused revision. The fastest expected-value path is: fix the desk-reject blockers (a weekend), reframe the novelty (a day of writing), and let the four queued jobs land — the heterogeneous-minority result and a second on-env algorithm are what carry it over the line.*
