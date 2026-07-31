# Changelog — edits applied to `AnonymousSubmission2027_2.tex` (and `references.bib`, `SupplementaryMaterial2027_2.tex`)

Every change is quoted **BEFORE → AFTER**. Files touched:
`AuthorKit27/AnonymousSubmission2027_2.tex`, `AuthorKit27/references.bib`, `AuthorKit27/SupplementaryMaterial2027_2.tex`.

Scope you chose: **all text-level fixes** · affine law = **keep, rename + baseline caveat** · abstract = **full rewrite**. Items needing new experiments, figure re-rendering, or numbers I don't have are **not** silently invented — they're listed under "Out of scope / author action required" at the end.

Structural check after edits: braces balanced, `$` delimiters even, all `equation/tabular/figure/table` environments matched. Full PDF compile could not be run in the sandbox (missing AAAI font package `newtxtext.sty`, no network to fetch); this is an environment limit, not an edit error. All 47 citation keys resolve against `references.bib`.

---

## Abstract — full rewrite (pointers 1, 2, 3, 4, 5, 7)

**BEFORE**
> Cooperative multi-agent reinforcement learning (MARL) can collapse into degenerate *lazy-agent* policies when each agent's local reward overwhelms the team signal. We study this failure in reward-mixed cooperative MARL, where each learner is trained on $x_i(w)=(1-w)\,r_i^{\mathrm{local}}+w\,r^{\mathrm{global}}$, and ask *where*, along the reward-centralization axis $w$, coordination gives way to collapse. We introduce a reusable boundary instrument: pre-specify a task-level collapse label, train across a grid of $w$, fit $P(\mathrm{collapse}\mid w)$, and report the critical centralization $\wcrit$ with bootstrap confidence intervals. Using this instrument, we test prospect-theoretic reward transforms applied after reward blending. *Loss aversion* shifts $\wcrit$ rightward: agents trained under loss-averse transforms require more team-level reward signal to coordinate. The direction replicates across four environment-algorithm combinations, the finite-horizon boundary slides left with additional training, and a mean-field account supports the sign; near a commonly used moderate loss-aversion setting ($\lambda\approx3$), the shift is $+0.089$ net of curvature ($+0.188$ in isolation). On the primary supply-chain substrate, the instrument also exposes scoped secondary findings: value-function curvature lowers the boundary, a bounded affine law fits the loss-aversion sweep, and a single downstream loss-averse node can nearly set the boundary. These findings are substrate-specific unless replicated. The contribution is a measurement protocol for reporting where a cooperative learning system crosses into collapse, rather than only asserting that centralization helps.

**AFTER**
> Cooperative multi-agent reinforcement learning (MARL) can collapse into degenerate *lazy-agent* policies when each agent's local reward overwhelms the team signal. We ask not whether centralization helps but *where*: at what critical team-reward fraction $\wcrit$ a trained population crosses into a pre-specified collapse region. We measure this boundary by sweeping the local-to-team reward mixture $w$, fitting $P(\mathrm{collapse}\mid w)$, and reporting $\wcrit$ with bootstrap confidence intervals; on our primary supply-chain testbed $\wcrit=0.666$ $[0.635,0.694]$. Using the instrument, we then test a loss-averse reward transform, which weights losses more heavily than equal gains, applied after reward blending. Loss aversion shifts $\wcrit$ rightward, so agents need more team reward to coordinate; the direction replicates across four environment-algorithm combinations, while longer training slides the boundary left. At a moderate setting ($\lambda{=}3$) the full transform moves the boundary $+0.089$ relative to the linear reward, or $+0.188$ once value-function curvature is separated from loss aversion. On the supply chain the instrument also exposes scoped secondary findings, including that a single downstream loss-averse node can nearly set the boundary. The contribution is a protocol for measuring where a cooperative learning system crosses into collapse, rather than only asserting that centralization helps.

What this fixes: leads with the CI'd number; **inverted decomposition corrected** ("+0.089 net of curvature" was backwards — +0.089 is the *with-curvature* deployment shift, +0.188 is the *curvature-removed* one; now stated as "+0.089 relative to the linear reward, or +0.188 once curvature is separated"); defines "loss-averse transform" inline (2); drops the duplicated $x_i(w)$ equation (7); removes "commonly used" from λ≈3 and uses λ=3 (canonical value is ≈2.25); keeps the "supply-chain" noun but drops the "substrate-specific unless replicated" hedge (4); moves the thesis sentence's job up front (5); trims the secondary-findings dump.

---

## Introduction

### Para 1 — soften overclaimed citations, add inline cites, cut placeholder (pointers 6, 8, 9, 14)

**BEFORE**
> …learnable signal and the lazy policy wins. The field knows this direction (*more centralization helps* [dmarl2025, liu2023lazy]). What is missing is the location: lazy-agent and credit-assignment work mitigates the pathology without locating a boundary, and the closest phase-diagram studies place their critical points on *environmental* or opinion-dynamics axes rather than reward structure; none estimates a confidence-intervalled crossing… Our central question is therefore twofold: … The first half is a measurement problem (the paper's primary contribution), and the second is a question the instrument then makes answerable; **we differentiate against the closest prior work below.**

**AFTER**
> …learnable signal and the lazy policy wins. That decentralized reward shaping cannot match centralized training is broadly accepted [dmarl2025], and lazy-agent and credit-assignment work [liu2023lazy, peng2023curo, sunehag2018vdn, rashid2018qmix, foerster2018counterfactual] mitigates the pathology; what none of it supplies is the *location*. The closest phase-diagram studies place their critical points on *environmental* [yamaguchi2025] or opinion-dynamics [capannoli2026phase] axes rather than reward structure, and none estimates a confidence-intervalled crossing… Our central question is therefore twofold: … The first is a measurement problem (the paper's primary contribution); the second is a question the instrument then makes answerable.

Removes the "field knows… (Akella/Liu)" overclaim (neither varies a centralization dial); adds the six credit-assignment + two phase-diagram citations inline; deletes the "we differentiate below" placeholder.

### Para 2 — applicability envelope, four cells, DAG, horizon-as-measurement (pointers 10, 12, 13; AI-tell removed)

**BEFORE**
> Pinning down this number… $\wcrit$ is not universal: it is a measured property of an environment, learner, horizon, and collapse label. **That is precisely what makes it useful:** it tells a designer where their own system crosses into collapse. Our primary testbed is automated replenishment, where local inventory decisions and global service outcomes conflict; more broadly, any cooperative learning system that blends local and team rewards, or trains with preference-shaped objectives… faces the same design question. We use prospect theory only as a controlled reward-transform family…

**AFTER**
> Pinning down this number… $\wcrit$ is not universal: it is a measured property of an environment, learner, training horizon, and collapse label, and it tells a designer where their own system crosses into collapse. **Applying the instrument requires three things: a reward-mixing dial, a policy class with a continuous scalar action channel, and a learner that can coordinate at high centralization. We estimate $\wcrit$ in four environment-algorithm combinations: a six-node inventory network under PPO and under MAPPO, and Cooperative Navigation (MPE) and Level-Based Foraging (LBF) under MAPPO. The inventory network is a branching directed-flow graph, which later lets us ask *which* node's reward design sets the boundary. Because the boundary is finite-horizon, we re-measure it at several horizons and report the rate at which extra training substitutes for centralization, a second measurement rather than a caveat.** More broadly, any cooperative learning system that blends local and team rewards, or trains with preference-shaped objectives… faces the same design question. We use prospect theory only as a controlled reward-transform family…

Removes the "That is precisely what makes it useful:" AI-cadence tell; adds the scope envelope, the four named cells, the DAG preview, and reframes the horizon slide.

### Para 3 — stop restating the protocol; 3 contributions; promote cascade; drop "checked premise" (pointers 21, 15, 14, 3-iii)

**BEFORE**
> **We therefore build the ruler first, and treat it as the paper's central object. Our protocol is simple.** Before fitting any boundary we fix a task-specific binary collapse label… then sweep $w$, fit $P(\mathrm{collapse}\mid w)$, and bootstrap confidence intervals… Two further design choices keep the readings interpretable. First, the prospect-theoretic value transform is applied after reward blending… Second, because a finite-horizon boundary can slide… The contributions are: (i) a confidence-intervalled boundary instrument…; (ii) an application showing that loss aversion shifts the boundary rightward…; (iii) **sign-level mechanism support under a checked supply-chain PPO premise**; and (iv) scoped secondary findings… **These findings are enabled by, but not required for, the main claim, and are reported as substrate-specific unless separately replicated.**

**AFTER**
> **We build the instrument before using it. Three design choices make a reading trustworthy: the collapse label is a behavioral rule fixed *before* any run is trained; matched policies differ only in the scalar training signal, because the prospect-theoretic transform is applied after blending, outside the optimizer; and every $\wcrit$ is reported with a bootstrap confidence interval.** "Phase boundary" is used only in this operational sense… The contributions are: (i) a confidence-intervalled boundary instrument for reward-mixed cooperative MARL; (ii) a replicated finding that loss aversion shifts the boundary rightward, so agents require more centralization to coordinate, with that direction and the finite-horizon slide holding across four environment-algorithm combinations; and (iii) a scoped supply-chain analysis, comprising a sign-level mean-field account, a bounded one-parameter summary of the $\lambda$-sweep, and **a directed-flow finding that a single downstream node's reward design can nearly set the whole team's boundary.** The magnitudes and secondary structure are substrate-specific unless separately replicated.

Collapses four contributions to three; the four-step protocol is no longer restated here (it lives canonically in Method); "the ruler"/"Our protocol is simple" tells removed; "checked… premise" removed; the cascade is promoted from apology (iv) into contribution (iii).

---

## Related Work

### Mean-field grouping split (pointers 6, 17)

**BEFORE**
> …\citet{capannoli2026phase} prove a metastable boundary in opinion dynamics under a fixed update rule, and the mean-field programme for multi-agent learning [barfuss2019deterministic, hu2019meanfield, hussain2025convergence] **derives Fokker–Planck descriptions of networked Q-learning.** We borrow…

**AFTER**
> …\citet{capannoli2026phase} prove a metastable boundary in opinion dynamics under a fixed update rule. **Mean-field and dynamical-systems analyses derive deterministic or distributional limits for multi-agent learning: a Fokker–Planck description of $Q$-learning in large populations [hu2019meanfield], the deterministic limit of temporal-difference learning in stochastic games [barfuss2019deterministic], and convergence in random networks [hussain2025convergence].** We borrow…

Corrects the false blanket claim that all three derive "networked Q-learning" (Hu is large-population repeated games; only Leonte/hussain is networked).

### "Thin bridge" topic sentence, KTO promotion, Mengesha direction, Lalmohammed softening (pointers 17, 6, 25)

**BEFORE**
> **A thin bridge connects human preference structure to collective learning dynamics:** \citet{hughes2018inequity} show inequity aversion improves cooperation… Risk-sensitive RL optimizes tail outcomes…, distinct from loss aversion… In the multi-agent setting, \citet{mengesha2025} **prove loss aversion shifts a tipping point in evolutionary game theory over fixed strategies**, and \citet{lalmohammed2025} bring CPT into cooperative MARL but **fold it into the update, entangling its components.**

**AFTER**
> **Human preference structure has entered collective learning only narrowly.** \citet{hughes2018inequity} show inequity aversion improves cooperation…, **and loss aversion is now a training objective at scale: KTO aligns language models with a Kahneman–Tversky value function carrying an explicit loss-aversion coefficient [ethayarajh2024kto].** Risk-sensitive RL optimizes tail outcomes…, distinct from loss aversion… In the multi-agent setting, \citet{mengesha2025} **find loss aversion has a *non-monotone* effect on equilibrium selection in an evolutionary model with quantal response**, and \citet{lalmohammed2025} bring CPT into cooperative MARL but **fold it into the actor–critic update, so its value and probability-weighting components vary jointly with the optimizer.**

Replaces the gap-asserting metaphor topic sentence (also #1 AI-tell); promotes KTO into a sentence; **corrects Mengesha** (their result is non-monotone/partly opposite, not a clean tipping-point shift); softens the loaded "entangling."

---

## Preliminaries and Method

### Opening — CTDE, independent-PPO correction, `w`, one-dial sentence (pointers 18, 19, 20)

**BEFORE**
> We study *reward-mixed* cooperative MARL: settings in which $N$ agents share an environment **under CTDE** [lowe2017multi] and each learner's scalar training reward is a blend… The restriction is deliberate: it yields **a controllable reward-design axis.** … Unless stated otherwise, each experiment varies one dial at a time: **the $w$-sweep holds $\alpha,\beta,\lambda$, learner, environment, horizon, and collapse label fixed, and each of those quantities varies only in its own named sweep.**

**AFTER**
> We study *reward-mixed* cooperative MARL: settings in which $N$ agents share an environment and each learner's scalar training reward is a blend… **The learners range from independent policy gradients to centralized training with decentralized execution (CTDE) [lowe2017multi]; our primary supply-chain learner is independent PPO, while the transfer settings use a centralized critic.** The restriction to a blended reward is deliberate: it yields **a controllable reward-design axis $w$.** … Unless stated otherwise, **each experiment varies one factor at a time, holding the learner, environment, horizon, transform parameters, and collapse label fixed while a single named sweep moves.**

Expands CTDE, **corrects the framing that the whole paper is "under CTDE"** (primary PPO is independent), adds `w`, and removes α/β/λ from a sentence that appears before they're defined.

### Transform paragraph — define curvature, why-measured, probability weighting, "linear utility", Lalmohammed (pointers 22, 23, 24, 25, 31)

**BEFORE**
> …we use $v(x)=x^\alpha$ for $x\ge0$ and $-\lambda(-x)^\beta$ for $x<0$, where $\alpha,\beta\in(0,1]$ **govern value-function curvature** and $\lambda>1$ is loss aversion (**$\alpha{=}\beta{=}1,\lambda{=}1$ is linear utility**). **We do not model probability weighting; the experimental transform consists only of curvature and loss aversion.** We apply it after blending, [eq] **the decisive difference from \citet{lalmohammed2025}, who fold CPT *into* the actor–critic update, entangling curvature with loss aversion and confounding any shift with a change to the optimizer.** The optimizer and architecture are identical…

**AFTER**
> …we use $v(x)=x^\alpha$ for $x\ge0$ and $-\lambda(-x)^\beta$ for $x<0$. **The exponents $\alpha,\beta\in(0,1]$ control *curvature*: below one the transform is concave, so a reward of $100$ is worth less than twice a reward of $50$ and large outcomes are compressed toward small ones. The multiplier $\lambda>1$ is separate: it scales the entire loss branch, so a loss of magnitude $x$ counts $\lambda$ times as much as a gain of $x$ (*loss aversion*), and $\alpha{=}\beta{=}1,\lambda{=}1$ leaves the reward unchanged, $v(x)=x$. Because curvature is present in every loss-aversion condition, it must be measured on its own, or its effect is silently folded into the $\lambda$ shift; this is why we run a curvature-only baseline. Full cumulative prospect theory also reweights outcome probabilities; we omit that channel, so the only thing that varies between matched runs is the scalar value transform.** We apply it after blending, [eq] **which keeps the optimizer and architecture bit-identical across matched conditions. This is the difference from \citet{lalmohammed2025}, who embed CPT *inside* the actor–critic update; that is appropriate for their aims but varies the value transform, probability weighting, and the optimizer together.** …

Defines curvature concretely + states *why* it's decomposed; reframes probability weighting as a differentiator; replaces "linear utility"; positive, non-uncharitable Lalmohammed contrast.

### Collapse-label paragraph — S̄/π̄, floor redundancy, MPE/LBF names, pre-registration footnote (pointers 26, 27, 28)

**BEFORE**
> A *collapse label* is a pre-specified binary evaluation rule… In the supply chain, collapse is $\mathbf{1}[\bar S<10\wedge\bar\pi\le-127.1]$ ($\bar S$ mean base-stock, $\bar\pi$ mean profit), which corresponds to low/no ordering with poor profit. In MPE, collapse is landmark occupancy below $\theta=0.144$…; in LBF, collapse is failed cooperative loading / low food collection under the pre-specified evaluation threshold. **The supply-chain label, analysis code, and probes were committed before execution; the anonymized bundle freezes them for review, with the de-anonymized commit hash released on acceptance.** Threshold robustness holds…

**AFTER**
> A *collapse label* is a pre-specified binary evaluation rule…, scored on held-out evaluation episodes rather than training reward. In the supply chain, collapse is $\mathbf{1}[\bar S<10\wedge\bar\pi\le-127.1]$, with $\bar S$ the mean base-stock and $\bar\pi$ the mean profit over deterministic evaluation; this corresponds to low/no ordering with poor profit. **Collapsed runs pile up at the lazy attractor's deterministic profit floor ($\approx-128$), so the profit clause is effectively a redundancy check on $\bar S<10$ rather than a tuned cutoff.** In MPE (Cooperative Navigation), collapse is landmark occupancy below $\theta=0.144$…; in LBF (Level-Based Foraging), collapse is a cooperative food-collection rate below the pre-specified evaluation threshold (stated with the LBF setup in the Supplement).**^footnote{The supply-chain label, analysis code, and mechanism probes—including two probes that later returned negative results (Supplement)—were committed before execution; the anonymized bundle freezes them, and the de-anonymized commit hash is released on acceptance.}** Threshold robustness holds…

Defuses the −127.1 magic-number worry; expands MPE/LBF; moves pre-registration mechanics to a footnote **and** notes the two failed probes there. (LBF numeric threshold still points to the Supplement — see author-action list.)

### Estimation + setup — b₀/b₁, Firth+why+cite, bootstrap, PPO scope, MAPPO named, inventory, reading guide (pointers 29, 30, 32, 33, 34)

**BEFORE**
> For each condition we fit $P(\mathrm{collapse}\mid w)=\sigma(b_0+b_1 w)$ and report $\wcrit=-b_0/b_1$… **We bootstrap the 95% CI; steep high-$\lambda$ cells use Firth penalized regression, whose $\wcrit$ agrees with the bootstrap estimate to within $0.019$…** … **Agents run PPO [schulman2017ppo] via Stable-Baselines3 [raffin2021sb3] and PettingZoo/SuperSuit…** **The 1,524 primary supply-chain runs cover structural boundaries, the null reproduction, decomposition, the loss-aversion sweep, basin control, and horizon checks; headline $\lambda$ sweeps use 20 seeds/$(w,\lambda)$ cell. Full setup is in the Supplement.**

**AFTER**
> For each condition we fit a logistic model $P(\mathrm{collapse}\mid w)=\sigma(b_0+b_1 w)$, **where $b_0$ is the intercept and $b_1$ the slope in $w$ (negative, since collapse falls as $w$ rises)**, and report $\wcrit=-b_0/b_1$… **Confidence intervals are percentile bootstrap over runs [efron1993bootstrap] (2,000 resamples). When collapse is near-perfectly separated across adjacent $w$ (steep high-$\lambda$ cells), the ordinary logistic slope diverges, so those cells use Firth's bias-reduced logistic regression [firth1993], whose $\wcrit$ agrees with the bootstrap estimate to within $0.019$…** … **Supply-chain agents run independent PPO [schulman2017ppo] via Stable-Baselines3 [raffin2021sb3]; MPE, LBF, and the cross-algorithm supply-chain control use centralized-critic MAPPO [yu2022surprising]. All environments are wrapped with PettingZoo/SuperSuit…** **Headline $\lambda$ sweeps use 20 seeds per $(w,\lambda)$ cell; the full run inventory, including the 450k/900k horizons, the finer high-$\lambda$ grid, and the entropy ablation, is itemized in the Supplement. Reading guide. All primary results are measured on the supply chain under PPO; MPE, LBF, and MAPPO are transfer settings used to test which claims travel, and the claim ledger (Table 2) states which is which.**

Defines b₀/b₁; cites Firth (with rationale) and Efron; documents the bootstrap; **scopes PPO to the primary substrate and names MAPPO as the transfer learner**; replaces the "1,524 runs cover…" overclaim with a pointer to a (to-be-completed) full inventory; adds a reading guide.

---

## Results

### "Two effect sizes" — numeric consistency + table pointer + affine rename (pointers 39, 48, 62)

**BEFORE**
> …at $\lambda{=}3$, $\wcrit=0.774$ is $+0.089$ over $0.685$; at $\lambda{=}7$, **$+0.166$.** The *isolated loss-aversion shift* … : **$+0.188$ and $+0.265$ respectively (reconciled by the curvature offset, $-0.100+0.265\approx0.166$).** We highlight $\lambda{=}3$ as **a moderate, empirically motivated setting**… The full sweep is tracked by a **one-parameter, theory-bounded affine law** (derived below) to RMSE $0.014$…

**AFTER**
> …(Table 1). …at $\lambda{=}3$, $\wcrit=0.774$ is $+0.089$ over $0.685$; at $\lambda{=}7$, **$+0.167$.** The *isolated loss-aversion shift* … : **$+0.188$ and $+0.266$ respectively. The two differ by the curvature offset ($-0.100$ against the linear baseline; Table 1).** We highlight $\lambda{=}3$ as **a moderate setting near the canonical loss-aversion range**… The full sweep is additionally summarized by **a one-parameter saturating law** (below) to RMSE $0.014$…

Fixes the +0.166/+0.167 and +0.265/+0.266 mismatches against Table 1; removes the fragile arithmetic "reconciliation"; adds the table pointer; renames the affine law; softens the λ=3 "empirically motivated" (canonical is ≈2.25).

### Figure 1 caption (pointer 36)

**BEFORE**
> …curves show fitted $P(\mathrm{collapse}\mid w)$ **at 150k and 300k steps.** (b) Increasing the training horizon shifts the boundary left, log-linearly in $T$ (RMSE 0.027; **20 seeds per horizon**)…

**AFTER**
> …curves show fitted $P(\mathrm{collapse}\mid w)$ **at two representative horizons (150k and 300k steps). (b) All four measured horizons: increasing the training horizon shifts $\wcrit$ left**, log-linearly in $T$ (RMSE 0.027; **20 seeds per $w$-cell at each horizon**)…

Clarifies that panel (a) is two representative curves and (b) is all four (your "missing curves" concern was a caption ambiguity, not a data gap); fixes "20 seeds per horizon."

### Table 1 body (pointer 39)

**BEFORE**
> `Curvature … & $-0.100^{\checkmark}$ & 0` · `Loss av. λ3 … & $+0.089^{\checkmark}$ & $+0.188$` · `Loss av. λ7 … & $+0.167^{\checkmark}$ & $+0.266$`

**AFTER**
> `Curvature … & $-0.100^{\checkmark}$ & --` · `Loss av. λ3 … & $+0.089^{\checkmark}$ & $+0.188^{\checkmark}$` · `Loss av. λ7 … & $+0.167^{\checkmark}$ & $+0.266^{\checkmark}$`

Tautological "0" (curvature vs. itself) → "—"; adds the ✓ significance marks to the "vs. curv." column the text calls "the cleaner estimate."

---

## Mechanism

### Lead with the barrier; define κ_T; median first-passage (pointers 41, 42, 43)

**BEFORE**
> …this section adds a scoped mean-field account… **under a local sign premise checked on that substrate.** … the Supplement writes the Gaussian policy mean as a diffusion with drift $F$ and update-variance $\Sigma$. **Near the boundary, the relevant object is escape signal-to-noise, not drift alone:** [eq S] where the escape interval runs from the lazy basin to the coordinating basin. A finite-horizon boundary is the crossing $\mathcal{S}\approx\kappa_T$. In the Kramers first-passage picture…, $\tau_{L\to C}$ is the time to escape…, so **collapse at horizon $T$ means $\tau_{L\to C}>T$.** Increasing $w$ shortens that escape time…

**AFTER**
> …this section adds a scoped mean-field account… (premise clause moved). … the Supplement writes the Gaussian policy mean as a diffusion with drift $F$ and update-variance $\Sigma$, **so that escape from the lazy basin to the coordinating basin is a Kramers first-passage problem [kramers1940brownian]. The mean escape time scales as $\tau_{L\to C}\asymp\tau_0\exp(\Delta U)$ with barrier $\Delta U(w,\lambda)$; equating the median first-passage time to the training budget $T$ gives $\Delta U\approx\ln(T/\tau_0)$, and if the barrier is approximately linear in $w$ near the crossing this predicts the observed log-linear slide $\wcrit=a-b\ln T$. A run collapses at horizon $T$ when its realized first-passage time exceeds $T$, so the $50\%$ boundary is where the median first-passage time equals $T$. As a monotone proxy for the barrier we track the escape signal-to-noise ratio** [eq S] over the same escape interval, with the finite-horizon boundary at the crossing $\mathcal{S}\approx\kappa_T$, **where $\kappa_T$ is a horizon-dependent critical value (longer horizons require less escape SNR).** Increasing $w$ shortens the escape time…

Now the **barrier picture that actually predicts the observed `2.44 − 0.138 ln T`** leads; SNR ratio demoted to a proxy; κ_T defined; mean-vs-median first-passage corrected.

### Split the SNR premise (pointer 44)

**BEFORE**
> The sign result is **conditional on a local premise checked on the supply-chain substrate:** increasing centralization raises escape SNR, while increasing loss aversion lowers escape SNR… The loss-aversion effect may operate through drift, diffusion, or both; we therefore state the premise at the SNR level…

**AFTER**
> The sign result rests on a two-part local premise, and the two parts have different evidential status. **The centralization half, $\partial_w\mathcal{S}>0$, is supported by the measured incentive signs on the supply chain (Supplement). The loss-aversion half, $\partial_\lambda\mathcal{S}<0$, is *not* directly measured: our pre-committed probes reproduced the boundary without localizing it, so we test it only indirectly through the entropy dissociation above…** it is consistent with the data but not independently verified.

Breaks the circularity (premise was "checked" by the very λ-monotonicity it explains).

### Entropy dissociation honesty (pointer 45)

**BEFORE**
> The same account is consistent with an exploration-noise dissociation… Injected entropy **raises $\wcrit$ on the supply chain and LBF**, where injected exploration is a binding noise term, **but stays flat on MPE**… This supports the SNR account…

**AFTER**
> The account also predicts an exploration-noise dissociation… Injecting entropy is a second SNR-lowering intervention: it **raises $\wcrit$ clearly on LBF (non-overlapping CIs), shifts it in the same direction but within noise on the supply chain (overlapping CIs), and leaves it flat on MPE**… **LBF thus confirms the prediction, the supply chain is directionally consistent, and MPE is a well-powered null. This is indirect support only…**

Stops applying two evidentiary standards (SC "rises" and MPE "flat" both on overlapping CIs).

### Proposition 1 ownership (pointer 46)

**BEFORE**
> The proposition fixes only the direction; the magnitude depends on unobserved diffusion scale and basin geometry.

**AFTER**
> **Proposition 1 is elementary; its content is premise (ii), an empirical claim about this substrate rather than a theorem.** It fixes only the direction, and the magnitude depends on unobserved diffusion scale and basin geometry.

### Affine law → saturating summary + caveats (pointer 48, your choice: keep+rename)

**BEFORE**
> **A theory-bounded law, tested out of sample.** The supply-chain PPO sweep is additionally summarized by a bounded affine law… held-out RMSE 0.016; all-seven RMSE 0.014… The sign transfers; the affine form does not…

**AFTER**
> **A one-parameter saturating summary of the sweep.** … held-out RMSE 0.016 and all-seven RMSE 0.014… **Two caveats bound the claim. First, the held-out $\lambda$ share the same environment, algorithm, and horizon as the calibration points, so this is an interpolation check within one sweep, not out-of-distribution validation, and the RMSE is comparable to the CI half-widths… Second, the $\rho\in[0,1]$ bound is slack here ($\rho=0.61$, interior) but binds on LBF, where the same form saturates and misfits by RMSE 0.15…** The sign transfers; the summary form does not…

Renamed; adds the within-sweep-interpolation caveat and the "the bound only does work on LBF" point. Figure 4 caption similarly retitled and given the interpolation caveat.

---

## Generalization

### Opening — name four cells, drop "It can:" (pointer 49)

**BEFORE**
> The first thing that must generalize is the *instrument*… **It can: we estimate a CI'd $\wcrit$ in two further environments and under a second algorithm**…

**AFTER**
> The instrument is only useful if it can be measured beyond the supply chain. **We estimate a CI'd $\wcrit$ in four environment-algorithm combinations: the supply chain under MAPPO (changing only the learner), and Cooperative Navigation (MPE) and Level-Based Foraging (LBF) under MAPPO (changing the environment). The design is not fully crossed, because independent PPO floors out on the shared-reward transfer tasks, so the supply-chain/MAPPO cell is what separates the learner change from the environment change.**…

### Herding wrapper reframed as method + wcrit=0.74 (pointer 50)

**BEFORE**
> …a herding wrapper adds a local reward for staying near other agents, creating the required local-versus-team conflict. **Its scale is chosen so the no-transform baseline boundary is interior to the measured $w$-grid**, and a run is labeled collapsed when landmark occupancy falls below the pre-specified threshold $\theta=0.144$.

**AFTER**
> …**because MPE has no native local–global conflict, we construct one with a herding wrapper… and we calibrate its scale so the no-transform boundary is interior ($\wcrit=0.74$)—otherwise there is no crossing to shift. The reported MPE $\wcrit$ is therefore a property of our wrapper, not of MPE; the scale was fixed before the $\lambda$-sweep, not selected on loss-aversion outcomes.** A run is labeled collapsed when landmark occupancy (fraction of landmarks covered, averaged over evaluation) falls below the pre-specified threshold $\theta=0.144$.

Converts the passive "its scale is chosen…" (garden-of-forking-paths risk) into a stated, pre-committed calibration; states the interior value; defines occupancy.

### Inconsistent λ endpoints explained (pointer 51)

**BEFORE**
> …On the *supply chain under MAPPO*…, $\wcrit$ rises $0.796\to0.889$…; on *MPE* the loss-aversion excess is **$+0.158$ at $\lambda{=}7$**; on *LBF* $\wcrit$ rises $0.26\to0.83$ by $\lambda{=}4$…

**AFTER**
> …**We report each setting at its largest $\lambda$ with an interior boundary estimate—$\lambda{=}7$ for the supply chain and MPE, $\lambda{=}4$ for LBF, whose crossing saturates against the $w{=}1$ ceiling beyond that point—and at the common moderate value $\lambda{=}3$ for a like-for-like comparison.** On the *supply chain under MAPPO*…; on *MPE*, $\wcrit$ rises $0.78\to0.89$, **a $+0.158$ shift over the no-transform null at $\lambda{=}7$ ($+0.101$ at $\lambda{=}3$, matching the supply chain's $+0.089$)**; on *LBF*…

States the "last interior λ" rule so the per-environment λ differences read as principled, not arbitrary; disambiguates the MPE "+0.158" baseline.

### Table 4 (pointer 51 / R4 P0-4)

**BEFORE (rows/caption)**
> caption "Endpoints are reported as λ=1→7 where both are interior; LBF uses the last interior endpoint." · header "Endpoints / shift · Affine law?" · `MPE (MAPPO) & 0.78→0.89 ($+0.158$ excess) & no` · `LBF (MAPPO) & 0.26→0.83 by λ=4 & no (0.15)`

**AFTER**
> caption "Endpoints are within-sweep (α=β=0.88)…; LBF uses its last interior endpoint (λ=4). Parenthetical values are endpoint differences; shifts against each setting's no-transform null appear in the text…" · header "Endpoints (λ=1→last) · Summary form?" · `MPE (MAPPO) & 0.78→0.89 ($+0.11$) & no` · `LBF (MAPPO) & 0.26→0.83 ($+0.57$, by λ=4) & no (0.15)`

Standardizes the column to within-sweep endpoint differences; fixes the mislabeled MPE "+0.158 excess" (raw endpoint Δ is +0.11).

### Scope condition reframed (pointer 52)

**BEFORE**
> **On the supply-chain base-stock control, the signal-to-noise crossing depends on a continuous scalar action representation…** it is a scope condition of the supply-chain mechanism… MPE is natively discrete in action labels but remains a continuous spatial coordination task and retains a boundary. Since value-based learners show no measurable boundary here, whether the loss-aversion direction holds under them is not well-posed, and we claim nothing about it.

**AFTER**
> **The instrument needs a continuous scalar action channel.** … **The positive control is MPE, which is natively discrete in action labels yet remains a continuous spatial coordination task and retains a boundary—so discreteness *per se* is not the culprit.** … it is a scope condition of the instrument, **established here at $K{=}11$ (whether finer bins recover the boundary is untested).** … **both learners we do use (PPO and centralized-critic MAPPO) are policy-gradient, so applicability to value-based learners is an open scope question rather than a tested negative.**

Promotes the MPE positive control; flags the K=11-only limit and the all-policy-gradient caveat pre-emptively.

---

## Minority section

### Lead with reasoning + prediction framing (pointers 53, 55)

**BEFORE**
> The main sweep is homogeneous… We now test a heterogeneous setting, **measured only on the supply-chain directed-flow substrate**… MPE and LBF do not have this directed resource-flow topology, so the cascade is a falsifiable directed-flow prediction rather than a tested generalization.

**AFTER**
> **This experiment is necessarily supply-chain-only: it requires a directed resource-flow topology (a DAG) that MPE and LBF, as non-networked coordination tasks, do not have.** The main sweep is homogeneous… Because MPE and LBF lack this topology, the cascade is **a falsifiable prediction for any branching directed-flow domain—traffic networks, power grids, multi-echelon logistics—that we state here and test only on the supply chain.**

### Dose-response confound + operational sharpening (pointers 56, 57a)

**BEFORE**
> …converging on the *fully* loss-averse value (0.852) by λ=7… **One node, biased hard enough, nearly matches the entire network's midpoint requirement; the homogeneous result is the saturation of a continuous law, not a special case.**

**AFTER**
> …converging on the *fully* loss-averse value (0.852) by λ=7… **The single-node curve has the same saturating shape as the homogeneous $\lambda$-sweep and shares its start and endpoint, though its independent variable is one node's $\lambda$ rather than the team's; whether the two curves coincide at intermediate $\lambda$—which would make the homogeneous law effectively this node's law—is a question we flag but do not resolve here. Operationally, the team's collapse boundary is not an average over agents: it is set, nearly entirely, by whichever node's output reaches demand most directly, so an audit of reward transforms should target that node, not the mean.**

Surfaces the confound (Reading B) instead of the cryptic "not a special case," and gives the operational payoff.

### Mengesha "cf." corrected (pointer 6 follow-through)

**BEFORE**
> …its product flows to demand **(cf. the EGT tipping-point result of mengesha2025, here in deep-MARL with a topological signature).**

**AFTER**
> …its product flows to demand. **Where \citet{mengesha2025} find loss aversion's effect on cooperation is *non-monotone* in an evolutionary model, the effect here is monotone in deep MARL and carries a topological signature, which suggests the sign is a property of the dynamics rather than of the preference structure.**

---

## Discussion

### Novelty reattached to the decomposition (pointer 57b)

**BEFORE**
> **That value-function curvature and loss aversion exert opposite effects on the primary substrate is, to our knowledge, novel;** prior work applies prospect theory as a unified model… a $+0.167$ midpoint margin that scales with $\lambda$; **settings with effective $\lambda$ near 2–3 would incur a smaller but positive shift under this substrate.**

**AFTER**
> **To our knowledge, no prior work separates the value transform's curvature and loss-aversion components in cooperative MARL;** prior work applies prospect theory as a unified model… **Decomposing it shows that on our primary substrate the two components move the boundary in *opposite* directions and would partially cancel if reported jointly, so a single "prospect-theoretic" condition can under-report the loss-aversion effect by about half. The curvature sign is substrate-specific (it does not transfer to MPE or LBF), but this methodological point survives all three settings.** …a $+0.167$ margin that scales with $\lambda$.

Moves the novelty claim from the finding (which doesn't transfer — Table 2 contradicts it) to the decomposition method (which does).

### Operational interpretation — concrete action, trim dual-use drama (pointer 58)

**BEFORE**
> The boundary is a design quantity: a team can estimate its own $\wcrit$, budget for the loss-aversion margin… The same topology information can be used defensively… **or adversarially, to identify where an induced bias would most degrade coordination; we report both openly so the effect stays auditable.**

**AFTER**
> The boundary is a design quantity. A team can estimate its own $\wcrit$… **if architectural constraints cap $w$ at, say, 0.8, the usable loss-aversion budget is bounded by where $\wcrit$ crosses that cap, about $\lambda\approx4$ on this substrate.** … **This topology information is dual-use—the same map that hardens the most influential node also identifies where an induced bias would most degrade coordination—which we address in the ethics statement.**

Adds a concrete design action; compresses the dual-use "drama" to a pointer to the ethics statement.

### Scope & Future Work — plain sentences, "combinations", drop bodyguard line (pointers 59, 60)

**BEFORE**
> Three boundaries each point to a concrete next step. *(i) From one substrate to many.* … replicate across **environments and a second algorithm**… *(ii) From calibrated to predictive.* … *(iii) From bracket to calibration.* Estimating the effective $\lambda$… **None undercuts the central, replicated claim.**

**AFTER**
> Three limits each point to a concrete next step. **First,** … replicate across **the four measured environment-algorithm combinations**… **Second,** the mechanism fixes sign and form but not magnitude… **so a localizing observable that predicts $\wcrit$ without a full sweep is the clearest open problem. Third, our $\lambda$ is set by hand; in deployed preference-aligned systems $\lambda$ is induced by the training objective and its effective value is unknown, so estimating it from behavior would connect the reward-transform axis to real training pipelines.**

Unpacks the opaque "From X to Y" triple into plain sentences; precise "combinations"; deletes the defensive "None undercuts…" closer.

---

## Conclusion — operational inversion (pointer 61)

**BEFORE**
> We introduced a confidence-intervalled boundary instrument… Using it, we showed, with sign-level mechanism support, that loss-aversion reward transforms shift the boundary rightward across four environment-algorithm combinations… The protocol is mechanism-agnostic once a control dial and collapse label are specified; **prospect-theoretic reward transformation is its first application.**

**AFTER**
> **Reward centralization is usually discussed as a direction: more team signal helps. We have shown it also has a *location*—a critical fraction $\wcrit$ that a team can measure on its own system to within a few hundredths—and that the location moves: rightward under loss-averse reward transforms, leftward with more training, and, on a directed network, almost entirely at the discretion of a single node near demand. The direction and the horizon slide generalize across the four measured settings; the magnitudes are ours. The instrument is anyone's.**

Replaces the abstract-restating summary with a memorable close.

---

## references.bib

**`hu2019meanfield` — BEFORE**
> title = {A mean-field theoretic approach to multiagent reinforcement learning Q-learning dynamics}, author = {Hu, Yan and Leung, Ho-fung and Leung, Cheuk Hang}, booktitle = {NeurIPS}, year = {2019}

**AFTER**
> title = {Modelling the Dynamics of Multiagent Q-Learning in Repeated Symmetric Games: A Mean Field Theoretic Approach}, author = {Hu, Shuyue and Leung, Chin-wing and Leung, Ho-fung}, booktitle = {NeurIPS}, volume = {32}, pages = {12102–12112}, year = {2019}

**`rabin2000risk` — BEFORE** `@inproceedings{… booktitle={Econometrica} …}` → **AFTER** `@article{… journal={Econometrica} …}`

**Added** `@article{firth1993, … Biometrika 80(1):27–38, 1993}` and `@book{efron1993bootstrap, … An Introduction to the Bootstrap, 1993}` (both now cited in Method).

---

## SupplementaryMaterial2027_2.tex

**p-value overclaim (R4 P0-2) — BEFORE**
> all seven slopes are significant **($p<10^{-3}$)** and the Firth $\wcrit$ matches the bootstrap to within 0.019

**AFTER**
> all seven slopes reject a flat relationship **(max $p=1.5\times10^{-3}$)** and the Firth $\wcrit$ matches the bootstrap to within 0.019

(The table's λ=7 row is p=1.5×10⁻³, which contradicted "p<10⁻³".)

---

## Out of scope / author action required (NOT changed — I won't invent data)

These are referenced by my edits or flagged by the council but need your numbers, new runs, or figure re-rendering:

1. **LBF numeric collapse threshold.** The supplement states the LBF label only qualitatively ("cooperative food-collection rate," "same collapse-midpoint criterion"). My main-text edit points to the Supplement; **you must insert the actual numeric threshold + aggregation window there.**
2. **MAPPO implementation library.** I named the learner "centralized-critic MAPPO [yu2022surprising]" but the supplement never names the *code* (SB3 has no MAPPO). Add the implementation (e.g., EPyMARL/your own) in both main text and supplement §Algorithm.
3. **Supplement Table 1 run inventory.** My main text now says the 450k/900k horizons, finer high-λ grid, and entropy ablation are "itemized in the Supplement." **Add those rows** (with true run counts) so every reported estimate maps to an inventoried run — the table currently sums to exactly 1,524 without them.
4. **"600k heatmap" ghost text (supplement §2).** References "the 600k row of the main-paper heatmap"; there is no heatmap in the main paper. Reconcile the leftover prior-draft text with the actual horizon set {150k, 300k, 450k, 900k}.
5. **"non-overlapping by 0.221"** (Results): from the rounded CIs the gap looks like ~0.217. Verify against unrounded values or soften to "by more than 0.2."
6. **Figure 2** (dotted λ curves): the dotted wcrit lines are illegible at print size — regenerate as a wcrit rug on the x-axis or thin to λ∈{1,3,5,7}. Text/caption already tolerant.
7. **Figure 3** (entropy dissociation): plot the censored ent=0.05 supply-chain point as a censored bound/arrow, not a point with an error bar. (Figure regeneration.)
8. **Figure 6** (cascade): label bars by topological role, ideally a DAG diagram in panel (a). (Figure regeneration.)
9. **Proposition 1 hypotheses** should be made identical between main text and supplement (main assumes h, S differentiable; supplement assumes only "strictly decreasing in S" then uses IFT).
10. **Verify the recent arXiv IDs** yourself (yamaguchi2025, weng2026, mengesha2025, capannoli2026phase, hussain2025convergence, etc.) — I cannot open URLs to confirm title/authors match.

## Experiments that would materially strengthen the paper (see separate `EXPERIMENTS_NEEDED.md`)
The two highest-value new runs (SC-MAPPO horizon sweep for a real 4-panel Figure 1, and the single-node-vs-homogeneous dose-response overlay) are specified in that file.

---

# Round 2 — five manual corrections + resolution of author-action items

## Correction 1 — "continuous scalar action channel" overgeneralized (your item 1)

**Intro envelope — BEFORE**
> Applying the instrument requires three things: a reward-mixing dial, **a policy class with a continuous scalar action channel**, and a learner that can coordinate at high centralization.

**AFTER**
> Applying the instrument requires a reward-mixing dial and a learner that can coordinate at high centralization; **a third condition is narrower and empirical, in that on the supply-chain base-stock control a coarse action discretization erases the boundary, whereas action-discrete environments such as MPE still exhibit one** (see the scope condition below).

**Scope-condition paragraph — BEFORE**
> **The instrument needs a continuous scalar action channel.** On the supply-chain base-stock control, a single-variable control isolates why… it is a scope condition of **the instrument, established here at $K{=}11$**…

**AFTER**
> **On the supply-chain base-stock control, the boundary depends on how finely the scalar action is discretized.** A single-variable control isolates why… it is a scope condition of **this substrate's coarse action encoding rather than a general requirement of the instrument, and is established here only at $K{=}11$**…

The MPE positive control is now used to say discreteness *per se* is fine; only coarse K=11 base-stock binning erases the boundary.

## Correction 2 — Kramers "predicts" → "consistent with / motivates" (your item 2)

**BEFORE**
> …gives $\Delta U\approx\ln(T/\tau_0)$, and **if the barrier is approximately linear in $w$ near the crossing this predicts the observed log-linear slide $\wcrit=a-b\ln T$.**

**AFTER**
> …gives $\Delta U\approx\ln(T/\tau_0)$. **If the barrier is additionally taken to be approximately linear in $w$ near the crossing—an assumption we adopt for interpretation but do not derive—this account is consistent with the observed log-linear slide $\wcrit=a-b\ln T$, which we treat as motivation for the picture rather than a derivation of it.**

## Correction 3 — pre-specification scoped to the supply chain (your item 3)

**Intro contributions — BEFORE**
> …the collapse label is a behavioral rule **fixed *before* any run is trained**; matched policies differ only in the scalar training signal…

**AFTER**
> …the collapse label is a behavioral rule **scored on evaluation behavior rather than training reward, and pre-registered on the supply-chain substrate (label, analysis code, and probes committed before execution)**; matched policies differ only in the scalar training signal…

(The Method footnote already scoped pre-registration to the supply chain; this removes the remaining over-general implication that every MPE/LBF label was committed before any run.)

## Correction 4 — de-AI three phrases (your item 4)

**Conclusion — BEFORE**
> …leftward with more training, and, on a directed network, **almost entirely at the discretion of a single node near demand.** The direction and the horizon slide generalize across the four measured settings; **the magnitudes are ours. The instrument is anyone's.**

**AFTER**
> …leftward with a longer training horizon, and, on a directed supply network, **largely with the reward design of the downstream node nearest demand.** The direction and the horizon slide generalize across the four measured environment-algorithm combinations; **the numerical thresholds are specific to the systems we study, but the measurement protocol is not.**

**Intro — BEFORE**
> …we re-measure it at several horizons and report the rate at which extra training substitutes for centralization, **a second measurement rather than a caveat.**

**AFTER**
> …we re-measure it at several horizons; **the rate at which extra training substitutes for centralization is itself one of the quantities we report.**

## Item 5 — author-action items RESOLVED from your repo (no longer flagged)

I found the real values in the codebase and applied them, rather than re-flagging:

**LBF collapse threshold** (was: "never stated"). Found in `experiments/analyze_lbf_horizon.py` (`theta=0.626`, `collapsed = mean_reward < theta`).
- Main text BEFORE: "collapse is a cooperative food-collection rate below the pre-specified evaluation threshold (stated with the LBF setup in the Supplement)."
- Main text AFTER: "collapse is **mean team return below $\theta=0.626$ (the calibrated collapsed-versus-coordinated midpoint)**, i.e. agents fail to cooperatively load food."
- Supplement LBF subsection: "same collapse-midpoint criterion **(mean team return below $\theta{=}0.626$;** 1,600 runs)."

**MAPPO implementation** (was: "never named"). Found in `experiments/exp_mappo_mpe.py` (`PPO(CentralizedCriticPolicy, …)`; header: "MAPPO is PPO with a wider centralised critic").
- Main text: "MPE, LBF, and the cross-algorithm supply-chain control use **MAPPO—PPO with a centralized (joint-state) critic [yu2022surprising], implemented on the same Stable-Baselines3 stack.**"
- Supplement: "we therefore train MPE with centralized-critic MAPPO…, **implemented as Stable-Baselines3 PPO with a centralized (joint-state) critic policy so the optimizer matches the supply-chain PPO.**"

**Complete run inventory** (was: "table sums to 1,524 without the reported horizons"). Verified the runs exist on disk and counted them: 450k = **160**, 900k = **160** (both α=β=1, matching the main-text risk-neutral horizons; `docs/expHorizon_extra_figures/horizon_extra_raw_data.csv`), entropy ablation = **400** (`docs/expEnt_raw_data.csv`), finer high-λ grid = **120** (`docs/expHighLambda_finegrid_figures/`).
- Supplement Table 1: added rows **T @450k (160), T @900k (160), Entropy (400), High-λ grid (120)**; relabeled the old "T (600k)" curvature run as **"T @600k — Metastability (curv. probe)"**; **total corrected 1,524 → 2,364** (verified sum). Inventory intro sentence rewritten to match.

**600k "main-paper heatmap" ghost text** (was: references a heatmap that doesn't exist).
- Supplement BEFORE: "**The 600k row of the main-paper heatmap** uses α=β=0.88… only **the figure's** 600k magnitude is curvature-inflated."
- AFTER: "**The exploratory 600k metastability probe (Table 1, T @600k)** uses α=β=0.88… **The main-text four-horizon slide is measured at matched α=β=1 and is therefore already curvature-free**… confirming it is not a curvature artifact." (No reference to a nonexistent heatmap/figure.)

**Still genuinely open (cannot be resolved by editing — need you):**
- **Figure readability** (Figs 2, 3, 6): these require re-rendering the plots (dotted-line legibility, censored-point marker, DAG topology labels). I can't regenerate the figures from here; specified in `EXPERIMENTS_NEEDED.md`/the figure notes.
- **"non-overlapping by 0.221"**: verify against unrounded CIs (rounded values suggest ~0.217).
- **arXiv ID verification** for the 2025–26 preprints: only you can open the URLs.
- **Proposition 1 hypotheses** should be made identical between main text and supplement (wording alignment).

Structural re-check after Round 2: both `.tex` files have balanced braces / even `$` / matched environments; supplement inventory sums to 2,364; all 47 main-text citation keys resolve.
