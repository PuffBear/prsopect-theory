# Reviewer Kill-Shots

*The 20 strongest objections an AAAI MAS committee can raise, and the answer to each.
Format per item: **the kill-shot** (stated as harshly as a hostile reviewer would),
then **the answer** (what is already in the paper / what to add), then **residual
risk** (honest — what the answer does NOT fully close).*

The five named priority objections are #1–#5.

---

## #1 — Why "phase boundary" and not just "a threshold"?

**Kill-shot.** "Calling a logistic midpoint a *phase boundary* is physics cosplay.
You have a sigmoid in one variable. Drop the thermodynamic vocabulary and you have a
threshold fit dressed up to sound profound."

**Answer.** The phase-boundary framing earns its keep by making four *non-obvious,
falsifiable* predictions that a bare threshold does not:
(i) **sharpness** — the transition width $\Delta w$ should be narrow and should
*shrink* as policies converge (observed: $0.278\!\to\!0.199$ from 150k→300k);
(ii) **metastability** — the boundary should *slide* with training horizon because
collapse is a finite-time escape, not an absorbing state (observed: $0.822\!\to\!0.666$,
and replicated on MPE and LBF);
(iii) **a Kramers $-\ln T$ functional form** for that slide (not just "it moves");
(iv) a low-noise **deterministic limit** $w_{\text{crit}}^0 = c/(c+g)$ toward which the
finite-$T$ boundary descends.
A "threshold" predicts none of these — it just reports a number. We use "phase
boundary" in the *empirical* sense (a sharp metastable threshold whose $T\!\to\!\infty$
value is the boundary proper), and we say so explicitly, disclaiming any
thermodynamic-singularity claim. The word is doing scientific work: it generated the
metastability experiment.

**Residual risk.** A purist may still dislike the term. Mitigation: keep the explicit
one-sentence disclaimer in the abstract; never call it a "thermodynamic" transition.

---

## #2 — Why not just a threshold fit? Isn't the logistic the whole paper?

**Kill-shot.** "Strip the framing and the contribution is: fit a logistic, read off
the 0.5 point, bootstrap a CI. That's a homework exercise, not a AAAI paper."

**Answer.** The logistic is the *ruler*, not the result. The contribution is what the
ruler measures and the discipline around it: (a) a **pre-registered, frozen** collapse
label (commit `03c3fc6`) so the ruler can't be retrofitted; (b) a **three-way causal
decomposition** (null / curvature / curvature+loss-aversion) that attributes boundary
*movement* to specific psychological components — impossible with a single fit;
(c) an **a priori sign theorem** that predicts the direction of one of those movements
before the fit is run; (d) **replication of that direction across 3 environments and 2
algorithms**; (e) a **locus result** (downstream minority) that a threshold number
cannot even express. A logistic fit is step one of about six. See the dedicated
appendix *"Why This Is Not Just a Logistic Threshold Fit."*

**Residual risk.** Low, once the decomposition and theorem are foregrounded. The risk
is *presentational* — if the paper leads with the fit machinery instead of the
discovery, reviewers anchor on "it's a fit." Fix in the rewrite.

---

## #3 — Why isn't this just reward engineering / hyperparameter tuning?

**Kill-shot.** "You found that changing the reward weight $w$ changes behavior. Of
course it does. You engineered a reward and reported the knob's setpoint. Where is the
science?"

**Answer.** Three structural facts separate this from reward engineering:
1. **$w$ is the object of study, not a tuned nuisance.** We do not *pick* a good $w$;
we map the *entire* response of collapse to $w$ and locate the critical point. The
deliverable is the function $P(\text{collapse}\mid w)$, not a winning configuration.
2. **The optimiser is byte-for-byte identical across all conditions.** CPT is applied
as a *reward transform after blending* (Eq. 2), never inside the policy-gradient
update. So a measured shift in $w_{\text{crit}}$ *cannot* be a tuning artifact — the
learning algorithm is unchanged; only the agents' value function differs. This is the
decisive design difference from prior CPT-in-MARL work that folds CPT into the update.
3. **Everything else is frozen** — environment, hyperparameters, seeds — so $w$ and the
behavioral parameters $(\alpha,\beta,\lambda)$ are the *only* moving parts. "Sole
structural intervention" is not rhetoric; it's a controlled experiment.
Reward engineering asks "what reward makes it work?" We ask "where does cooperation
break, and what law governs that break?"

**Residual risk.** A skeptic may say the per-agent reward blend is itself a designed
object. True — but it's the *standard* CTDE mixing dial, not a bespoke shaping term,
and we cite its lineage. Keep the "standard $w$-mixing, not custom shaping" sentence.

---

## #4 — Why isn't the mechanism post-hoc story-telling?

**Kill-shot.** "The mean-field Langevin/Kramers section is a narrative laid over
results you already had. You fit $\rho$, $g/c$, and a noise scale to the data, then
claim the theory 'explains' the data. That's circular."

**Answer.** The mechanism is **pre-committed and sign-level**, and we built a ledger
(Table: "What the mean-field mechanism claims") that physically separates *derived
(no fit)* from *calibrated* from *not claimed*:
- **Derived a priori, zero fit:** the *opposite signs* of curvature vs loss aversion;
the *monotonicity* of $w_{\text{crit}}(\lambda)$; the *affine* form of $\phi(\lambda)$
with a slope *bounded to $[0,1]$* by theory; the *sharpness*; the *sign* of the horizon
slide; the *sign* of the exploration-noise effect. These were stated as predictions and
the data bore them out — including an **out-of-sample** test: $\rho$ and $g/c$ fixed on
$\lambda\in\{1,3,7\}$, then the affine law predicts $\lambda\in\{2,4,5,6\}$ at RMSE 0.014.
- **The sign theorem (Thm 3)** proves $\partial_\lambda w_{\text{crit}}>0$ from
$\partial_\lambda F<0$ alone, *independent of the noise magnitude*. A post-hoc story
cannot produce a sign that is guaranteed before data.
- **We report the mechanism's one failure honestly:** the pre-registered direct SNR
probe (Appendix E) came back *inconclusive* on the drift-sign observable, and we say so.
A post-hoc narrative would have quietly dropped the failed test.
- **A dissociation test:** the exploration-noise effect appears on SC and LBF and is
*absent on MPE exactly where the mechanism says intrinsic exploration dominates* — a
prediction that could have failed and didn't. Post-hoc stories don't risk dissociations.

**Residual risk.** The *magnitude* of $w_{\text{crit}}$ is calibrated, not predicted —
we state this repeatedly. A reviewer who demands a first-principles magnitude will be
unsatisfied; the honest answer is "that's future work, and no claim rests on a predicted
magnitude." Keep that sentence verbatim.

---

## #5 — Why is the collapse definition valid (and not gerrymandered)?

**Kill-shot.** "You defined collapse as $\bar S<10 \wedge \bar\pi\le-127.1$. Those
numbers look hand-picked to produce a clean sigmoid. Change them and your boundary
moves."

**Answer.** Four defenses, in increasing strength:
1. **Pre-registration.** The definition was frozen *before any analysis* at a named
commit (`03c3fc6`). It could not have been reverse-engineered from the boundary.
2. **Robustness sweep.** We re-estimate $w_{\text{crit}}$ under nine perturbations of
both thresholds ($\bar S\in\{8,10,12\}$, $\bar\pi\in\{-102.5,-127.1,-152.5\}$); across
all estimable variants the maximum $\Delta w_{\text{crit}}=0.015$ — well inside the CI
width. The boundary is not an artifact of the specific cutoffs.
3. **Behavioral, scale-invariant labels off the supply chain.** On MPE/LBF, collapse is
*per-step landmark occupancy* / loading behavior — a behavioral metric *invariant to the
reward scale*, hence unaffected by the CPT transform. The boundary survives a definition
that has nothing to do with profit numbers.
4. **Mechanistic anchoring.** The label corresponds to the *lazy-agent* policy
(near-zero base stock, near-floor profit) that the prior literature already names as the
failure mode — we're operationalizing an established phenomenon, not inventing one.

**Residual risk.** The supply-chain thresholds are still substrate-specific. Mitigation:
foreground the scale-invariant MPE/LBF labels as proof the *concept* of collapse
transfers even when the *numbers* don't.

---

## #6 — Most of your best results are on ONE substrate (or-gym).

**Kill-shot.** "The affine law and the minority cascade — your two most memorable
results — are SC/PPO only. This is a single-environment paper wearing a generality
costume."

**Answer.** We separate, in a **claim–evidence ledger**, exactly what generalizes from
what doesn't. The *central* behavioral claim — loss aversion raises $w_{\text{crit}}$ —
replicates across **3 environments (SC, MPE, LBF) and 2 algorithms (PPO, MAPPO)**. The
affine *constants/form* and the cascade are scoped to SC, and we *say so* rather than
overclaim. Crucially, the cascade is framed not as a universal claim but as a
**falsifiable, structurally-specified prediction**: any domain with (i) directed
resource flow, (ii) genuine local-vs-global tension, (iii) a learning-reachable
boundary should show it — and we report that *most benchmarks fail one of these*, which
is itself a finding about where the phenomenon can live.

**Residual risk.** This is the paper's single largest vulnerability. The fix that would
*close* it is a second cascade substrate (congested traffic / staged allocation). Until
then, the defense is confident scoping, not proof. **Highest-value optional experiment.**

---

## #7 — Baselines may be under-tuned (the empiricist's reflex).

**Kill-shot.** "Did you tune MAPPO/QMIX as hard as your PPO? A 5% gain on an untuned
baseline is noise."

**Answer.** This paper is unusually robust to this attack because **we are not claiming
to beat a baseline.** There is no "our method vs SOTA" table to game. The comparison is
*within-condition*: the same learner at different $w$ and $\lambda$, identical
hyperparameters across conditions by construction. The cross-algorithm result (MAPPO on
the *identical* substrate, grid, seeds, label) varies *only* the learner — so any
tuning asymmetry would have to fake a *consistent directional* effect across both
learners, which is far less plausible than a one-off score gap.

**Residual risk.** Low. Worth one explicit sentence: "No claim in this paper is a
benchmark-beating result; comparisons are within-condition with shared hyperparameters."

---

## #8 — 20 seeds is too few; AAAI reviewers want 30+.

**Kill-shot.** "20 seeds, steep transitions, and you're reading off a 0.5 crossing.
Your CIs are optimistic."

**Answer.** We use **bootstrap CIs at the run level (2,000 resamples)** and **Wilson
intervals** for per-$w$ fractions — not Gaussian approximations that fail at the
extremes. For steep high-$\lambda$ transitions where MLE is ill-conditioned, we use
**Firth penalized** logistic regression and show Firth and bootstrap $w_{\text{crit}}$
agree to within 0.019. The headline behavioral claim rests on **non-overlapping**
endpoint CIs (e.g. $\lambda{=}1$ vs $\lambda{=}7$ separated by 0.221), not on marginal
significance. Total run count is **1,524** on SC alone, plus 1,760 (MPE), 1,600 (LBF),
700 (MAPPO-SC).

**Residual risk.** A reviewer may still want 30 seeds at the high-$\lambda$ cells. Cheap
to add for the camera-ready; mention a finer $w$-grid above 0.85 would tighten those CIs
(already noted in the paper).

---

## #9 — High-$\lambda$ ($\lambda=5,7$) boundaries are resolved by one grid point.

**Kill-shot.** "At $\lambda=7$ your transition falls between two grid points. The
$w_{\text{crit}}$ there is an extrapolation."

**Answer.** Acknowledged in-paper. The $w{=}0.85$ cell is *interior* (11/20 collapsed),
so the boundary is identifiable; only the *unpenalized* slope diverges, which is why we
use Firth. The behavioral conclusion does not depend on the exact high-$\lambda$ value —
it depends on the *monotone trend over seven points* and the non-overlapping endpoints.

**Residual risk.** Add $w\in\{0.80,0.90\}$ cells for camera-ready; trivial compute,
removes the objection entirely.

---

## #10 — The theorems are about a LATENT variable you can't measure.

**Kill-shot.** "Theorems 1–3 concern an order parameter $x$ you admit (line 850) is not
recoverable from policy statistics. Unfalsifiable math."

**Answer.** The theorems are deliberately *scoped to sign and form*, the only things
that don't require measuring $x$. Thm 3's conclusion ($\partial_\lambda w_{\text{crit}}>0$)
is an *observable* boundary shift, and it is *confirmed* empirically across 4 settings.
The latent $x$ is the vehicle; the *prediction* it generates is measurable and was
measured. We never claim to track $x$; we claim its law fixes a sign, and the sign shows
up in the data.

**Residual risk.** A theory hardliner may still call it decorative. Mitigation: in the
rewrite, state immediately after Thm 3 "the theorem's *content* is the observable
inequality, confirmed in Fig. (generalization); $x$ is a vehicle, not a claim."

---

## #11 — MPE loss-aversion curve is non-monotone at $\lambda=2,4$.

**Kill-shot.** "You say it 'replicates,' but the MPE curve dips at intermediate
$\lambda$. That's not replication, that's a noisy trend."

**Answer.** We report this explicitly and read MPE as a **direction**, not a strict
ordering: the *envelope* rises ($0.78\!\to\!0.89$), the $\lambda{=}7$ excess is
$+0.158\,[+0.108,+0.212]$ and the $\lambda{=}3$ excess $+0.101\,[+0.055,+0.150]$ —
both significant, the latter *quantitatively matching* the SC $+0.089$. The intermediate
dips are within seed noise of the null, consistent with MPE's shared-global reward being
noisier. The claim "loss aversion raises $w_{\text{crit}}$" is endpoint-significant on
every substrate.

**Residual risk.** Moderate. More seeds at MPE intermediate $\lambda$ would smooth it.
Honesty here is a trust asset; don't hide it.

---

## #12 — CTDE strictness: do your "decentralized" agents peek at global state?

**Kill-shot.** "You claim CTDE. Prove your actors don't receive global info at
execution."

**Answer.** Explicitly clean: in MAPPO conditions the **actor consumes only its local
observation** (plus a one-hot agent ID to break parameter-sharing symmetry); only the
**critic** sees the joint state, and only at training. This is textbook CTDE and we
state it for every centralized-critic result. The supply-chain PPO agents are
independent learners with local observations.

**Residual risk.** Very low. Keep the explicit "actor = local obs only" sentence in each
relevant section; reviewers audit this aggressively.

---

## #13 — "First quantitative phase boundary" — is the novelty claim safe?

**Kill-shot.** "Someone will find prior art and your 'first' becomes an overclaim,
poisoning trust in everything else."

**Answer.** We pre-empt the nearest neighbors explicitly: Yamaguchi 2025 (phase diagram
over *environmental* axes for *independent Q-learning*, holding $w$ fixed),
Capannoli 2026 (metastable boundary in *opinion dynamics* under a *fixed update rule*),
Mengesha 2025 (loss aversion moves a tipping point but in *EGT over fixed strategies*),
Lalmohammed 2025 (CPT in MARL but *entangled*, no boundary). Our delta is precise: a
*reward-structure* threshold in *gradient-trained* cooperative policies, *with CIs*, that
a *behavioral bias decomposed into components* then moves. We scope "first" to that exact
conjunction.

**Residual risk.** The literature moves fast. Mitigation: soften "first" to "to our
knowledge, the first … with this exact conjunction," and do a fresh arXiv sweep the week
before submission.

---

## #14 — Curvature effect flips sign across substrates — is your decomposition real?

**Kill-shot.** "Curvature lowers $w_{\text{crit}}$ on SC but *raises* it on LBF. If your
own decomposition doesn't replicate, why trust the loss-aversion half?"

**Answer.** We *predicted* this split: the mechanism says only the **loss-aversion sign**
is substrate-invariant (it inflates downside variance regardless of substrate), while the
**curvature sign depends on substrate-specific variance compression**. The claim ledger
labels curvature as substrate-specific *up front*. The sign flip on LBF is therefore a
*confirmation* of the scoping, not a contradiction — and it's evidence the decomposition
is measuring something real per-substrate rather than a fixed artifact.

**Residual risk.** Subtle argument; easy for a hurried reviewer to misread as
inconsistency. Mitigation: a single sentence — "the loss-aversion sign is invariant by
theorem; the curvature sign is substrate-specific by mechanism, and the LBF flip confirms
the latter."

---

## #15 — Is loss aversion confounded with reward scale / magnitude?

**Kill-shot.** "$\lambda$ multiplies the loss domain, so you also changed reward
*magnitude*. Maybe you're seeing a learning-rate-on-rewards effect, not psychology."

**Answer.** Two controls: (a) the **curvature condition** ($\alpha{=}\beta{=}0.88,
\lambda{=}1$) *also* changes magnitude but moves $w_{\text{crit}}$ the *opposite* way —
a pure magnitude story can't produce opposite signs from two magnitude changes; (b) the
**MPE/LBF behavioral labels are scale-invariant**, so a reward-magnitude artifact would
not register there, yet the effect persists. The asymmetry-around-reference, not the
scale, is what moves the boundary.

**Residual risk.** Low. Worth stating control (a) explicitly as an anti-confound.

---

## #16 — The metastability "slide" might just be undertraining.

**Kill-shot.** "Your boundary moves with $T$ because the short runs aren't converged.
That's not metastability, that's an unconverged baseline."

**Answer.** That *is* metastability, operationalized: collapse at finite $T$ means escape
time $\tau>T$; as $T$ grows a smaller SNR suffices, so $w_{\text{crit}}(T)$ descends —
and it descends along the predicted $-\ln T$ Kramers trend, not arbitrarily. The
basin-control experiment (Exp C) rules out initialization as the driver. And the slide
replicates on MPE and LBF. "Unconverged" would not predict a *specific functional form*
or a *deterministic limit*.

**Residual risk.** No asymptotic $w_{\text{crit}}$ bound is estimated (stated as a
limitation). A reviewer wanting $T\!\to\!\infty$ evidence won't get it; the honest answer
is "finite-horizon result, asymptotic bound is future work."

---

## #17 — The "downstream minority" result could be an artifact of node indexing.

**Kill-shot.** "Maybe 'downstream' just happens to be the node with the most leverage
for unrelated reasons (demand proximity, lead time)."

**Answer.** The cascade is *directional and asymmetric*: a downstream biased node moves
the boundary as far as a *fully* biased population (0.853 vs 0.851), while the same bias
upstream moves it far less (0.726). The asymmetry *mirrors the bullwhip structure* of the
substrate (perturbations propagate upstream along the ordering chain), which is a
mechanistic prediction, not a coincidence of indexing. We also vary which node and how
many.

**Residual risk.** Single-substrate (see #6). The bullwhip-mirroring argument is strong
but unprovable without a second directed-flow domain.

---

## #18 — Why these specific environments? Cherry-picking?

**Kill-shot.** "SC, MPE, LBF — did you try others that didn't work and drop them?"

**Answer.** The environments span **three distinct failure modes** by design:
resource-competition (SC hoarding), coordination-failure (MPE herding), public-goods
free-riding (LBF). They were chosen to be *structurally different routes to the same
boundary*, which is a stronger generality test than three similar gridworlds. We also
*report* where the phenomenon does *not* live (value-based/discretized learners go flat;
regulation-style tasks show no $w$-boundary) — the opposite of cherry-picking.

**Residual risk.** Low, and the reported negatives (ablations, failed substrates) are a
strong honesty signal. Keep them.

---

## #19 — Action-representation ablation: does discreteness kill it or not?

**Kill-shot.** "QMIX/IQL go flat, but you blame discretization, not value-learning. Yet
MPE is natively discrete and *works*. Your ablation contradicts itself."

**Answer.** Resolved by a clean single-variable control: **PPO on the same $K{=}11$
discrete shell also goes flat**, so it's *coarse discretization of a continuous control*,
not the value-based paradigm and not discreteness-per-se. MPE's *native* discreteness
visits the near-threshold region; the coarse-binned SC snaps to a stable interior level
and never visits the collapse region — exactly what the continuous-gradient mechanism
predicts. Not a contradiction; a scoped boundary condition.

**Residual risk.** The argument requires the reader to track "coarse-discretization-of-
continuous ≠ natively-discrete." State it in one crisp sentence.

---

## #20 — So what? The practical payoff may be unusable.

**Kill-shot.** "Even if true, a deployer can't measure their agents' $\lambda$ or freely
set $w$. The 'safety margin' is academic."

**Answer.** Two usable levers fall out: (a) where $w$ is cappable (privacy, comms,
policy), the boundary tells you whether your cap is *below* the requirement — a go/no-go
test; (b) where it is, the decomposition exposes $\lambda$ as a *second lever* — lower
the agents' effective loss aversion via the training objective (KTO-style objectives are
prospect-theoretic in form) instead of raising $w$. And the minority result changes
*where you look*: audit the most downstream biased participant, not the average. That's
an actionable design rule, not just a number.

**Residual risk.** Estimating $\lambda$ in a fielded system is itself open. Frame the
payoff as a *design-time* margin and an *audit-target* heuristic, not a runtime knob.

---

## Triage summary (what to actually fix before submission)

| Priority | Kill-shot | Action |
|----------|-----------|--------|
| **P0 — must address** | #6 single-substrate | Confident scoping rewrite; optionally a 2nd cascade substrate |
| **P0** | #2/#4 "it's a fit"/"post-hoc" | Foreground decomposition + ledger + sign theorem early |
| **P1 — cheap, high-value** | #8/#9 seeds & grid | Add $w\in\{0.80,0.90\}$ and 30 seeds at high $\lambda$ for camera-ready |
| **P1** | #13 "first" claim | Soften wording + fresh arXiv sweep pre-submission |
| **P1** | #11 MPE non-monotone | Keep honest; add seeds if time |
| **P2 — wording only** | #1,#3,#5,#10,#12,#14,#19 | One-sentence clarifications each |
