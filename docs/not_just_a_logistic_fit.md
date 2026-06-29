# Why This Is Not Just a Logistic Threshold Fit
### An adversarial appendix, written against ourselves

*We write this section as the hostile reviewer would, then answer as the paper. The
charge is the most dangerous one we face because it is the most superficially plausible:
that everything here reduces to fitting a sigmoid and reading off its midpoint. If that
charge stuck, the paper would be a homework exercise. It does not stick, for seven
independent reasons. We state the prosecution in full before the defense.*

---

## The prosecution (stated at full strength)

> "Strip away the vocabulary — 'phase boundary,' 'metastability,' 'Kramers escape,'
> 'mean-field Langevin' — and what remains is mechanical. The authors run an environment
> at several values of a reward-mixing weight $w$, record a binary outcome, fit
> $P(\text{collapse}\mid w)=\sigma(b_0+b_1 w)$, solve $-b_0/b_1$, and bootstrap a
> confidence interval. That is a one-dimensional logistic regression — a technique from
> the 1950s. Every downstream 'result' is just this fit repeated at different parameter
> settings and the midpoints compared. The physics framing is decoration over a curve
> fit. Reject as incremental."

We accept this is *exactly* what a tired reviewer will think on a first skim. Here is why
it is wrong.

---

## The defense: seven things a logistic fit cannot do

### 1. A fit reports a number; we predict the number's *behavior* — and pre-registered the predictions.

A logistic fit is descriptive: it tells you where the midpoint *is*. It says nothing about
where the midpoint *should be*, *should move*, or *why*. Our mean-field account commits, in
a ledger separating *derived-no-fit* from *calibrated*, to predictions a fit cannot make:
the **opposite signs** of the two prospect-theory components; the **monotonicity** of
$w_{\text{crit}}(\lambda)$; the **affine functional form** of $\phi(\lambda)$ with a slope
*bounded a priori to $[0,1]$*; the **sharpness** of the transition; the **sign of the
horizon slide**; the **sign of the exploration-noise effect**. These were stated as
predictions *before* the corresponding fits and confirmed afterward. A logistic regression
has no mechanism to be *wrong in advance*; ours could have been, and wasn't.

### 2. The affine law is tested *out of sample* — a fit cannot do this to itself.

We fix the two parameters of the loss-aversion law ($g/c$ and the loss-domain share
$\rho$) on three calibration points ($\lambda\in\{1,3,7\}$), *freeze* them, and then
**predict** four held-out points ($\lambda\in\{2,4,5,6\}$). The frozen one-parameter law
tracks all seven points at **RMSE 0.014**. This is a generalization test of a *theory*,
not a fit of a curve to its own data. A logistic threshold fit, by contrast, is evaluated
on the data it was fit to; it has no held-out structure to predict. The very existence of
an out-of-sample test proves there is a model above the fit.

### 3. The *sign* is a theorem, not a regression coefficient.

A logistic fit's slope sign is whatever the data make it. Our loss-aversion direction is
fixed by **Theorem 3** — a sign invariant of the escape barrier: from
$\partial_\lambda F<0$ pointwise it follows that
$\partial_\lambda w_{\text{crit}}^{\text{Emp}}>0$ *regardless of the noise magnitude*.
The direction is guaranteed before a single run. When the fits then come back with that
sign across four environment–algorithm settings, that is **confirmation of a proof**, not
a coincidence of curve-fitting. No amount of logistic regression yields a guaranteed sign;
it only yields an observed one.

### 4. The decomposition is *causal attribution*, impossible from a single fit.

The scientific payload is not any one midpoint but the **difference** between midpoints
under a controlled three-way design: null ($\alpha{=}\beta{=}1,\lambda{=}1$), curvature
($\alpha{=}\beta{=}0.88,\lambda{=}1$), and curvature+loss-aversion. Because the optimiser
is **byte-for-byte identical** across conditions (CPT applied as a reward transform *after*
blending, never inside the update), the shift in $w_{\text{crit}}$ is *attributable* to a
named psychological component. This is the logic of a controlled experiment, not of a fit.
A logistic regression on pooled data could *never* separate the curvature channel (which
moves the boundary **left**) from the loss-aversion channel (which moves it **right**) —
it would average them into a muddle. The opposite-signs result is the signature that we
are decomposing a cause, not describing a curve.

### 5. Metastability is a *dynamical* claim with a functional form — a static fit has no time axis.

A logistic fit in $w$ has no notion of training horizon. We show $w_{\text{crit}}$ is not a
constant but *slides* with $T$ ($0.822\!\to\!0.666$), along the **$-\ln T$ Kramers trend**
the escape picture predicts, with a **deterministic $T\!\to\!\infty$ limit**
$w_{\text{crit}}^0=c/(c+g)$ toward which it descends. We rule out the mundane alternatives:
initialization is excluded by the basin-control experiment (Exp C, near-zero
init-vs-converged correlations); the slide *replicates* on MPE and LBF. "Just a fit" cannot
predict that the midpoint will move in a specific direction, at a specific rate, toward a
specific limit, as a *third* variable changes. That is a model of the dynamics.

### 6. A dissociation that could have failed — and a failed probe we report.

Two facts no curve-fitter would manufacture. First, a **dissociation**: raising
exploration noise moves the boundary on the supply chain and LBF but **not on MPE**,
*exactly where the mechanism says intrinsic exploration already dominates the
signal-to-noise ratio*. A prediction that the effect is *absent* in a specific place, borne
out, is the opposite of curve-fitting (a fit always "explains" its data). Second, we
**pre-registered a direct test** of the signal-to-noise mechanism (Appendix E) and report
that its drift-sign observable came back **inconclusive** — and we keep it in the paper.
A logistic-fit-with-a-story would have deleted the failed test; a model under genuine test
reports it. The mechanism survives because it stands on its sign-and-form predictions,
which the failure does not touch — and we say precisely that.

### 7. The locus result is not expressible as a threshold at all.

"A single downstream loss-averse node sets $w_{\text{crit}}=0.853$, indistinguishable from
a fully biased population (0.851), while the same node upstream moves it only to 0.726" is
a statement about **network position and cascade direction**. A logistic fit in $w$ has no
vocabulary for *which agent* or *where in the flow*; it cannot represent the asymmetry that
bias propagates *upstream* along the ordering chain (mirroring the bullwhip structure). The
result emerges only because we treat the boundary as an *object that can be probed* —
perturb one node, re-measure the boundary — not as a number to report.

---

## The synthesis: the logistic fit is the *thermometer*, not the discovery

The honest core of the rebuttal is this analogy. A thermometer is a simple, century-old
instrument; reading one is trivial. But *measuring the boiling point of water as a function
of altitude*, discovering it is **not constant**, predicting the **direction** it moves
before you climb the mountain, proving that **one** dissolved impurity in the right place
shifts it as much as saturating the whole pot, and writing down the **law** that governs
all of it — none of that is "just reading a thermometer," even though a thermometer is the
only instrument used.

The logistic regression is our thermometer. The paper is the altitude law. We use the
instrument precisely *because* it is simple and trustworthy — a complicated estimator would
itself become a confound. The contribution was never the estimator. It is (i) the
pre-registered discipline that makes the reading trustworthy, (ii) the controlled
decomposition that turns a reading into a cause, (iii) the theorem that fixes a direction
before the reading, (iv) the dynamical metastability law, (v) the cross-substrate
replication of the direction, and (vi) the network-position locus result. Remove the word
"logistic" and every one of these stands. That is the test of whether a method is the
contribution or merely the tool: *delete the tool's name and see if the science survives.*
Here, it does.

---

### One-sentence version for the rebuttal box

> The logistic fit is the thermometer, not the finding: our contribution is the
> pre-registered boundary used as an *instrument* to (a) causally decompose prospect theory
> into opposite-signed components, (b) confirm a sign we *prove* a priori, (c) establish a
> metastability law in the training horizon, and (d) show a downstream minority sets the
> boundary — none of which a threshold fit can express, predict, or attribute.
