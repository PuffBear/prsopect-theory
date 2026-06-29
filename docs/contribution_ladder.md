# Contribution Ladder

*One sentence each. The test: a reviewer reading these five in sequence should never
ask "isn't 2 just a restatement of 1?" Each rung is a different **kind** of claim —
an artifact, a discovered fact, a behavioral law, a mathematical guarantee, a
deployment rule — so that no two compete for the same slot in a reviewer's memory.*

---

**1. Core contribution (the *artifact* — what we built).**
We construct the first quantitative, confidence-intervalled **phase boundary of
coordination collapse** in deep cooperative MARL: a pre-registered binary collapse
label, logistic estimation of $P(\text{collapse}\mid w)$, and bootstrap CIs on the
critical reward-centralization $w_{\text{crit}}$ — turning the field's qualitative
"centralization helps" into a measured object with error bars.

**2. Scientific contribution (the *fact* — what we discovered about the world).**
Using that boundary as an instrument, we discover that $w_{\text{crit}}$ is **not a
constant but a metastable threshold** that slides left with the training horizon and
is set by reward structure and agent psychology — i.e. coordination collapse is an
escapable, structured phenomenon, not an absorbing fixed property of an environment.

**3. Behavioral contribution (the *law* — what human-like bias does to it).**
We show that the two components of prospect theory move the boundary in **opposite
directions** — probability-weighting curvature lowers $w_{\text{crit}}$ while loss
aversion raises it (the loss-aversion direction replicating across three
environments and two algorithms) — and that the boundary is set by a **minority by
network position**, a single downstream loss-averse agent fixing the whole system's
centralization requirement.

**4. Theoretical contribution (the *guarantee* — what is true a priori, not just observed).**
We prove that the **sign** of the loss-aversion shift is fixed before any data are
seen — a sign invariant of the Kramers escape barrier under the mean-field
policy-gradient dynamics (Theorem 3) — so the direction of the effect is a
*prediction* of the mechanism, not a curve fit to the result.

**5. Practical contribution (the *rule* — what a deployer should do Monday).**
The boundary is a **measurable safety margin**: it tells a practitioner how much
reward centralization a cooperative system needs for reliable coordination, how much
*more* once its agents are loss-averse (quantified: $+0.089$ at the human-range
$\lambda\!\approx\!3$, up to $+0.166$ at $\lambda\!=\!7$), and *which* participant —
the most downstream biased one — sets that requirement.

---

## Why these five are mutually exclusive (the anti-confusion table)

| Rung | Question it answers | Object type | Could it exist without the others? |
|------|--------------------|-------------|-------------------------------------|
| 1 Core | "What did you build?" | A measurement instrument | Yes — methodology stands alone |
| 2 Scientific | "What did you learn?" | An empirical fact (metastability) | Needs 1 to measure, but is a separate claim |
| 3 Behavioral | "What does bias do?" | A directional + locus law | Needs 1; independent of 4's proof |
| 4 Theoretical | "Why must it be so?" | An a priori sign guarantee | Pure math; would hold even if 3 failed empirically |
| 5 Practical | "What do I do with it?" | A deployment decision rule | Consumes 1–4; adds the design-margin reading |

**The one-line elevator version (if a reviewer only remembers one sentence):**
*We turn coordination collapse into a measured boundary, prove that human-like loss
aversion must shift it in a sign we predict a priori, show it generalizes, and show a
single downstream biased agent can set it for the whole team.*
