# If This Paper Disappeared

*The hidden question every strong AAAI reviewer asks: "If this paper vanished tomorrow,
would the multiagent community lose an important idea — or just another benchmark
result?" This document answers that in one page. It is written to **become the
Introduction**: the spine below is the order the intro should follow.*

---

## The one-paragraph answer

If this paper disappeared, the field would lose the first **measuring instrument** for a
failure it has only ever described in words — and the first evidence that the failure is
governed by the **psychology** of the agents, in a direction that is **provable before
the experiment is run**, and set by a **single participant in the right structural
position** rather than the population. The community would be left exactly where it is:
saying "centralization helps" without being able to say *how much*, *when it stops
helping*, or *who breaks it first*.

---

## What, concretely, the field loses

**1. It loses the conversion of a slogan into a quantity.**
Cooperative MARL has a well-documented failure — *coordination collapse*, the
lazy-agent policy where agents abandon cooperation for a degenerate zero-action
strategy. The entire literature treats it qualitatively: "push reward centralization
$w$ too low and cooperation dies." *Where* it dies — the critical $w_{\text{crit}}$ at
which collapse probability crosses one-half — had never been measured. Without this
paper, $w_{\text{crit}}$ remains a thing everyone gestures at and no one reports with an
error bar. The loss is not a number; it is the *ability to have the number at all* —
a pre-registered collapse label, a logistic boundary, bootstrap CIs — a template any
cooperative-MARL paper could adopt to stop hand-waving about centralization.

**2. It loses the discovery that the boundary is alive, not fixed.**
The field's mental model treats collapse as a property of an environment — it collapses
or it doesn't. This paper shows $w_{\text{crit}}$ is a **metastable** threshold that
*slides* with the training horizon ($0.822\!\to\!0.666$) along a predicted Kramers
$-\ln T$ trend, replicating across three environments. The lazy attractor is escapable.
Lose the paper and the community keeps a false static picture of a dynamic phenomenon —
and keeps mis-budgeting training for cooperative systems, because the minimum viable
centralization *depends on how long you train*.

**3. It loses the bridge between behavioral economics and multi-agent learning.**
Agents are increasingly trained to act like people — through preference-aligned
objectives that are themselves prospect-theoretic in form — or to act *alongside* people.
The single most robust fact about human choice is **loss aversion**. We know what loss
aversion does to one decision-maker; we did not know what it does to the *coordination*
of many learning together. This paper answers it: loss aversion **raises** the
centralization a cooperative team needs to avoid collapse, and — distinctively —
probability-weighting curvature pushes the *opposite* way. Decomposing prospect theory
*against a measured boundary* is the move prior CPT-in-MARL work could not make because
it folded the components together. Lose this and the behavioral-MARL literature stays
entangled: unable to say which part of human-like risk preference helps coordination and
which part poisons it.

**4. It loses an a priori guarantee where the field usually has only correlations.**
MARL results are almost always empirical: "we observed X." This paper proves the **sign**
of the loss-aversion effect *before any data* — a sign invariant of the policy-gradient
escape barrier (Theorem 3). The *direction* is a prediction of the mechanism, not a fit
to the outcome, and it is then confirmed across two algorithms and three environments.
Lose the paper and the field loses a rare worked example of a *mechanistically forced
direction* in deep multi-agent learning — the kind of result that turns an observation
into a principle.

**5. It loses "position beats prevalence."**
The most surprising, most portable idea here is that the boundary is set not by the
*average* bias but by a **minority at a particular network position**: a single
*downstream* loss-averse agent fixes the whole system's centralization requirement,
indistinguishable from a fully biased population, while the same agent upstream barely
moves it. Stated generally — *in any directed-flow cooperative system, the centralization
requirement is set by the most downstream biased participant, not the mean* — this is a
law-shaped claim with reach into mechanism design, organizational economics, and the
safety of mixed human–agent teams. Lose the paper and this prediction is never put on the
table, never falsified, never built on.

---

## What would *not* be lost (the honest boundary of the claim)

The field would not lose a new neural architecture, a SOTA score, or a closed-form value
of $w_{\text{crit}}$ — and that is the point. This paper's worth is not a benchmark win.
Its worth is an *instrument, a fact, a law, a guarantee, and a design rule*, none of which
exist elsewhere. A paper whose disappearance would only cost the field "another 5% on a
benchmark" fails the hidden question. This one removes a *capability* the community
currently does not have: the capability to measure where cooperation breaks, and to say
who breaks it.

---

## Spine for the Introduction (use this order)

1. **Hook:** cooperative systems now mix human decision-makers with agents trained to act
   like them; both inherit loss aversion. (Why the reader should care.)
2. **The gap:** coordination collapse is well-documented but *never measured* — "where"
   in $w$ it happens has no number. (The unmeasured boundary.)
3. **Why prior work doesn't close it:** risk-sensitive MARL ≠ loss aversion; CPT-in-MARL
   is entangled; EGT shows bias moves tipping points but over fixed strategies, no learned
   policies. (The clean novelty delta.)
4. **Why the supply chain is the *only* substrate where the real question is askable:**
   directed flow makes "upstream/downstream" definable, so we can ask not just *whether*
   bias matters but *where in the network* it matters most. (Lead with this — it is the
   best justification, not the fifth.)
5. **The move:** treat collapse like a phase boundary — pre-registered label, logistic
   estimate, bootstrap CIs — then use the boundary as an *instrument* to decompose
   prospect theory against it.
6. **The one hypothesis sentence (verbatim, repeated in abstract & conclusion):**
   *Human-like loss aversion forces cooperative learning agents to require more reward
   centralization to avoid collapse — a shift we predict in sign a priori and confirm
   across three environments and two algorithms — and a single downstream loss-averse
   agent can set that requirement for the whole system.*
7. **Findings preview + contribution ladder** (the five distinct rungs).
