# RESULTS_A — Second directed-flow substrate (cascade generalization)

**Verdict (one line): NOT RUN — the search for a second directed-flow substrate was
already conducted and exhausted; the cascade stays supply-chain-only and is reported
in the paper as future work. No new sweep was launched.**

This is the honest outcome the brief's §6 asks for: *"If Experiment A can't find a
genuinely different directed-flow topology in or-gym (i.e., the only option is a
trivial relabeling of the same network) — flag it; the generalization claim needs a
real topological difference."* That condition is met, and the decision predates this
session.

---

## Why no new substrate sweep was run

Experiment A asks for a **second** directed-flow topology on which to replicate the
minority/cascade result. The requirements are strict: the substrate must have
(i) per-node local profit, (ii) a system profit, (iii) a directed flow to a
demand-facing node, (iv) at least one branch/merge so "path concentration to demand"
is a non-trivial variable, **and** — critically — (v) a real `w_crit` collapse
boundary (local vs. global incentive tension that `w` can trade off), plus
(vi) be installable/gateable on this stack.

A full search against exactly these requirements was carried out (2026-06-22 → -23)
and every candidate fails:

| Candidate | Directed flow? | Outcome |
|---|---|---|
| **or-gym `NetworkManagement-v1` topology variants** (brief option 1) | yes | **Same inventory family** as the paper's primary substrate. A reviewer already flagged a v1 variant as "the same env." A topology relabel does **not** close the weakness — it re-triggers the "same substrate" objection. |
| **or-gym `InvManagement-v1`** (brief option 2) | yes | Same inventory family (serial multi-echelon of the *same* model class). Same objection as above. |
| **MPE / LBF / MAgent (battle, pursuit)** | no | Peer-symmetric: **no upstream/downstream axis**. The cascade's independent variable — minority *position* along a directed flow — does not exist here. Category mismatch, not a missing run. |
| **RWARE** | yes | Exploration wall robust to all four geometry-safe fixes (deeper request queue, longer horizon, single-agent, built-in TWO_STAGE reward). Remaining fixes contaminate the hoarding geometry. See memory `rware-4th-substrate-gate`. |
| **SMAC / SUMO (sumo-rl)** | SUMO yes | Cluster-only binary wall (no SC2/SUMO binary on this arch; multi-week integration, high wall-risk, no local gating). |
| **gym-anm (ANM6Easy power feeder)** | yes | Best find: pip-installable, genuinely directed power flow, principled per-node reward decomposition. **Full CPT + w-dial wrapper built (`env/anm_wrapper.py`), plumbing gate PASSED.** But the w-dial produces **NO `w_crit` boundary**: the 5-point interior curve is flat and non-monotone and the two direction probes disagree → training variance, not a boundary. Structural cause: ANM6Easy is a voltage/loss *regulation* task on a tiny 6-bus feeder where local and global objectives are nearly aligned, so there is little local-vs-global tension for `w` to trade off. |

**Bottom line:** no cheap, installable, gateable, *tension-having* second directed-flow
MARL substrate exists. The only genuinely-new flow benchmark (SUMO) has the worst
risk/gateability profile of everything tried, and the closest installable one
(gym-anm) has no boundary to measure.

This is a research finding, not an implementation gap. It is recorded as the frozen
project decision in memory `cascade-generalization-closed` (with the full search log),
so it is not silently re-run.

## Consistency with the paper

The current submission already scopes the cascade honestly as single-substrate and
does **not** rely on a second flow env:

- `AuthorKit27/AnonymousSubmission2027.tex:559–561` — "…only under a directed resource
  flow, which MPE and LBF lack. We therefore report the cascade as a **falsifiable
  prediction, not a tested generalization**; testing it needs a second directed-flow
  substrate, and is future work, not an oversight."
- `AuthorKit27/AnonymousSubmission2027.tex:600–605` — Scope and Future Work paragraph:
  a second directed-flow domain "would make it a tested generalization rather than a
  [prediction]."

RESULTS_A therefore confirms and does not contradict the paper. If a larger gym-anm
network (with real congestion, hence genuine incentive tension) or a new installable
flow benchmark becomes available, the parked `env/anm_wrapper.py` decomposition is the
correct starting point — but that is speculative and out of scope for this submission.

## What was *not* done, deliberately

- No `experiments/exp_orgymB_cascade.py` / `run_orgymB_cascade.pbs` were written. Per
  the brief's own escalation clause and the frozen decision, building a
  `NetworkManagement-v1` relabel would produce a same-family result that a reviewer
  can (and did) dismiss as "the same env" — i.e., it would not strengthen the paper,
  and could weaken it by looking like an attempt to dress up the same substrate as two.
- No collapse label was re-fit for a new topology (there is no new topology to fit to).

**Integrity note:** this file reports a null/closed result as-is, per the brief's
non-negotiable rule 2. Nothing was tuned or manufactured to produce a boundary.
