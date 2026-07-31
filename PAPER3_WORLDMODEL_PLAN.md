# Paper 3 — plan & literature review

**Working title (placeholder):**
*The Lazy Imaginer: Measuring the Imagination-Sharing Collapse Boundary in
Multi-Agent World Models*

**One-sentence pitch:** paper 1 measured a coordination-collapse boundary in
*reward* mixing; this paper transplants the same pre-registered instrument onto a
new, architecturally distinct axis — *imagination* mixing in multi-agent world
models — and asks whether an RLVR-trained agent's local world model goes *lazy* as
imagination centralization rises, at a measurable critical point.

**Status:** planning + verified lit review. Depends on papers 1 (instrument) and 2
(shape/scope) existing. Date: 2026-07-04.

**Why this is the ambitious one:** it sits at the unclaimed overlap of *three*
literatures that do not yet touch — multi-agent world models (have the centralization
axis, no RLVR, no measured boundary), RLVR-World (has RLVR, single-agent only), and
your own collapse instrument (has the boundary, never pointed at imagination). Top
venues (ICLR/NeurIPS) reward exactly this kind of three-way gap closure.

---

## 1. The core idea (two coupled dials, one instrument)

Paper 1 mixed **reward**:
`r_i^PT = v((1-w) r_i^local + w · r^global)`.

Paper 3 introduces a second, architecturally distinct dial — **imagination mixing** —
because in a world-model agent, laziness can occur *before the reward is ever
computed*, at the level of whose latent dynamics the agent thinks with:

```
z_i^{t+1} = (1 - w_wm) · f_i^local(z_i^t, a_i^t)  +  w_wm · f_shared(z^t, a^t)
```

- `w_wm = 0` → each agent unrolls its *own* independent RSSM-style transition
  (the MA-Dreamer "naive imagination" regime; diverges under partial observability).
- `w_wm = 1` → a joint transition conditioned on all agents' latents/actions
  (the fully-centralized MAWM regime; hits the "global context unavailable at
  decentralized execution" wall).

Couple this to RLVR-World's move (decoded prediction-accuracy as a *verifiable*
reward), applied per agent:

```
r_i^WM = w_wm · V(ẑ_shared, z^true)  +  (1 - w_wm) · V(ẑ_i^local, z^true)
```

**Research question:** as `w_wm → 1`, does agent i's local transition `f_i^local`
degrade toward uninformativeness — a *lazy imaginer* contributing nothing to the
joint latent state because the shared model's gradient dominates — and is there a
critical `w_wm^crit` where this collapses, measurable with the *same* pre-registered,
bootstrapped protocol as paper 1's `w_crit`?

**Laziness label (new, pre-registered):** not a profit/inventory threshold, but
agent i's *marginal contribution to joint prediction accuracy* collapsing below a
frozen floor:
`Δ_i = V(ẑ_shared) − V(ẑ_shared∖i) < floor`.

**The payoff either way:** two coupled critical points — `w_crit` (reward-level
laziness, paper 1) and `w_wm^crit` (imagination-level laziness, new). Whether they
*coincide or are structurally different* is a real finding regardless of which way it
comes out. That is the hypothesis-can't-lose structure reviewers like.

## 2. Literature review — four clusters + the gap table

Mirror paper 1's reviewer-tested structure (each cluster ends with one sentence on
what's covered, one on what's precisely missing), but with **four** clusters. All
papers below were verified to exist via web search (2026-07-04); forward-dated
2025/2026 arXiv IDs must be re-confirmed to resolve before submission — fabricated
references are an explicit AAAI/ICLR ethics violation.

### Cluster 1 — World models, single-agent
- **Dreamer / DreamerV3** (Hafner et al., *Mastering Diverse Domains through World
  Models*, arXiv:2301.04104) — policies trained entirely on imagined latent rollouts.
- **RLVR-World** (Wu, Yin, Feng, Long, NeurIPS 2025, arXiv:2505.13934) — *first* to
  train the world model itself with RLVR, using decoded prediction-accuracy as the
  verifiable reward instead of MLE; text games, web nav, robot manipulation.
  Explicitly single-model, single-agent.
- *Covered:* RLVR-for-world-models exists and works. *Missing:* never extended past
  one agent.

### Cluster 2 — World models, multi-agent
- **MA-Dreamer** (arXiv:2204.04687) — naive independent imagination diverges under
  partial observability; proposes *shared imagination* as the fix.
- **CoDreamer** (arXiv:2406.13600) — decentralizes it via latent message passing to
  synchronize local world models.
- **Decentralized Transformers with Centralized Aggregation** (arXiv:2406.15836) and
  the C-MAWM / D-MAWM survey framing — names both failures: full centralization needs
  "global context unavailable under decentralized execution"; decomposed MAWMs exist
  specifically to "address centralization limitations."
- **GAWM: Global-Aware World Model** (arXiv:2501.10116); **Dreaming of Others**
  (arXiv:2605.31361); **MetaMind** (arXiv:2603.00808) — *verify IDs before citing.*
- *Covered:* the centralization axis is *implicitly present* (naive vs. shared
  imagination; C-MAWM vs. D-MAWM). *Missing:* nobody sweeps the degree of imagination
  sharing as a **continuous variable with a measured critical point.**

### Cluster 3 — Lazy agents & coordination collapse
- **Sunehag et al.** (VDN) — origin of the lazy-agent problem.
- **Liu et al. 2023** — causal formalization of fully-lazy agents/teams.
- **Your paper 1** (coordination-collapse boundary) — laziness-from-centralization
  measured at the *reward* level, with a CI'd boundary.
- **LLM lazy-agent / free-riding in multi-agent RLVR** (emerging thread; verify a
  concrete cite) — laziness now documented at the *reasoning* level.
- *Covered:* laziness-from-centralization is documented at the reward level (classical
  MARL) and the reasoning level (LLM RLVR). *Missing:* never at the
  **imagination / world-model level.**

### Cluster 4 — RLVR beyond single-model reasoning
- **Tülu 3** (Lambert et al., arXiv:2411.15124) — coined RLVR for LLM post-training.
- **RLVR-World** (as above) — RLVR expands from LLM post-training to *world-model*
  post-training.
- Token/turn-level credit-assignment fixes (UCAS/PACR-style) — single-policy only.
- *Covered:* RLVR's expansion from LLMs to world models. *Missing:* its multi-agent
  credit-assignment behavior is unexamined in the **world-model** setting.

### The gap table (build this FIRST — it *is* the contribution)

Every existing row has "no" in the last two columns. Filling that column is the gap
statement.

| Paper | Substrate | Sharing mechanism | Training signal | Centralization = continuous var? | Collapse measured w/ boundary + CI? |
|---|---|---|---|---|---|
| DreamerV3 | single-agent | n/a | MLE | no | no |
| RLVR-World | single-agent | n/a | **RLVR** | no | no |
| MA-Dreamer | multi-agent | naive → shared imagination | MLE | no (binary) | no |
| CoDreamer | multi-agent | decentralized msg-passing | MLE | no | no |
| D-MAWM / C-MAWM | multi-agent | centralized vs decomposed | MLE | no (architectural) | no |
| GAWM | multi-agent | global-aware | MLE | no | no |
| Liu et al. 2023 | multi-agent | reward-level | RL | no | partial (causal, no CI) |
| **Your paper 1** | multi-agent | **reward mixing (continuous w)** | PT-transformed RL | **yes** | **YES (CI'd)** |
| **This paper** | multi-agent | **imagination mixing (continuous w_wm)** | **RLVR** | **yes** | **yes (CI'd) — the gap we close** |

**Gap sentence (drops straight into related work):** *"Prior multi-agent world models
vary the degree of imagination sharing architecturally (MA-Dreamer, C-MAWM/D-MAWM) but
none treats it as a continuous parameter with a measured critical point, and no work
combines RLVR-trained world models with a multi-agent setting at all — the gap we
close."* Same move paper 1 makes against Yamaguchi/Capannoli; ICLR reviewers respond
to the table because it *is* the evidence rather than an assertion.

## 3. What must be built (new runs — nothing before AAAI paper-1 deadline)

1. A minimal multi-agent world-model testbed with an explicit `w_wm` blend between a
   per-agent RSSM `f_i^local` and a joint `f_shared`. Start from a Dreamer/MA-Dreamer
   base; the blend is the one architectural addition.
2. RLVR-World-style verifiable reward `V(·)` = decoded prediction accuracy, applied
   per agent as in the `r_i^WM` equation.
3. Pre-registered lazy-imaginer label `Δ_i < floor` (marginal contribution to joint
   prediction accuracy), frozen in version control before analysis — exactly paper 1's
   protocol, new observable.
4. Sweep `w_wm ∈ [0,1]`, ≥20 seeds/cell, logistic fit + bootstrap CI on `w_wm^crit`.
5. **The money experiment:** measure `w_crit` (reward) and `w_wm^crit` (imagination)
   in the *same* environment and test whether they coincide. Either answer is a result.

## 4. What makes this ICLR/NeurIPS-worthy (vs. borderline)

- **Three-literature gap closure** — the table is unambiguous and reviewer-legible.
- **The instrument is already reviewer-tested** (paper 1) — you are transplanting a
  validated protocol onto a new axis, not inventing measurement from scratch.
- **A dissociation-style result is built in** — two critical points that may or may
  not coincide is the same "prediction that can be null" structure that made paper 1's
  entropy dissociation strong.
- **RLVR is hot and the multi-agent world-model gap is real and current** (2025–26
  papers actively circling it without closing it).

## 5. What it explicitly does NOT claim

- Not a new world-model *architecture* — the contribution is the *measured boundary*
  on the imagination-sharing axis, plus the RLVR-multi-agent-world-model first.
- Not a magnitude prediction of `w_wm^crit` from first principles (paper 1's
  calibrated-not-predictive limit likely carries over; be honest about it up front).
- Not a claim that `w_wm^crit = w_crit` — that coincidence is the *question*, tested.

## 6. Collaboration / sequencing

- Natural venue: ICLR or NeurIPS (world-model + RLVR crowd), not AAAI-MS.
- Strong Yamaguchi/others angle less direct here; better co-author is someone in the
  Dreamer/world-model lineage.
- Sequence: paper 1 (AAAI, instrument) → paper 2 (shape/scope, AAMAS/AAAI) →
  paper 3 (this, world-model transplant, ICLR/NeurIPS). Paper 3 is the highest-ceiling
  and highest-effort of the three; do not start build before paper 1 is submitted.

---

## Verified-real references (checked 2026-07-04)

- DreamerV3 — arXiv:2301.04104 ✓
- RLVR-World (NeurIPS 2025) — arXiv:2505.13934 ✓ (thuml/RLVR-World repo confirmed)
- MA-Dreamer — arXiv:2204.04687 ✓
- CoDreamer — arXiv:2406.13600 ✓
- Decentralized Transformers w/ Centralized Aggregation (C/D-MAWM) — arXiv:2406.15836 ✓
- GAWM — arXiv:2501.10116 ✓
- Tülu 3 (coined RLVR) — arXiv:2411.15124 ✓

**Verify before citing (forward-dated, surfaced but not confirmed to resolve):**
Dreaming of Others 2605.31361, MetaMind 2603.00808, and any specific LLM-lazy-agent
RLVR paper for Cluster 3. Do NOT cite any of these until you personally open the arXiv
page — fabricated/incorrect references are a punishable ethics violation.
