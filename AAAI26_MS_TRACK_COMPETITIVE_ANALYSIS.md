# AAAI-26 Multiagent-Systems Track — Competitive Analysis vs. `main_standard`

*Comparison set: 20 accepted AAAI-26 MS-track papers in `papers/`. Reference paper: your
"Coordination-Collapse Phase Boundary in Policy-Gradient Cooperative MARL" (`latex/main_standard.tex`).*

---

## 1. Generalization claim — verified safe

I read the full experiments sections of the two nearest neighbors:

- **Geng et al. (16339)** validate their theory by varying **learning rate α and graph degree k**, with
  theory↔simulation cross-validation. All experiments are on a **two-state Prisoner's Dilemma on a regular
  lattice** — a single environment class.
- **Leung et al. (26290)** test **robust cooperation across population sizes (20–50)** and learning/exploration
  rates, all within the **iterated Prisoner's Dilemma** — a single environment class.
- **Capannoli et al. (23560)** "generalize" only analytically (results extend to arbitrary integer stubbornness);
  no experiments.

**Conclusion:** none of the three replicates an effect *across distinct environments* or *across learning
algorithms*. Your claim — same qualitative effect across **3 environments (supply chain, MPE, LBF) × 2 algorithms
(PPO, MAPPO)** — is the broadest *kind* of generalization in the cohort and is safe to state as a distinguishing
contribution. The honest framing: prior MARL-dynamics findings establish robustness *within* a setting; you show
a behavioral effect *transfers across* settings and algorithms.

---

## 2. Where your paper sits — category

Your paper is an **empirical + mechanistic-theory study of a multi-agent learning phenomenon** (measure a phase
boundary, explain its sign a priori via a mean-field SNR argument, show it generalizes). It is **not** an
infrastructure paper, **not** a method/architecture paper, **not** an algorithm paper.

Only **3 of 20** share this category: **Geng (16339)**, **Leung (26290)**, **Capannoli (23560)**.
Two are theory-leaning borderline: **Asmar (28724)**, **Aruta (12849)**.
The other 15 are method/architecture, pathfinding/combinatorial algorithms, security eval, or infrastructure.

---

## 3. Structure & page budget across all 20

All papers are AAAI-26 camera-ready, **2-column, 9 pages** (body + refs within the standard limit) except:
**27666 (Fu B., Amazon) = 11 pages** (extra ref/appendix pages), **12892 = 8 pages**.
Reference counts run **~17–59**; behavioral/economics-flavored related work is absent across the board.

| # | ID | Short title | Pages | ~Refs | Category | Formal theory? |
|---|----|-------------|-------|-------|----------|----------------|
| 1 | 04800 FuJ | Counterfactual Planning (LLM agents) | 9 | 40 | Method | No |
| 2 | 05260 ChenB | G-UBS implicit-feedback recsys | 9 | 49 | Method | No |
| 3 | 07031 GaoZ | GRDC role discovery + comm (MARL) | 9 | 40 | Method/arch | No |
| 4 | 07978 ChenC | GUI-Eyes visual grounding | 9 | 58 | Method | No |
| 5 | 09360 HuD | Orion personalized web agent | 9 | 25 | Method | No |
| 6 | 11125 HuJ | TAPA programmatic agents | 9 | 53 | Method | No |
| 7 | 12060 AritaT | Local guidance for MAPF | 9 | 30 | Algorithm | No |
| 8 | 12849 ArutaM | HumanATL[F] (logic, bounded rationality) | 9 | 50 | Theory (logic) | **Yes (5+)** |
| 9 | 12892 Kidushim | Robust Multiagent Combinatorial PF | 8 | 17 | Algorithm | Partial |
| 10 | 14479 JainR | Graph-attention MAPF | 9 | 40 | Method/algo | 1 |
| 11 | **16339 GengY** | **Fast-slow Q-learning dynamics** | 9 | 36 | **Empirical+theory** | Derivation-heavy |
| 12 | 20075 GuoL | Agent-SAMA mobile GUI agent | 9 | 38 | Method | No |
| 13 | 21682 HuY | GRIP budget-constrained routing | 9 | 35 | Method | No |
| 14 | 23248 ChenJ | Adversarial attack on MAS | 9 | 47 | Security eval | No |
| 15 | **23560 Capannoli** | **Phase transition, opinion dynamics** | 9 | 59 | **Theory** | **Yes (2)** |
| 16 | 23798 HuT | Unreal-MAP platform | 9 | 44 | **Infrastructure** | No |
| 17 | **26290 LeungC** | **Learn to cooperate, minimal observ.** | 9 | 46 | **Empirical+theory** | No |
| 18 | 27053 CatesB | Mental-model lie generation | 9 | 32 | Theory/method | **Yes (4)** |
| 19 | 27666 FuB | BRaP dense rearrangement (Amazon) | 11 | 36 | Algorithm | No |
| 20 | 28724 AsmarD | Shared action suggestions (Dec-POMDP) | 9 | 43 | Method+complexity | Complexity |

### Section templates of the 3 most-similar papers

- **Geng (16339):** Introduction → Model (Environment: Stochastic Game) → Theoretical Derivation
  (SDE→FPE→Full Characterization) → Experiments → Conclusion. *Theory-first, single-environment validation.*
- **Leung (26290):** Introduction (+ inline Related Work) → Preliminaries → Model (Observer / Local-Info
  Models) → Experimental Results (Emergence of Cooperation) → Conclusions & Future Work. *Model-then-experiments.*
- **Capannoli (23560):** Introduction → Our Contribution → Related Work → Model and Results → Results
  → Interpretation → Conclusion & Future Work. *Pure analytical; main theorem = phase transition.*

Your structure (collapse definition → measured boundary → SNR theory → multi-setting replication → minority/
network-position result) is closest to **Geng's theory-first template** but adds the cross-environment
replication block none of them have.

---

## 4. Novelty claims — genuine novelty vs. rephrasing

**Genuinely novel theory (new formal object / new result):**
- **Capannoli (23560):** new nonlinear opinion model; *main theorem identifies a phase transition* —
  genuine analytical novelty. Closest to your "phase boundary as a formal object" stance.
- **Aruta (12849):** new logic HumanATL[F] with fuzzy + resource-bounded semantics; new model-checking
  complexity results (P / NP-complete / Δ₂P-complete) — genuine theory.
- **Asmar (28724):** new complexity-reduction result for Dec-POMDP-Com via shared action suggestions
  (NEXP → tractable regime) — genuine, though deliverable is an algorithm.

**Theory present but more *framework/derivation* than new theorem:**
- **Geng (16339):** unifies SDE→Fokker-Planck description of networked Q-learning. Strong, but it's a
  *new application/derivation* of established statistical-physics machinery (replicator dynamics, mean-field,
  Barfuss-style deterministic limits) to graph-structured stochastic games — closer to rigorous re-derivation
  + extension than a brand-new theorem. **This is the same posture as your SNR account**, so frame yours
  carefully: your *sign-invariant proof* is the part that reads as genuinely novel theory; the mean-field SNR
  scaffolding is, like Geng's, an application of known machinery.

**No formal theory — novelty is the artifact/method (the 15 others):**
Every method paper (GRDC, Orion, TAPA, GUI-Eyes, Agent-SAMA, GRIP, G-UBS, AdapAM, BRaP, MAPF variants)
claims novelty in a *system, framework, or benchmark*, not in a result about behavior. Unreal-MAP (23798)
is explicitly infrastructure.

**Takeaway for your framing:** your strongest novelty is the *quantified phase boundary + a priori sign
invariant*. Present the SNR derivation honestly as "applying mean-field machinery to fix the sign," not as
new theory — reviewers in this cohort (cf. Geng) will recognize the lineage. The behavioral angle
(loss aversion governing coordination) is **unique in the cohort** — see §6.

---

## 5. Reusable references (harvested from the 3 nearest papers)

These appear in Geng/Leung and are directly on-topic for your MARL + cooperation + collective-dynamics framing.
**Verify each before citing** (titles transcribed from PDF text).

### Core — collective learning dynamics & cooperative AI (from Geng 16339)
- **Barfuss, Donges, Kurths (2019)** — *Deterministic limit of temporal-difference RL for stochastic games*,
  Phys. Rev. E 99(4):043305. → direct precedent for your mean-field/deterministic-limit move.
- **Barfuss et al. (2020)** — *Caring for the future can turn tragedy into comedy…under risk of collapse*,
  PNAS 117(23). → "collapse"-framed collective action; thematically perfect.
- **Barfuss et al. (2025)** — *Collective cooperative intelligence*, PNAS 122(25). (also in Leung) → flagship
  framing cite for "collective" cooperation.
- **Hammond et al. (2025)** — *Multi-agent risks from advanced AI*, arXiv:2502.14143. → safety-margin framing.
- **Hughes et al. (2018, NeurIPS)** — *Inequity aversion improves cooperation in intertemporal social dilemmas.*
  → behavioral-preference × MARL precedent; closest existing work to your loss-aversion angle.
- **McAvoy et al. (2025)** — *Unilateral incentive alignment in two-agent stochastic games*, PNAS 122(25).
- **Su, McAvoy, Wang, Nowak (2019)** — *Evolutionary dynamics with game transitions*, PNAS 116(51).
- **Hu, Leung, Leung (2019, NeurIPS)** — *Mean-field theoretic approach to multiagent Q-learning dynamics.*
- **Hussain et al. (2025)** — *Multi-Agent Q-Learning Dynamics in Random Networks: Convergence due to
  Exploration and Sparsity*, arXiv:2503.10186. → ties exploration noise to dynamics (your noise ablation).
- **Pan et al. (2025, ICLR workshop)** — *Why do multiagent systems fail?* → failure-mode motivation.
- **Leibo et al. (2017)** — *MARL in sequential social dilemmas* (you already cite this).
- **Dafoe et al. (2020)** — *Open problems in cooperative AI* (also in Leung).

### Foundational cooperation / evolutionary (from Geng + Leung)
- Nowak (2006), *Five rules for the evolution of cooperation*, Science 314.
- Ohtsuki, Hauert, Lieberman, Nowak (2006), Nature 441 — cooperation on graphs.
- Hilbe, Šimsa, Chatterjee, Nowak (2018), Nature 559 — evolution of cooperation in stochastic games.
- Axelrod (1984), *The Evolution of Cooperation*.

### Reinforcement-learning cooperation (from Leung 26290)
- Pérolat et al. (2017) — multi-agent RL model of common-pool resource appropriation.
- Multiple "direct punishment / social payoff / reputation in MARL" entries — useful if you discuss
  incentive mechanisms shifting cooperation.

> **High-value gap fill:** Hughes et al. (2018) "inequity aversion" + Barfuss (2019/2020) are the most
> citation-worthy additions — they are the *existing* bridge between behavioral preferences and MARL
> collective dynamics, which is exactly the bridge your loss-aversion paper extends.

---

## 6. Strategic positioning summary

1. **Category:** empirical + mechanistic-theory MARL paper. Nearest peers: Geng (16339), Leung (26290),
   Capannoli (23560). Position against these in related work; cite Unreal-MAP (23798) as available
   infrastructure, not a competitor.
2. **Generalization moat:** you are the only one replicating an effect across *environments and algorithms*;
   the peers vary parameters within one environment. State this explicitly.
3. **Behavioral moat:** **0 of 20** papers touch prospect theory, loss aversion, or Kahneman/Tversky. Your
   behavioral-economics framing is unoccupied in this cohort — lean on it.
4. **Theory honesty:** present the SNR mean-field part as *applied known machinery* (à la Geng), reserve the
   "novel theory" label for the sign-invariant proof and the downstream-minority prediction.
5. **Format:** 9 pages, 2-column, ~35–50 refs is the norm — you are well within expectations.
