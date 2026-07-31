# Experiments still needed for the paper

Scope: only experiments (new runs / new analysis over existing runs). Pure text, figure-rendering, and reference fixes are handled in `CHANGELOG_edits.md` and its "author action" list. Each item says **what to run, why, what it unblocks, rough cost, and the pointer/reviewer it answers**. Priorities: **P0** = a referee can quote the current state as a gap or error; **P1** = converts a prediction into a result and materially raises the ceiling; **P2** = strengthens/insurance.

A note on what already exists (from the supplement + `run_*.pbs`), so nothing is re-run needlessly:
- SC-PPO horizon sweep: **4 points** (150k/300k/450k/900k) — the metastability fit exists.
- SC-MAPPO: λ-sweep at **one horizon only** (~700 runs) — **no horizon sweep.**
- MPE-MAPPO horizon: **2 points** (2M→8M). LBF-MAPPO horizon: **2 points** (1M→4M).
- Single-node dose-response and homogeneous λ-sweep both exist, but are **not overlaid or compared with a shared fit.**
- Action-representation ablation: **K=11 only.**
- MPE herding scale: **one value** (null wcrit=0.74), no sensitivity sweep.

---

## P0 — close before submission (referee-quotable gaps)

### P0-A. Reconcile the horizon run inventory (may be zero new runs)
**Do:** confirm the 450k and 900k SC-PPO horizon runs, the finer high-λ grid (w∈{0.80,0.90}), and the entropy-ablation runs actually exist on disk, and add their rows (with true seed/run counts) to Supplement Table 1. Delete/repair the "600k row of the main-paper heatmap" ghost text.
**Why:** the paper reports these horizons and the log-linear fit `wcrit = 2.44 − 0.138 ln T`, but the inventory table sums to exactly 1,524 **without** them, so the headline fit rests on runs the inventory doesn't list. This is the single most reviewer-lethal item.
**Unblocks:** the metastability claim; the main-text sentence I edited ("…is itemized in the Supplement").
**Cost:** likely **analysis only** (runs exist) — a few hours to re-tabulate. If any horizon is missing, re-run that one cell (see P0-B).
**Answers:** R1 item 2, R4 P0-5, pointer 33/36.

### P0-B. LBF collapse threshold — state the number
**Do:** report the exact LBF collapse statistic, aggregation window, and numeric threshold used (the same one already applied in code), plus its ±20% robustness like the supply chain.
**Why:** your contribution is *pre-specified labels*; two of three (SC, MPE) are numeric and the LBF one is currently a paraphrase. No new training — just surface the number from the eval code.
**Cost:** **analysis only.**
**Answers:** R1 item 5, pointer 27.

---

## P1 — turn predictions into results (highest ceiling-raising value)

### P1-A. SC-MAPPO horizon sweep (the experiment that makes a real Figure 1)
**Do:** run the SC-MAPPO λ=1 (risk-neutral) boundary at **3–4 horizons** matched in spirit to the SC-PPO set (e.g., scaled to MAPPO's convergence: pick 4 budgets spanning ~1 decade), 20 seeds/(w) cell, same w-grid and collapse label. Reuse `run_mappo_orgym.pbs` + `run_horizon_extra.pbs` as templates.
**Why:** the metastable slide currently replicates across environments only as **2-point** lines (MPE, LBF) plus the 4-point SC-PPO fit. A 4-point SC-MAPPO slide (a) gives the "horizon slide is not a supply-chain-PPO artifact" claim a second *log-linear* fit, and (b) supplies the missing panel for the 4-panel Figure 1 you wanted — which currently **cannot be built** because this data does not exist.
**Unblocks:** a genuine 2×2 Figure 1 (SC-PPO slide, SC-MAPPO slide, + boundary sharpness); the strongest generalization claim in the paper.
**Cost:** moderate — ~3–4 horizons × ~6–8 w-cells × 20 seeds ≈ 360–640 runs. R1 called this "the highest-value experiment left."
**Answers:** R1 items 37/38, R2/R3 on Figure 1, pointers 37, 38.

### P1-B. Third horizon point for MPE and LBF
**Do:** add one intermediate horizon to each (MPE: a point between 2M and 8M; LBF: between 1M and 4M), same cells/seeds.
**Why:** two points can only show *direction*; three let you claim the *log-linear* slide transfers, not just the sign. Cheap insurance that turns "endpoints move" into "the functional form travels."
**Cost:** low — ~2 environments × ~6 w-cells × 20 seeds × 1 horizon ≈ 240 runs.
**Answers:** completes P1-A's figure; pointer 37/38.

### P1-C. Single-node vs homogeneous dose-response overlay (your pointer 56 — the sharpest one)
**Do:** you already have both sweeps. **Fit the same one-parameter saturating form to each**, overlay the two curves with CIs on one axis, and test whether they coincide at intermediate λ (e.g., paired bootstrap of the curve difference at each λ). If the single node needs any missing intermediate λ, fill it in.
**Why:** single-node (0.586→0.856) ≈ homogeneous (0.586→0.852) at the endpoints. If they coincide *throughout*, the homogeneous λ-law is effectively **one node's law with four redundant agents attached** — which is a *sharper, more interesting* claim; if they diverge mid-range, that gap is the network's measurable contribution. Either way it pre-empts the reviewer's confound objection and upgrades the cascade finding.
**Unblocks:** the strongest reframing of the minority result; a cleaner Figure 6b (both curves overlaid).
**Cost:** mostly **analysis** over existing runs; at most a few fill-in λ cells (~40–80 runs).
**Answers:** R1 item 56, R2 pointer 56, pointer 56.

---

## P2 — strengthen / insure against obvious reviewer questions

### P2-A. Action-representation resolution sweep (K)
**Do:** repeat the discretized-PPO control at **K ∈ {5, 21, 51}** (you have K=11), same supply-chain boundary protocol.
**Why:** the scope-condition claim ("coarse discretization erases the boundary") rests on a single resolution. The first thing a skeptic asks is "does a finer grid bring it back?" If the boundary reappears monotonically as K grows, the claim becomes powerful rather than a one-point observation.
**Cost:** low–moderate — 3 K-values × ~6–8 w-cells × 20 seeds ≈ 360–480 runs. Template: the discrete-PPO control you already ran.
**Answers:** R2/R3 pointer 52.

### P2-B. More seeds on the supply-chain entropy cells
**Do:** add seeds to the SC entropy-injection cells so the SC arm's CIs either separate or clearly don't.
**Why:** the dissociation currently has one clean arm (LBF), one within-noise arm (SC, overlapping CIs), one null (MPE) — I've reframed the text honestly, but extra SC seeds could convert "directionally consistent" into a second clean confirmation. Optional; the text is now defensible either way. Template: `run_entropy.pbs`.
**Cost:** low.
**Answers:** R1 item 3, pointer 45.

### P2-C. MPE herding-scale sensitivity
**Do:** re-measure the MPE null boundary at 2–3 herding scales around the chosen one (which gives wcrit=0.74), and report that the loss-aversion *direction* is invariant to the scale.
**Why:** the MPE boundary is constructed by your wrapper; a reviewer may read the single tuned scale as "engineered to manufacture a boundary." A short sensitivity curve defuses that. Template: `run_mappo_mpe_highlambda.pbs` with varied herding coefficient.
**Cost:** low–moderate.
**Answers:** R1/R2 pointer 50.

---

## Explicitly NOT a quick experiment (research-grade open problem)

### The localizing observable (SNR / where-the-boundary-is)
Two pre-committed probes already failed to localize wcrit from drift/basin-flow statistics — this is now stated honestly as the central open problem, not a fixable gap. **Do not treat it as a to-do for this submission.** A predictive observable that estimates wcrit without a full w-sweep is future work; attempting and reporting another failed probe would not help the paper.

---

## Suggested ordering (given cost vs. payoff)
1. **P0-A, P0-B** (mostly analysis; unblock correctness).
2. **P1-C** (mostly analysis; your sharpest finding, cheap).
3. **P1-A + P1-B** (the real Figure 1; the biggest ceiling-raiser; ~600–900 runs total).
4. **P2-A** (insurance on the scope claim), then **P2-C**, **P2-B** if time/compute remain.

If you can only run one thing: **P1-A (SC-MAPPO horizons).** If you can only *analyze* one thing: **P1-C (the dose-response overlay).**
