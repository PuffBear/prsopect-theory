# Claude Code session brief — two new HPC experiments

**Owner:** Agriya · **Paper:** *Measuring the Coordination-Collapse Boundary in
Policy-Gradient Cooperative MARL* (AAAI-27 submission) · **Date:** 2026-07

You are extending an existing, working experimental pipeline. **Do not reinvent it —
mirror it.** Your job is to add two experiments that strengthen the paper:

- **Experiment A — Second directed-flow substrate (or-gym).** Replicate the
  minority/cascade result on a *second* directed-flow topology, converting it from a
  "falsifiable prediction" into a *tested generalization*. This closes the paper's #1
  flagged weakness (the cascade is currently supply-chain-only).
- **Experiment B — Melting Pot Cleanup (standard benchmark).** Transfer the
  *direction* result (loss aversion raises `w_crit`) to a recognized standard
  social-dilemma benchmark. This answers the "no standard-benchmark run" reviewer
  concern with a thematically on-point benchmark.

Both can run in parallel. Read this whole file before writing code.

---

## 0. NON-NEGOTIABLE integrity rules (read twice)

The paper's credibility rests on **pre-registration**. Violating these voids the
result:

1. **Freeze the collapse label in git BEFORE running the main analysis.** Commit the
   label definition (a single boolean function of episode summary stats) with a
   timestamped commit message, and record the commit hash. No peeking at `w_crit`
   before the label is frozen.
2. **Do not tune anything to get a "clean" result.** If the boundary comes out noisy,
   wide-CI, or null, **report it as-is.** A messy honest result is fine; a
   manicured one is misconduct. The paper already reports failed probes openly.
3. **Match the existing protocol exactly** where comparability matters: same seed
   count (20/condition), same `w`-grid, same `λ`-grid, same CPT transform, same
   estimator (Firth + bootstrap). Only change what the new substrate forces you to.
4. **Keep everything anonymizable.** No names, emails, or absolute user paths in code
   or configs that will ship in the supplementary bundle.

---

## 1. Orient yourself in the existing pipeline (do this first)

Read these before writing anything, to learn the conventions you must mirror:

- `experiments/exp_cascade_continuous.py` — the cascade driver. **This is your
  template for Experiment A.**
- `experiments/exp_mappo_mpe_entropy.py` and `experiments/exp_mappo_lbf_entropy.py`
  — MAPPO transfer drivers on non-supply-chain substrates. **Templates for the
  Experiment B pattern** (learner setup, reward blend, seed loop, CSV output).
- Any `experiments/exp_*.py` that runs the or-gym `NetworkManagement-v1` main sweep —
  learn how `w`, `lambda_loss`, seeds, and the CPT transform are wired.
- `docs/expCascade_continuous_figures/cascade_continuous_raw_data.csv` and
  `docs/expPPO_lambda_sweep_figures/ppo_lambda_sweep_raw_data.csv` — the **output CSV
  schema** you must match (columns like `run_idx, w, lambda_loss, seed, mean_S,
  profit, ..., collapsed`).
- The analysis code that fits `P(collapse|w)=σ(b0+b1 w)`, computes `w_crit=−b0/b1`,
  bootstraps the 95% CI, and applies **Firth penalized logistic** for steep cells.
  Reuse this verbatim; do not re-implement the estimator.
- `run_*.pbs` at repo root — the PBS/HPC job template. Copy its structure for the new
  jobs.

**Key shared definitions (confirm against the code, don't take my word):**

- Reward blend (applied per agent, per step/episode as the code does it):
  `r_i = (1 − w) · r_i^local + w · r^global`.
- CPT transform applied **after** blending (never inside the optimizer):
  `r_i^PT = v(r_i)`, with `v(x) = x^α` for `x ≥ 0` and `v(x) = −λ (−x)^β` for `x < 0`;
  `α, β ∈ (0,1]` curvature, `λ ≥ 1` loss aversion. Risk-neutral null = `α=β=1, λ=1`.
- Standard grids: `w ∈ {0.5, 0.65, 0.75, 0.85, 0.95}` (add finer `{0.80, 0.90}` for
  steep high-λ cells), `λ ∈ {1,…,7}`, **20 seeds/condition**.
- Supply-chain pre-registered label (for reference): `1[ S̄ < 10 ∧ π̄ ≤ −127.1 ]`
  (`S̄` mean base-stock, `π̄` mean profit). You will define analogous labels per new
  substrate.

---

## 2. Experiment A — Second directed-flow or-gym topology

### Goal
Show the cascade/minority result is **not** an artifact of the one six-node network:
on a *different* directed-flow topology, (a) a `w_crit` boundary exists, (b) loss
aversion raises it, and (c) a single downstream loss-averse node can set the whole
network's requirement, with the effect governed by **path-concentration-to-demand**
(proximity to demand, not chain index).

### Substrate choice
Use a **different** or-gym directed-flow network than the paper's primary six-node
`NetworkManagement-v1` configuration. Options, in preference order:

1. A different `NetworkManagement-v1` **topology/parameterization** — different node
   count and/or branching structure (e.g., a deeper chain, or a different split/merge
   DAG), so the demand-path structure differs from the original.
2. `InvManagement-v1` (serial multi-echelon) if a branching variant is available.
3. A minimal custom DAG in the same or-gym style if neither gives a genuinely
   different topology.

**Requirement:** it must have (i) per-node local profit (`r_i^local`), (ii) a system
profit (`r^global`), (iii) a directed flow to a demand-facing node, and (iv) at least
one branch/merge so "path concentration to demand" is a *non-trivial* variable
(otherwise the position result is undefined). Document the topology as a small
adjacency description in the driver's header.

### Pre-registered label
Derive a collapse label analogous to the supply-chain one but **re-fit to the new
topology's cost/profit scale** (the `−127.1` profit threshold and `S̄<10` are specific
to the original network). Procedure:
1. Run a **small pilot** (few seeds, `λ=1`, a couple of `w` values) *only* to read the
   scale of `S̄` and `π̄` under obvious-collapse vs obvious-coordination.
2. Set the label thresholds from that scale (mirroring the original's logic: near-zero
   base-stock AND deeply negative profit).
3. **Freeze it in git with a timestamped commit** before the full sweep. Record the
   hash. Do not adjust it afterward.

### Design (mirror the cascade experiment)
- **Boundary + direction:** full `w × λ` sweep, 20 seeds, homogeneous agents. Fit
  `w_crit(λ)` with Firth+bootstrap. Expected: monotone rise with `λ`.
- **Dose-response:** sweep the most-downstream node's `λ ∈ {1,…,7}` with others
  risk-neutral. Expected: strictly monotone rise in `w_crit`.
- **Position:** place one `λ=7` node at each network position; regress `w_crit` on a
  path-concentration-to-demand measure. Expected: nodes feeding demand directly
  dominate.
- Learner: **independent PPO** (as in the primary supply-chain sweep), unless the
  topology requires the centralized-critic MAPPO used elsewhere — match whatever the
  original cascade driver uses.

### Outputs
- `docs/expOrgymB_cascade_figures/orgymB_cascade_raw_data.csv` (schema matching the
  existing cascade CSV; add a `node_position` / `biased_node` column for the position
  sweep).
- A short `RESULTS_A.md` summarizing: topology used, frozen label + commit hash,
  `w_crit(λ)` table with CIs, dose-response curve, position ranking, and a one-line
  honest verdict (did the cascade replicate, yes/no/partially).

---

## 3. Experiment B — Melting Pot Cleanup (standard benchmark)

### Goal
Transfer the **direction** result (loss aversion raises `w_crit`) to **Cleanup**, a
recognized standard social-dilemma benchmark whose public-good structure matches the
paper's local-cost/global-benefit mechanism. Commons Harvest is the backup substrate.

### Environment
- Suite: **DeepMind Melting Pot** (`google-deepmind/meltingpot`). Substrate:
  **`clean_up`** (primary), `commons_harvest__open` (backup). N = 7 agents.
- Observations are egocentric RGB → you need a **CNN encoder**. Use MAPPO
  (centralized critic) — reward is collective-facing and pixel-based, so match the
  MAPPO setup used for the MPE/LBF transfer drivers.
- Wrap Melting Pot to a PettingZoo-parallel API if the existing pipeline expects that
  (check the MPE/LBF drivers); otherwise adapt the SB3/MAPPO loop minimally.

### Reward blend (the crux — get this right)
- `r_i^local` = the agent's **own** collected reward (apples it personally eats;
  Melting Pot gives per-agent reward natively — use it directly).
- `r^global` = the **collective** signal. Use total team return per step
  (`Σ_j r_j`), OR a sustainability proxy (apple density / river-cleanliness) if the
  substrate exposes it. Prefer total team return for a clean analogue to `r^global`.
- Blend then transform, exactly as elsewhere:
  `r_i^PT = v( (1−w)·r_i^local + w·r^global )`, CPT `v` as in §1.
- **Cleaning has no individual reward** — that's the point (local cost, global
  benefit). Do not add a shaped cleaning reward; let `w` carry the incentive.

### Pre-registered label
Define a NEW collapse label for Cleanup and **freeze it before analysis**. Candidates
(pick one, justify in the driver header):
- Collective return over the episode below a frozen threshold (commons collapsed /
  free-riding dominated), OR
- Apple-density / sustainability metric below a frozen fraction of its coordinated
  level.
Set the threshold from a **small pilot** (as in §2), then commit + hash it. Do not
retune.

### Design
- Full `w × λ` sweep, `w ∈` standard grid, `λ ∈ {1,…,7}`. Match seed count to the
  other transfer substrates (**≥20 seeds**; see risk note below).
- Primary claim to test: **direction** — does `w_crit` rise with `λ`? (Magnitude and
  functional form are *expected* to be substrate-specific; that's fine and already the
  paper's stance.)

### Compute + risk (flag to the human)
- Pixel obs + CNN + N=7 + full `w×λ×seeds` is **heavy**. Budget GPU jobs; expect this
  to dominate the compute for both experiments. Write the PBS to request GPU.
- **Expect a noisier boundary** than the supply chain: partial observability + pixels
  → wider transition, looser CIs, possibly a less crisp logistic. The *direction*
  should still show. If CIs are too wide to resolve the direction at 20 seeds,
  **flag it and propose more seeds rather than silently massaging** — see §5.
- If `clean_up` is too heavy/noisy to resolve, fall back to `commons_harvest__open`
  (simpler dynamics) and note the substitution.

### Outputs
- `docs/expMeltingpot_cleanup_figures/meltingpot_cleanup_raw_data.csv` (schema:
  `run_idx, w, lambda_loss, seed, collective_return, own_return, sustainability,
  collapsed`).
- `RESULTS_B.md`: substrate, frozen label + commit hash, `w_crit(λ)` with CIs, a
  clear statement of whether the direction replicated, and the observed transition
  width vs the supply chain (honest comparison).

---

## 4. Analysis (shared, reuse existing code)

For every cell: fit `P(collapse | w) = σ(b0 + b1 w)`, estimate `w_crit = −b0/b1`,
bootstrap the 95% CI (same B as the existing pipeline). Use **Firth penalized
logistic** for steep/separated cells and confirm the two estimators agree (report the
max disagreement, as the paper does). Report transition width `Δw` alongside each
`w_crit`. **Do not invent a new estimator** — import the paper's.

---

## 4b. PBS / HPC job design (read before writing the `.pbs`)

**Scheduler is PBS/Torque** (there are existing `run_*.pbs`). The cluster-specific
directives — **queue/partition name, account/project code, module names, and the
conda/venv activation path** — are NOT knowable from this brief. **Lift them verbatim
from an existing `run_*.pbs`.** Do not invent queue names, module versions, or account
codes. If a GPU queue cannot be found in any existing script, STOP and ask the human
(see §6).

### Structure both jobs as PBS **array jobs**
- One array element = one run cell `(w, lambda, seed[, node_position])`. Fan the whole
  sweep out as an array, not a monolithic loop.
- Emit a **runmap** file first (`runmap_A.csv` / `runmap_B.csv`):
  `index -> (w, lambda_loss, seed, ...)`. Commit it. This is both the recovery map and
  the array index space. The driver reads the array index (`$PBS_ARRAY_INDEX`, or the
  variant the existing scripts use — check whether they are PBS Pro `-J` or Torque
  `-t`) and looks up its config from the runmap.
- **Idempotent + resumable:** each element writes to a **unique** result file
  (`results/expA/run_${idx}.csv`); a separate **merge step** then concatenates into the
  final `docs/exp*_figures/*_raw_data.csv`. If an element's output already exists and is
  complete, **skip it** — so requeued/failed elements re-run without redoing finished
  ones.
- **Fail loudly:** any exception -> nonzero exit, so failed elements show as failed in
  the scheduler instead of silently emitting empty rows.
- **Logs:** per-element stdout/stderr via `#PBS -o`/`-e` to
  `logs/expA/${PBS_JOBID}.${index}.{out,err}`.

### Experiment A — or-gym (CPU)
- Per element: 1 node, `ncpus=2-4`, `mem ~ 4-8GB`, **no GPU**.
- Walltime: size to one 300k-step PPO run **+ ~50% buffer**; measure the single-run
  time in the dry-run and set from that (likely well under 2h/run).
- Array size = number of rows in `runmap_A` (main `w x lambda x seed` sweep **plus**
  the dose-response and position sweeps). If the cluster caps array length or
  concurrency, chunk and/or use the `%<N>` concurrency cap.

### Experiment B — Melting Pot Cleanup (GPU)
- Per element: `ngpus=1`, higher `ncpus` for env workers (`~8`), `mem ~ 16-32GB`
  (pixel replay buffers), request the **GPU partition/queue** (lift from an existing
  GPU `.pbs`; if none exists, ask).
- Walltime: size to one Cleanup MAPPO run **+ generous buffer** — pixel CNN + N=7 +
  long episodes means **hours per run**; measure in the dry-run and pad heavily.
- **Checkpoint the training** (SB3 `save`/`load`) so a killed or preempted element
  resumes from its last checkpoint instead of restarting — essential for long GPU jobs.
- Cap concurrent GPU elements to the user's GPU allocation (`%<N>`).

### Before launching the full array (mandatory)
1. Submit a **1-element dry array** first to confirm the PBS directives, module loads,
   venv activation, and env import all work *on the cluster* (not just locally).
2. Only after that succeeds, submit the full array.
3. After completion, run the **merge + verify** step: assert every runmap index has a
   result row; print any missing indices for a targeted requeue.

---

## 5. Deliverables + final checklist

Produce:
- [ ] `experiments/exp_orgymB_cascade.py` + `run_orgymB_cascade.pbs`
- [ ] `experiments/exp_meltingpot_cleanup.py` + `run_meltingpot_cleanup.pbs`
- [ ] Frozen-label commits for **both** substrates, with hashes recorded in the
      driver headers and in `RESULTS_A.md` / `RESULTS_B.md`.
- [ ] Raw-data CSVs in the `docs/exp*_figures/` convention.
- [ ] `RESULTS_A.md`, `RESULTS_B.md` with honest verdicts.
- [ ] A run-index map per experiment (how many runs, seed→index mapping) so partial
      HPC failures are recoverable.

Before declaring done:
- [ ] **Dry-run each driver locally** with 1 seed / 2 `w` values / `λ∈{1,7}` to confirm
      it produces a well-formed CSV row and the reward blend + CPT are applied
      correctly (log the blended and transformed reward for one step and eyeball it).
- [ ] Confirm seed count, grids, and CPT match the existing supply-chain driver
      (comparability).
- [ ] Confirm no PII / absolute user paths in any shipped file.
- [ ] Confirm the frozen labels were committed **before** any full-sweep analysis.

## 6. Things to STOP and ask the human about

- If Experiment A can't find a genuinely *different* directed-flow topology in or-gym
  (i.e., the only option is a trivial relabeling of the same network) — flag it; the
  generalization claim needs a real topological difference.
- If Cleanup's boundary is unresolvable at 20 seeds (CIs span the grid) — report the
  numbers and propose a seed increase or the Commons Harvest fallback; **do not**
  pick thresholds to manufacture a boundary.
- If either result comes out **null or reversed** — that is a real finding. Write it
  up honestly in the RESULTS file and stop; the human decides how to frame it. Do not
  bury or retune it.
- If the cluster-specific PBS directives can't be recovered from existing `run_*.pbs`
  — specifically a **GPU queue/partition name, account/project code, or module
  versions** for Experiment B — STOP and ask; do not guess them (a wrong account code
  or queue silently fails or wastes the allocation).

---

*This brief is a spec, not a script. Prefer reading the existing drivers and matching
them over following my pseudocode literally. When the existing code and this document
disagree, trust the existing code and flag the discrepancy.*
