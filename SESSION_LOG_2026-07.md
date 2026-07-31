# Session Log — AAAI-27 "Coordination-Collapse Boundaries" paper

Complete record of everything done in this working session: HPC jobs, experiments,
three review rounds and the responses, all code and paper changes, decisions, and open
items. Written so a future session (or a human) can pick up with full context.

**Paper:** *Measuring Coordination-Collapse Boundaries in Reward-Mixed Cooperative MARL*
(AAAI-27 submission). **Cluster:** `sharc` (PBS Pro, `ssh sharc`, repo
`~/prospect-theory`, conda env `marl`, queues `cpu`/`gpu`, `-J` array syntax; login-node
compute is FORBIDDEN — always `qsub`). **Local repo:** `/Users/Agriya/Desktop/prospect theory`.

---

## 0a. UPDATE (2026-07-31 continuation session) — paper is submission-ready

**Everything below section 0's original TL;DR is from the prior session. This update covers
a long continuation session that took the paper from "weak-accept, 6/10, cascade job running"
to a fully reconciled, submitted-looking state.** Read section 11 onward for the full record;
short version:

- **Two more independent reviews came in (rounds 4 and 5) and were answered in full**: ϱ
  computed and found only partially invariant (honestly reported, not oversold), cascade
  position CIs added (revealed the fine-grained ordering was noise), horizon-asymptote hedge
  added, multiplicity statement added, abstract restored to the user's original registered
  wording, Eq. 3/affine-law and the entire SNR mechanism section **removed from the main paper
  and moved to supplement-only** (freed a full page), two full-width figures widened for
  legibility, arithmetic reconciled across every table (a real pre-existing 0.856→0.863 stale
  number was caught and fixed).
- **The separate-actor cascade job was run TWICE.** The first full run (job `129320[]`, 500
  cells, ~2 days wall-clock) had a **training-exposure bug**: each of the 5 independent actors
  got 5x more real per-agent environment interaction than agents got under the shared-actor
  baseline (SuperSuit's vec-env packs 5 agents into one env, so shared-actor's 300k
  `total_timesteps` is only 60k real ticks/agent; the separate-actor driver gave each
  independent actor a full 300k real ticks). This suppressed almost all collapse and made the
  experiment unable to answer its question. **Diagnosed, archived (not deleted,
  `results/expCascadeSep_5xoverexposed_v1`), fixed (timesteps 300k→60k), and rerun** (job
  `129394[]` + cleanup resubmits for walltime-killed stragglers, ~4h wall-clock at the
  corrected 1x exposure). **Final, trustworthy result: the position ranking does NOT survive
  without a shared gradient** — node_2 (strongest under the shared actor, rank 1) becomes the
  *weakest* under independent actors (rank 5); node_4 (weakest, rank 5) rises to rank 2. This
  is a genuine rank-inversion, not a reshuffle. Written into both papers as one continuous
  story (planned deconfound → result), not a pivot.
- **Both `_clean.pdf` submission files (main + supplement) were audited and passed**:
  metadata empty, zero identity/path leaks (byte-level scan), checklist properly filled in,
  anonymized code/data archive confirmed to exist and match what the supplement's text claims.
- **A real AAAI-27 policy gap was caught and (per user, addressed) closed**: the Method
  footnote originally said code/data would be "released on acceptance," which the AAAI-27
  reproducibility policy explicitly says earns zero credit at review time. Anonymized
  code/data archive (`aaai2027_code_data_archive_curated.zip`) now exists and is referenced
  correctly in the supplement.
- **Open/uncertain**: whether the horizon sweep was ever extended past 900k (an experiment,
  not a bug fix — ~5h/cell at 1.8M steps, ~500 cell-hours for a full new horizon point; was
  discussed as a cost estimate, not confirmed as run). Confirm before assuming it's in the
  paper.

---

## 0. TL;DR — where things stand now (ORIGINAL, prior session — see 0a above for current state)

- **Paper is at weak-accept / borderline (my read: 6/10, poster-worthy).** Three
  independent reviews went **4/10 → 5/10 → 6/10** as we revised; all converged on the
  same residual weaknesses, which we then addressed.
- **Working paper files are the `_3` copies** (`AnonymousSubmission2027_3.tex`,
  `SupplementaryMaterial2027_3.tex`). The `_2` versions are the reviewed originals, left
  untouched. Both `_3` compile clean (0 undefined refs), main body fits the 7-page AAAI
  limit (Ethics + References spill to p8, which don't count), supplement ~10pp (unbounded).
- **All prose em dashes removed** from both `_3` files (41 total), replaced with
  context-appropriate punctuation; en dashes (`actor--critic`, `10--90`) kept.
- **One experiment is RUNNING on the cluster:** the separate-actor cascade deconfound
  (job `129314[]`, 500 cells). Last successful check: running, 0 results yet. Its result
  updates the cascade claim when done.
- **2,700 control runs + reanalysis are DONE and folded into the paper.**
- **Nothing is git-committed** (user's choice); everything lives as working-tree files.

---

## 1. Starting brief: CLAUDE_CODE_NEW_RUNS.md (two proposed experiments)

The session opened working `CLAUDE_CODE_NEW_RUNS.md`, which asked for two new HPC
experiments. Both hit the "trust existing code, flag discrepancy" clause:

### Experiment A — second directed-flow substrate (cascade generalization)
- **Decision: CLOSED, not run.** The search for a 2nd directed-flow substrate was already
  exhausted and frozen (memory `cascade-generalization-closed`): or-gym variants = same
  inventory family (reviewer's "same env"); MPE/LBF = no flow axis; RWARE = exploration
  wall; SUMO/SMAC = binary wall; gym-anm = built the full wrapper, plumbing passed, but
  NO w_crit boundary (objectives too aligned).
- **Deliverable:** `RESULTS_A.md` documents the closed search + consistency with the paper
  (which already scopes the cascade as future work). No new substrate sweep written.

### Experiment B — Melting Pot Cleanup (standard benchmark)
- Built the full driver stack for the REAL DeepMind Melting Pot `clean_up`:
  `env/meltingpot_cleanup_wrapper.py`, `env/meltingpot_collapse.py`,
  `experiments/exp_meltingpot_cleanup.py`, `experiments/analyze_meltingpot_cleanup.py`,
  `run_meltingpot_cleanup.pbs` (+ dry-run pbs). CNN-MAPPO, 13-point w-grid × λ{1..7} ×
  20 seeds = 1820 cells, frozen-threshold label, runmap, checkpoint/resume.
- **BLOCKED at run time:** the real `meltingpot` depends on `dmlab2d`, whose Linux wheels
  need **glibc ≥ 2.31**; the cluster is **CentOS 7 / glibc 2.17** with no container
  runtime. Confirmed via install attempts on py3.11 AND a fresh py3.10 env (dmlab2d
  reports zero installable versions; the wheel exists but requires newer glibc).
- **Retargeted (user's call):** built `experiments/exp_cleanup_wlambda.py` +
  `env/cleanup_wlambda_collapse.py` + `run_cleanup_wlambda.pbs` against the in-repo
  pure-NumPy `env/cleanup_env.py` (same Leibo-2017 tragedy-of-commons, runs on glibc 2.17).
  Dry-run verified locally and on the cluster. NOT launched as a full sweep (superseded by
  the controls work that became the priority).
- **Deliverable:** `RESULTS_B.md` (substrate decision + glibc wall documented).
- HPC facts learned here: `pbsnodes` shows a `gpu` queue (compute3/4) but advertises no
  `ngpus` and `naccelerators=0` everywhere; qsub syntax `qsub -q gpu -l select=1:ncpus=8:ngpus=1`;
  PBS Pro `-J`; no account/project code needed.

---

## 2. The three decisive controls + Part-2 reanalysis (2,700 runs, all PASSED)

User pivoted to a higher-priority request: three reward-transform controls + a nonparametric
reanalysis, to answer the weak-reject review's confound criticisms.

### CPT wrapper changes (`agents/cpt_wrapper.py`) — shared file, all backward-compatible
Added three params, all defaulting to identity (verified: `v(-4) = -23.709` unchanged, so
every other experiment is unaffected):
- `reference_point` (Control A): `v((blended - z0)/scale)`, kink moves off 0.
- `local_scale`, `global_scale` (Control C): scale local/global reward BEFORE the blend.
- `output_scale` (Control B): divide the FINAL CPT reward by a per-λ constant (RMS match).

### Part 2 — Reanalysis (NO new training) — `experiments/reanalysis_controls.py`
- Added **isotonic** (nonparametric monotone) + reused logistic/Firth; **seed-paired**
  bootstrap (resampling unit = seed, drawn once and applied to both λ); raw collapse
  counts per w; boundary-region continuous-metric plots; explicit method reporting.
- **Result: the λ=1→7 rightward shift survives in ALL 3 envs** under logistic/Firth/isotonic,
  seed-paired CI excludes 0 everywhere: Supply **+0.269** [.232,.306], MPE **+0.208**
  [.156,.275], LBF **+1.03** [.789,1.664].
- MPE/LBF collapse labels recomputed with the paper's own recalibrated θ (stored CSV
  `collapsed` columns were under an un-recalibrated threshold). LBF λ=7 flagged
  right-censored (boundary above the w-grid).
- Outputs: `REANALYSIS.md` + `docs/reanalysis_controls/`.

### Control A — Reference point (job `128837[]`, 1200 runs) — ROBUST
- Arms z0 ∈ {0, mean −2.99, median −6.77, −0.5σ −5.14} × λ{1,3,7} × 5 w × 20 seeds.
- z0 measured by a pilot (job `128831`, light: 2 runs × 20k after the first pilot `128821`
  died on walltime), frozen to `refpoint_z0.json`, pre-registered via
  `PREREGISTRATION_z0.md` (written anchor, NOT git — user's choice).
- **Shift survives EVERY arm:** z0=0 +0.269, mean +0.228 [.187,.272], median +0.183
  [.122,.269], −0.5σ +0.193 [.141,.247]. All CIs exclude 0. frac_loss diagnostic rises
  0.29→0.49→0.58 with λ (mechanistic corroboration).
- Files: `experiments/exp_control_a_refpoint.py`, `analyze_control_a_refpoint.py`,
  `run_control_a_refpoint.pbs`, `run_control_a_pilot.pbs`.

### Control C — Reward-unit sensitivity (job `128838[]`, 900 runs) — EXPECTED
- Arms {baseline, 2× local, 2× global} × λ{1,3,7} × 5 w × 20 seeds. No pilot needed.
- **w_crit MOVES** (2× local → 0.779, +0.194 at λ=1; 2× global → 0.476, −0.109), so w is a
  NOMINAL mixture coefficient, not scale-free. **But the λ-shift PERSISTS in every arm**
  (baseline +0.269, 2×local +0.130 [.105,.153], 2×global +0.329 [.263,.430]).
- Motivates reporting the normalized incentive ratio ϱ=(1−w)cℓ/(w·cg) alongside w.
- Files: `experiments/exp_control_c_rewardunit.py`, `analyze_control_c_rewardunit.py`,
  `run_control_c_rewardunit.pbs`.

### Control B — Scale/variance match (job `128855[]`, 600 runs) — NOT amplification
- Arms {raw, stdmatched} × λ{1,3,7} × 5 w × 20 seeds.
- Per-λ CPT-reward RMS = 7.93 / 17.06 / 31.87 (λ=1/3/7), i.e. λ inflates magnitude ~4×.
  stdmatched arm divides by output_scale {1, 2.15, 4.02} (pilot job `128854`, frozen to
  `rms_output_scale.json`, `PREREGISTRATION_rms.md`).
- **Shift SURVIVES RMS-matching: +0.145 [.107,.185]** (vs raw +0.269). Shrinks but a clear
  residual remains → loss ASYMMETRY, not magnitude, drives w_crit.
- Files: `experiments/exp_control_b_scalematch.py`, `analyze_control_b_scalematch.py`,
  `run_control_b_scalematch.pbs`, `run_control_b_pilot.pbs`.

### HPC lessons from the controls
- **HARD RULE (saved to memory `no-login-node-compute`):** never run compute (train/pilot/
  dry-run) on the login node; always `qsub`. Violated twice early (killed both), then fixed.
- PBS Pro here **rejects the `-J X-Y%N` concurrency-cap suffix** ("illegal -J value") —
  use plain `-J 0-N`.
- Single-threaded PPO ≈ **50 min per 300k cell**; pilots must be tiny (20k, few w/seeds) or
  they hit walltime.

---

## 3. Master results CSV + an integrity audit (7 checks)

User asked to verify integrity before trusting the numbers. Ran 7 checks:
1. RMS constants from fixed pre-treatment data — ✅ (pilot → frozen JSON → main reads it, raises if absent).
2. Same calibration rule at every w — ✅ (output_scale has no w dependence; 1 distinct value per λ).
3. Bootstrap resamples common seed IDs — ✅ (one seed multiset per replicate, applied to both λ).
4. Every CI from the same 2000-replicate procedure — ✅ (single N_BOOT constant).
5. **Baseline-value consistency — my check was WRONG and the user corrected it.** I claimed
   0.666/0.685 "appear in zero files / are rounding variants of 0.585." **FALSE.** They are
   DIFFERENT experimental conditions: **0.585/0.586 = curvature baseline (α=β=0.88, λ=1);
   0.685 = linear/null baseline (α=β=1, λ=1, file `docs/expN_figures/expN_raw_data.csv`);
   0.666 = structural null / 300k horizon point (`docs/expA300k_figures/`).** I had grepped
   only the reanalysis tree, not the original run tree. The difference between 0.585 and
   0.685 IS the curvature result — collapsing them would have deleted the paper's decomposition.
   Verified the decomposition reproduces the paper exactly: deployment shift λ3 = 0.774−0.685
   = **+0.089** (abstract headline), λ7 = **+0.166≈+0.167**; curvature = 0.586−0.685 = **−0.100**.
6. LBF crossings outside [0,1] — partially: flagged `extrapolated`, isotonic n/a; master CSV
   marks `right_censored_at=0.95`.
7. Run counts reported as **training runs** (each row = one PPO.learn), not "cells": A=1200,
   B=600, C=900 = 2700 control training runs.
- **Frozen master CSV: `/tmp/MASTER_RESULTS.csv`** (rebuilt complete with linear anchor, every
  row labelled by (α,β,λ) so the two baseline axes never conflate). Showcase-only, not written
  into experiment dirs.

---

## 4. Review round 1 (weak-reject, 4/10) — response

First reputable-professor review. Most concerns were already conceded in the paper; the
actionable ones we had ALREADY answered with the controls above. Built `REVISION_PLAN.md`
(exact LaTeX edits, page-budget math) covering 8 points, then implemented into the `_3`
copies:
- **Controls A/B/C + seed-paired bootstrap** → new main-paper paragraph + supplement section+table.
- **Isotonic/probit + goodness-of-fit** → main sentence + supplement subsection.
- **Full hyperparameters** → replaced "SB3 defaults" with an explicit table (lr 3e-4, γ 0.99,
  GAE 0.95, clip 0.2, 10 epochs, 2×64 tanh, etc.).
- **MPE herding formula (κ=4) + LBF config (`Foraging-5x5-2p-2f-v3`, θ=0.626)** → supplement.
- **Code-artifact paragraph**; **mechanism language softened** ("An SNR Hypothesis"); **2
  citations** (DE-MADDPG, packet-routing).
- **Page fight:** additions pushed body to 8–9pp; did a disciplined redundancy-only
  compression pass + moved **Figure 6 (cascade) to the supplement (as Fig. 3)** with all 3
  main-paper refs rewired to static "Supplement Fig. 3". Landed at 7 body pages.

---

## 5. Review round 2 (weak-accept, 5/10) — response

Reviewing the revised `_3`. Two genuinely NEW sharp points; the rest overlap round 1.
- **§1 shared-policy confound:** main said "independent PPO", supplement said
  "shared-parameter". Confirmed in code: ONE shared PPO actor (`concat_vec_envs` +
  single `PPO`), identical global obs, no agent-ID. Reviewer's confound is REAL: a shared
  gradient can move the boundary from one biased node. **Fix:** consistent "parameter-shared
  PPO" wording; a rebuttal sentence using the position ranking to rule out a *uniform*
  shared-gradient artifact. **(Note: round 3 corrected this — see below.)**
- **§2/Q6 w-non-invariance:** promoted the incentive ratio; precise bootstrap-clustering
  statement (runs for absolute, seed-blocked for differences); Control C numbers inline.
- **§4 mechanism:** further shortened, labeled interpretive conjecture.
- **§6 transfer labels:** added a calibration subsection (MPE θ=0.144, LBF θ=0.626 fixed
  from λ=1 endpoints BEFORE treatment).
- **Minor:** 0.666/0.685 reframed as one condition measured twice; "sharp"→"resolved";
  Mao/Gong/Xiao citation; "pointwise loss-asymmetric transform" note.
- Re-compressed to hold 7 pages.

---

## 6. Review round 3 (weak-accept, 6/10) — response + the running experiment

Reviewing the further-revised `_3`. Best score yet. Key points:
- **§5 — the reviewer FALSIFIED my confound rebuttal.** My "a shared gradient cannot depend
  on position" claim was too strong: even with identical obs FORMAT, nodes at different
  positions see different obs VALUES / visitation / advantage distributions, so a shared
  actor CAN produce position-specific effects. **Fixed:** softened to "rules out a *uniform*
  shared-gradient artifact; full deconfounding needs a separate-actor replication."
- **Decision 1 (user chose 1b):** made the incentive ratio **co-primary** (abstract +
  conclusion reframed: w_crit is a coordinate; report ϱ as the coordinate-independent quantity).
- **Decision 2 (user: run it):** the **separate-actor cascade experiment** — see §7.
- **§2/Q1 bimodality (all 3 reviewers' shared ask) — answered from EXISTING data:**
  `experiments/make_bimodality_figure.py`. Honest per-substrate result:
  - **Supply chain: BIMODAL** (Sarle coefficient 0.59/0.57 at w=0.65/0.75, above the 0.555
    threshold, peaks AT the crossing) → "collapse" is a genuine attractor.
  - **LBF: SMOOTH continuum** (coeff 0.32–0.42) → a threshold-defined crossing, not a regime.
  - MPE not checked (no interior-w rows) → we do NOT claim attractor for MPE (caught an
    overclaim where I'd initially written "supply chain and MPE").
  - Figure → supplement (`fig:s_bimodality`, `Figures/bimodality_supply_vs_lbf.png`);
    2-sentence honest statement in main Results.
- **Writing fixes:** SNR proposition proof compressed + labeled conjecture; "one-parameter
  law" → "descriptive summary/curve fit"; transfer-endpoint CIs added to Table 4;
  "four combinations" qualified as "three environments, PPO-family".

### Coherence pass (did the human read FOR the user)
Full end-to-end read + systematic checks. Found and fixed two real bugs:
- **MPE-attractor overclaim** (claimed attractor for MPE without evidence) → scoped to supply chain.
- **Broken forward-ref**: bimodality paragraph inserted between "measured next" and the
  horizon paragraph → changed "next" to "below".
Verified: 0 broken refs, 0 undefined cites, decomposition arithmetic all consistent,
terminology consistent, abstract↔body claims aligned, main↔supplement cross-refs aligned.

### Em-dash removal (final request)
Removed all 41 prose em dashes (37 main + 4 supplement), context-appropriately (parens /
colons / semicolons / commas). Kept legitimate en dashes (`actor--critic`, `10--90`).
Both files: 0 em dashes, compile clean.

---

## 7. The separate-actor cascade experiment — full history (RESOLVED)

**Purpose:** deconfound the minority-node cascade (requested by review rounds 2, 3, 4, 5).
Replace the ONE shared-parameter PPO actor with FIVE INDEPENDENT PPO actors (separate
parameters, each trained only on its own node's transitions, others acting via their own
current policies in a round-robin, no shared gradient). Test whether the position ranking
("downstream/proximity-to-demand dominates") found under the shared actor survives.

### Attempt 1 — job `129314[]`: HPC walltime bug, killed and never usable
Original 6h walltime was far too tight; the driver wrote output only once, at the very end
of training, so cells that didn't finish lost 100% of progress. 100 cells burned a full 6h
each for zero output before this was caught. **Fixed** with per-round checkpointing (all 5
actor models + round index saved every round; resume-on-restart) and doubled walltime to
12h. Also discovered mid-fix: this cluster's PBS **does not actually substitute `%I`** in
`-o`/`-e` paths (confirmed via other jobs' logs literally named `iql_mpe_%I.out`) — fixed by
redirecting `exec > logfile 2> errfile` manually inside the script using
`$PBS_ARRAY_INDEX`. Cancelled and resubmitted clean as `129320[]`.

### Attempt 2 — job `129320[]` (+ cleanup resubmits): completed, but a real design bug
Ran ~2 days wall-clock (500 cells, ~100–185 concurrent). Finished with 0 errors, but 8 cells
hit the (still tight for outliers) 12h wall; resubmitted from checkpoint (`129378`–`129380`),
all 500 cells landed clean. **Result at the time looked dramatic and wrong**: collapse was
almost entirely ABSENT under separate actors (most cells 0/20 seeds collapsed; median
profit +431.7, nowhere near the collapse floor), vs. the shared actor's collapse-by-default
at low `w`. Investigated rather than accepted at face value — root cause found:

**The bug:** SuperSuit's `pettingzoo_env_to_vec_env_v1` packs all 5 shared-actor agents into
ONE vectorized env, so `model.learn(total_timesteps=300_000)` there gives each INDIVIDUAL
agent only ≈60k REAL environment ticks (300k spread across 5 parallel agent-slots). The
separate-actor driver's `SingleAgentView` is a plain single-agent env, so its
`learn(total_timesteps=300_000)` gave each of the 5 independent actors a FULL 300k real
ticks — **5x more real per-agent training exposure** than agents got under the shared-actor
baseline. This confounded "no shared gradient" with "5x more practice," and is a far more
parsimonious explanation for the near-total absence of collapse than any real effect of
removing the shared gradient.

**Archived, not deleted:** full 500-cell v1 dataset preserved at
`results/expCascadeSep_5xoverexposed_v1/` (byte-verified complete copy) before touching
anything live.

### Attempt 3 — job `129394[]` (+ cleanup resubmits `129526`–`129530`): the real result
**Fix:** `experiments/exp_cascade_separate_actors.py` default `--timesteps` changed
300,000→60,000 (segment 10,000→2,000, same 30-round structure), matching the shared actor's
REAL per-agent exposure. Walltime cut 12h→3h (per-cell cost dropped ~5-7x with the fix).
Live `results/expCascadeSep/` cleared of the invalid v1 results first (after archive
verification) so `_is_complete()` wouldn't skip cells based on stale data.
- **Validated with a single timed test cell first** (job `129390`, run_idx=0, node_2/w=0.5/
  seed=42) before committing the full array: v1 gave this exact cell `collapsed=False`
  (mean_S=31.8, profit=+151.4); the corrected run gave `collapsed=True` (mean_S=0.0,
  profit=-128.1, the deterministic floor) — direction-of-fix confirmed before spending the
  full compute budget. Elapsed 4718s (~1h19m), consistent with the ~1h/cell estimate.
- Full array `129394[]` launched, ~4h wall-clock at ~90-185 concurrency; 12 cells hit the 3h
  wall at the tail (all with healthy checkpoints, rounds 22-28/30) and were resubmitted as 5
  small padded arrays; final dataset landed with **0 errors across all 500 cells**.

**FINAL RESULT (trustworthy, 500/500 cells, balanced 100/position, all CIs finite/sane):**

| Position | Shared `w_crit` | Rank | Separate `w_crit` | Rank |
|---|---|---|---|---|
| node_2 | 0.863 [.846,.880] | **1** | 0.498 [.363,.581] | **5** |
| node_3 | 0.800 [.780,.820] | 2 | 0.714 [.683,.745] | **1** |
| node_5 | 0.797 [.740,.857] | 3 | 0.538 [.474,.591] | 4 |
| node_6 | 0.726 [.682,.769] | 4 | 0.547 [.495,.594] | 3 |
| node_4 | 0.673 [.622,.720] | 5 | 0.559 [.502,.608] | **2** |

**This is a rank-inversion, not a reshuffle.** node_2 (strongest under shared actor) becomes
the WEAKEST under independent actors; node_4 (weakest) rises to 2nd. The coarse
retail-adjacent-vs-split-path grouping also fails to replicate (shared: mean 0.832 vs 0.700,
clear gap; separate: 0.606 vs 0.553, no longer cleanly separated, inverted by node_2 alone).

**Conclusion written into the paper: the position-specific ordering under the shared actor
is substantially a shared-gradient artifact, not evidence for a directed product-flow
mechanism.** The topological reading is withdrawn. Per explicit user instruction, this is
written as ONE continuous story (we always planned this second sweep to test the shared-
gradient confound before drawing a topological conclusion → here is what it found), not as
a pivot or a "we later discovered" framing. Four places in the main paper were touched to
keep this consistent (see §12), plus a new supplement subsection
(`sec:s_cascade_separate`, §4.6) with the exposure-matching methodology and a fuller table.

---

## 8. Complete file inventory (created/modified this session)

### New experiment code (`experiments/`)
- `exp_control_a_refpoint.py`, `analyze_control_a_refpoint.py`
- `exp_control_b_scalematch.py`, `analyze_control_b_scalematch.py`
- `exp_control_c_rewardunit.py`, `analyze_control_c_rewardunit.py`
- `reanalysis_controls.py` (Part-2), `make_bimodality_figure.py`
- `exp_cascade_separate_actors.py` (running)
- `exp_meltingpot_cleanup.py`, `analyze_meltingpot_cleanup.py` (Exp B, blocked)
- `exp_cleanup_wlambda.py`, `analyze_cleanup_wlambda.py` (Exp B retarget, not swept)

### New env code (`env/`)
- `meltingpot_cleanup_wrapper.py`, `meltingpot_collapse.py` (real MP, blocked)
- `cleanup_wlambda_collapse.py`

### Modified (shared)
- `agents/cpt_wrapper.py` — added reference_point / local_scale / global_scale /
  output_scale (all identity-default, backward-compat verified).

### PBS
- `run_control_{a,b,c}_*.pbs`, `run_control_{a,b}_pilot.pbs`,
  `run_cascade_separate_actors.pbs`, `run_meltingpot_cleanup*.pbs`, `run_cleanup_wlambda.pbs`

### Results / planning docs (repo root)
- `RESULTS_A.md`, `RESULTS_B.md`, `REANALYSIS.md`, `REVISION_PLAN.md`, this file.
- `/tmp/MASTER_RESULTS.csv` (frozen master, scratch location).
- Pre-registration anchors: `docs/expControlA_refpoint_figures/PREREGISTRATION_z0.md` +
  `refpoint_z0.json`; `docs/expControlB_scalematch_figures/PREREGISTRATION_rms.md` +
  `rms_output_scale.json`.

### Paper (`AuthorKit27/`)
- `AnonymousSubmission2027_3.tex`, `SupplementaryMaterial2027_3.tex` — the working revision
  (copies of `_2`, which are the reviewed originals, untouched).
- `references.bib` — added `demaddpg2020`, `packetrouting`, `mao2020reward`.
- `Figures/bimodality_supply_vs_lbf.png` — new.

### Cluster job IDs (all `sharc`)
- Control A pilot: 128821 (FAILED walltime), 128831 (OK). Control A array: 128837[].
- Control C array: 128838[]. Control B pilot: 128839 (failed on stale driver), 128854 (OK).
  Control B array: 128855[].
- Separate-actor cascade: **129314[] (RUNNING)**.

---

## 9. Key decisions & standing constraints (for future sessions)

- **No login-node compute — EVER.** qsub everything (memory `no-login-node-compute`).
- **Cascade stays supply-chain-only** unless a real 2nd flow env appears (memory
  `cascade-generalization-closed`); the separate-actor job is a DECONFOUND, not a new env.
- **Real Melting Pot is unrunnable here** (glibc 2.17 < dmlab2d's 2.31, no container).
- **Pre-registration recorded as written anchors, not git commits** (user's choice).
- **Nothing committed to git** this session; all work is in the working tree.
- **Incentive ratio is co-primary** with w_crit (Decision 1b).
- **Score trajectory 4→5→6;** target is a poster; the two things that would push to 7 are a
  clean separate-actor cascade result and (already done) the bimodality evidence.
- **Baselines never conflate:** 0.585/0.586 = curvature (α=β=0.88); 0.666 = structural null
  300k; 0.685 = linear null (α=β=1). The 0.585 vs 0.685 gap = the curvature effect.

---

## 11. Review round 4 (weak-accept-leaning-borderline) — response

New reviewer, sharper and more checkable than earlier rounds (every major claim verified
against the actual files before acting on it, not taken at face value):

- **ϱ never had a numerical value anywhere in the paper** (main text OR supplement) despite
  being claimed as "reported alongside each w_crit" — confirmed by grep, a real gap not a
  reviewer misreading. Computed it properly from Control C's existing reward-unit-sweep data
  (`experiments/compute_varrho.py`, new script, no new training): **ϱ is NOT fully invariant
  across reward-unit arms** (spread 0.16 at λ=1 vs. w_crit's own spread of 0.30 — roughly
  halves but does not eliminate the unit-dependence). Investigated whether an RMS-based
  (measured, not nominal-multiplier) definition of c_ℓ,c_g would rescue full invariance —
  dead end: Control C never logged pre-blend local/global reward magnitudes separately, and
  no model checkpoints were retained, so testing this would need a new training sweep, not
  reanalysis. Reported the honest partial result everywhere ϱ appears (abstract, contributions
  list, Results, Future Work), rather than the originally-claimed full invariance.
- **Cascade position values had no CIs anywhere in the main text.** Computed them
  (`docs/reanalysis_controls/cascade_position_cis.csv`) — confirmed the reviewer's exact
  prediction: the coarse retail-adjacent-vs-split-path grouping holds, but two specific
  pairwise comparisons (node_3 vs node_5, node_4 vs node_6) are NOT statistically
  distinguishable at n=20/position. Paper's claim rescoped to match what the CIs actually
  support.
- **Horizon-asymptote hedge**: the log-linear fit (w_crit = 2.44 − 0.138 ln T) taken literally
  predicts w_crit→0 near T≈4.8×10⁷ steps. Added one sentence stating this is local to the
  tested range (150k–900k, one order of magnitude), not an extrapolable law.
- **Multiplicity/multiple-comparisons statement** added, anchored to the pre-registration
  footnote: confirmatory (the primary λ=1→7 shift, the collapse label) vs. exploratory
  (control arms, position sweep, horizon points) explicitly distinguished; no family-wise
  correction claimed, each CI relied on individually.
- **Straw-null reordering**: Results now leads with the three-estimator (logistic/probit/
  isotonic) agreement as the stronger check, with the Firth-vs-flat-null test explicitly
  demoted to "the (much weaker) null."
- **Abstract's "four combinations" oversell**: reworded to name that two of the four are
  constructed/censored transfer settings (MPE wrapper-built conflict, LBF's ceiling effect),
  not four equally-native environments.

## 12. Review round 5 + user's own catches — arithmetic reconciliation and a real internal
    contradiction

- **A genuine pre-existing stale number, caught via user's own line-by-line check**:
  0.856 (cascade dose-response endpoint) vs. 0.863 (my reanalysis of the same data with the
  paper's own Firth estimator) — investigated fully; 0.863 is reproducible from the checked-in
  CSV, 0.856 was stale BEFORE this session touched the file. Propagated 0.585/0.854 (not
  0.586/0.852) through every instance across BOTH papers (Table 1, Table 3, cascade section,
  supplement's full λ-sweep table, MAPPO comparison, affine-law prediction list — the latter
  required a genuine model REFIT, not a find-replace, since it's calibrated FROM those
  endpoints: ρ 0.61→0.63, held-out RMSE 0.016→0.014).
- **A real internal contradiction the user caught directly**: Discussion claimed the
  curvature/loss-aversion opposite-direction decomposition "survives all three settings," but
  Results (two pages earlier) already stated MPE's curvature shift spans zero and LBF's runs
  the SAME direction as loss aversion. Rescoped the Discussion sentence to
  supply-chain-PPO-only with an explicit cross-reference to where the transfer section states
  the honest scope.
- Deltas in Table 1/3 (+0.266/+0.167) were ALSO stale relative to the corrected point
  estimates (should be +0.269/+0.169) — verified against the paper's own seed-paired
  bootstrap methodology (not naive subtraction) before fixing, confirming +0.269 was already
  correct elsewhere in the paper (the reference-point control comparison) and the table
  values were simply never updated.
- Fixed: two dangling "below" cross-references left over from the SNR-section removal (now
  "in the Supplement"/"(Supplement, a descriptive fit)"); abstract's lost effect-size numbers
  restored (+0.089/+0.189, the corrected value) after a user push-back that trimming too far
  removed signal a reviewer would want; two leftover "affine" mentions in the claim-ledger
  table reworded to "summary form" to match the section's removal from the main paper.
- **A visible `§??` submission error**, caught by the user reading the compiled PDF directly:
  `\S\ref{sec:s_setup}` pointed at a label that only exists in the SEPARATE compiled
  supplement document — cross-document `\ref` can never resolve, this always renders as `§??`
  in a split main+supplement submission. Every other cross-document reference in the paper
  correctly uses a literal `\S``Section Title''` string; this was the one place that used the
  wrong mechanism. Fixed; confirmed via direct scan of the rendered PDF text (0 instances of
  `??` anywhere).
- **Conclusion contradicted the cascade result**: "moves ... largely with the node nearest
  demand on a directed network" directly contradicted the just-added deconfound finding.
  Rewritten. Also fixed: "directed-flow minority finding" in the Introduction's contributions
  list (too strong given the deconfound), and Future Work's "a second directed-flow domain
  would make the cascade a tested generalization" (wrong framing — the experiment showed the
  cascade wasn't cleanly a directed-flow effect in the first place; reframed as "does
  minority-agent boundary dominance under shared learning transfer across networks and
  architectures").

## 13. The bimodality/attractor claim — tested and withdrawn (this session)

Reviewer skepticism ("a bimodality coefficient a hair above a rule-of-thumb, with no
uncertainty, is not evidence for a distinct regime") was investigated properly rather than
just hedged further. Wrote `experiments/bimodality_significance.py`: bootstrap 95% CI on
Sarle's coefficient + Hartigan's dip test, both substrates, all w, from existing CSVs (no new
training). **Result: NO confirmed attractor anywhere.**
- Supply chain: dip test never rejects unimodality (p≥0.30 at every w tested, n=20/cell); the
  bootstrap CI on Sarle's coefficient is wide and straddles the 0.555 rule-of-thumb at every w.
- LBF: raw dip test showed NOMINALLY SIGNIFICANT bimodality at high w (p<0.05) — investigated
  why, found it traces to a ceiling-effect artifact (7-8/40 runs saturating at the metric's
  cap); excluding ceiling-clipped runs removes the significance (p≥0.11).
- **"Attractor" language dropped entirely; "collapse" now described throughout as a
  "threshold-defined behavioral crossing," not a confirmed distinct regime.** This closes off
  what was flagged as the sharpest reviewer criticism across multiple review rounds. Value is
  defensive (closes an attack line), not a positive headline finding — correctly NOT listed
  as one of the three contributions in the Introduction.

## 14. Structural edits: SNR section removed, figures widened, abstract restored

- **The entire "SNR Account of the Sign" mechanism section removed from the main paper**,
  moved to supplement-only (per explicit user instruction, after confirming with the user that
  it is NOT pivotal to the paper — the paper's own prior framing already called it
  "Not Load-Bearing"). Kept: Proposition 1, its proof, and the claim-ledger table intact in
  the supplement (per user's earlier explicit instruction not to touch those even when
  compressing). Freed a full page (9→8) in the main paper; body content (through Conclusion)
  now finishes comfortably within page 7 with real margin, vs. repeatedly fighting the budget
  earlier in the session.
- **Two main-paper figures widened to full-page-width (`figure*`)**: Figure 2 (7-curve
  loss-aversion sweep) and Figure 3 (cross-environment generalization plot) were cramped at
  single-column width; both are now legible at 0.6-0.7\textwidth spanning both columns.
  (One failed attempt: regenerating Figure 2's source PNG at a wider aspect ratio made it
  WORSE at display width — user caught this immediately ("this looks ugly... go back to the
  previous one"), reverted cleanly, original 1125×720 image restored.)
- **Abstract rewritten twice this session, both times per direct user request**: first
  trimmed to match the user's own ACTUALLY-REGISTERED Triple-AI abstract (removing all
  specific numbers/environment names/λ values that had accumulated from individual reviewer
  patches) — user's explicit intent: "I don't want too many numbers... generic... skimmable."
  Then partially reversed after a reviewer note argued the trim went one step too far and
  lost the decomposition finding (the thing Discussion calls unprecedented) — restored the
  primary w_crit value+CI and the curvature-opposite-direction clause, kept everything else
  generic. ϱ was NOT restored to the abstract (correctly demoted from the contributions list
  earlier after being found only partially invariant).

## 15. Submission-readiness checks (this session, near the end)

- **A real, previously-flagged AAAI-27 compliance gap**: the Method footnote said code/data
  "released on acceptance," which AAAI-27's stated reproducibility policy explicitly says
  earns zero credit at review time (materials not INCLUDED are assumed unavailable). By the
  end of the session, `aaai2027_code_data_archive_curated.zip` existed and was verified: 0
  identity/path leaks (byte-level scan), includes the session's actual scripts
  (`run_cascade_separate_actors.pbs`, `reanalysis_controls.py`, correctly dated), and the
  supplement's text correctly references it as included, not promised.
- **`AnonymousSubmission2027_3_clean.pdf` + `SupplementaryMaterial2027_3_clean.pdf` +
  `ReproducibilityChecklist_clean.pdf` audited and confirmed submission-ready**: PDF metadata
  (Author/Creator/Title/Keywords/XMP) fully empty on both papers; zero identity/institution/
  path leaks; checklist's two "Type your response here" hits are both inside the template's
  own instructional example, not real unanswered questions (every actual question checked has
  a real answer); supplement PDF confirmed compiled from the FINAL `.tex` state (spot-checked
  that the cascade deconfound section §4.6 and its correct final numbers are present, not a
  stale pre-fix snapshot).
- **No poster-specific submission track exists at AAAI** — poster vs. oral is a
  presentation-format assignment made after acceptance, not a separate submission tier with
  its own bar. "Is this good enough for a poster" and "is this good enough to be accepted" are
  the same question in AAAI's process.

---

## 10. Open items / next actions (UPDATED — see §0a for the short version)

**Resolved since the original list (all four original items are now closed):**
1. ~~When 129314[] finishes~~ → superseded by 129320[]→129394[], see §7. Cascade result is
   IN the paper, written as one continuous story per user's explicit instruction.
2. Custom-Cleanup 4th substrate — still optional/not done; not revisited this session, no new
   information.
3. Rebuttal letter — not drafted this session; all 5 review rounds now map cleanly to the
   paper's actual current text if this becomes needed.
4. `_3` files ARE the ones submitted (as `_clean.pdf` variants) — page count reverified
   multiple times this session (currently 8 pages main, 11 supplement, unbounded).

**Actually open now:**
1. **Horizon sweep extension (900k→1.8M) was discussed as a cost estimate only, NOT run.**
   ~5h/cell (scaling linearly from the session's known 300k≈50min baseline), ~500 cell-hours
   for a full new horizon point at the existing 20-seed×5-w design, ~3-6h wall-clock at
   observed concurrency. If the horizon-asymptote hedge (§11) isn't judged sufficient by
   reviewers, this is the next lever, not yet pulled.
2. Confirm the exact archive filename attached on OpenReview matches what the supplement's
   text claims contains the code/data (`aaai2027_code_data_archive_curated.zip` was the one
   audited; verify this is the one actually uploaded, not an older sibling zip in the same
   directory — `aaai2027_code_data_archive.zip` without `_curated` also exists on disk).
3. User's own external review pass (mentioned as the planned last step before final
   submission logistics) — status unknown to this log; check before assuming it happened.
