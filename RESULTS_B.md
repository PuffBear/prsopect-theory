# RESULTS_B — Cleanup social dilemma (Experiment B)

## SUBSTRATE DECISION — read first

Experiment B was specified against the **real DeepMind Melting Pot** `clean_up`
substrate. That path is **blocked on this cluster** and could not be run:

- Real Melting Pot depends on the compiled **`dmlab2d`** engine. Its Linux wheels
  require **glibc ≥ 2.31**; the only wheels on PyPI are `manylinux_2_31`/`_2_35`.
- The HPC cluster is **CentOS 7 / glibc 2.17**, with **no container runtime**
  (no singularity / apptainer / docker) to supply a newer-glibc userspace.
- So `pip install dm-meltingpot` fails (`dmlab2d` reports zero compatible versions on
  both py3.11 and a fresh py3.10 env). Building `dmlab2d` from source on CentOS 7 is a
  Bazel + newer-GCC toolchain effort with high dead-end risk — the same binary wall this
  project's notes already record for SUMO/SMAC.

**Decision (per the human): run Experiment B on the in-repo Cleanup instead.** The
`env/cleanup_env.py` substrate is a pure-NumPy implementation of the **same
Leibo-et-al-2017 tragedy-of-the-commons** structure (harvest-vs-clean; local cost /
global benefit), runs on glibc 2.17 today, and carries the **identical** reward-blend +
CPT contract. The direction claim (does loss aversion raise `w_crit`?) is substrate-
transferable and is what this tests.

**Honest scoping:** this is the *lightweight, in-repo* Cleanup, **NOT** the DeepMind
standard benchmark. It answers the "public-good social-dilemma transfer" question but
does **not** close the "no standard-benchmark run" reviewer point with the DeepMind
suite specifically. The real-Melting-Pot driver (`env/meltingpot_cleanup_wrapper.py`,
`experiments/exp_meltingpot_cleanup.py`, GPU PBS) is kept in the tree as **future work**,
runnable on any cluster with glibc ≥ 2.31 or a container runtime.

### What actually runs (custom-Cleanup Experiment B)
- Driver `experiments/exp_cleanup_wlambda.py` — MAPPO (MlpPolicy + wider critic) on
  `env/cleanup_env.py`; CPU-only; runmap / pilot / dry-run / merge / checkpoint-resume.
- Label `env/cleanup_wlambda_collapse.py` — independently-frozen threshold JSON.
- Analysis `experiments/analyze_cleanup_wlambda.py` — same logistic + Firth + bootstrap.
- Job `run_cleanup_wlambda.pbs` — `-q cpu` array (0–1819).
- **Locally verified:** compiles, runmap (1820 rows), dry-run confirms the reward blend +
  CPT (e.g. w=0.95,λ=7: r_local=1.0, r_global=0.333 → r_blend=0.367 → r_cpt=0.414), and a
  tiny cell trains end-to-end.

---

## (Future work) Real Melting Pot `clean_up` — CNN-MAPPO, BUILT, blocked on this cluster

The section below documents the real-benchmark driver, kept for a glibc ≥ 2.31 / container
environment. It is **not** runnable here (see substrate decision above).

---

## Substrate

- Suite: **DeepMind Melting Pot** (`google-deepmind/meltingpot`), substrate
  **`clean_up`**. Player count = the substrate's own default (7 for `clean_up`), read
  from the Melting Pot config — not hardcoded.
- Learner: **CNN-MAPPO** — SB3 `CnnPolicy` (NatureCNN over per-agent egocentric RGB) +
  a wider centralised critic head (Yu et al. 2022), mirroring the MAPPO setup used on
  the MPE/LBF transfer drivers.
- Reward blend (identical contract to every other substrate; blend then transform):
  `r_i^PT = v( (1−w)·r_i^local + w·r^global )`, with `r_i^local` = agent i's raw apple
  harvest, `r^global` = mean per-agent harvest, and `v` the paper's CPT value function
  (`α=β=0.88`, `λ` swept). Cleaning is **unrewarded** — `w` carries the incentive.
- Files:
  - `env/meltingpot_cleanup_wrapper.py` — real `clean_up` → PettingZoo parallel + w-blend.
  - `env/meltingpot_collapse.py` — frozen collapse label.
  - `experiments/exp_meltingpot_cleanup.py` — driver (sweep / runmap / pilot / dry-run / merge).
  - `experiments/analyze_meltingpot_cleanup.py` — logistic + Firth + bootstrap analysis.
  - `run_meltingpot_cleanup.pbs` — GPU array job; `run_meltingpot_cleanup_dryrun.pbs` — pre-flight.

## Sweep grid (comparability)

- `w ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0}` —
  full-range 13-point grid. The standard supply-chain grid floors at 0.5, but the
  boundary can sit LOW on non-supply-chain substrates (the custom Cleanup uses
  `w≥0.10`; the MPE/LBF transfers needed a `w∈{0,0.1,0.2}` extension to locate `w_crit`
  at all), so we span 0.0–1.0 to keep the `λ=1` anchor informative and avoid an
  off-grid boundary, while retaining the standard upper points `{0.75,0.85,0.95}` +
  finer `{0.80,0.90}` for comparability.
- `λ ∈ {1,2,3,4,5,6,7}` — null CPT + six loss-aversion levels.
- **20 seeds/condition** (42–61), `α=β=0.88`.
- Total: 13 × 7 × 20 = **1820 cells** → array indices 0–1819, mapped by the committed
  `docs/expMeltingpot_cleanup_figures/runmap_B.csv`.

## Pre-registered collapse label — FREEZE BEFORE ANALYSIS

- Label: `collapsed ⇔ mean_collective_return < THRESHOLD`, where `collective_return` is
  the mean over eval episodes of the **total apples harvested by the team** (the direct,
  substrate-native signature of commons collapse: free-riding → waste → apple supply → 0).
- THRESHOLD is set from a small pilot that reads the collective-return **scale** under
  obvious free-riding (`w=0.5`) vs coordination (`w=0.95`) at `λ=1`
  (`exp_meltingpot_cleanup.py --pilot`). It is the midpoint of that scale — **not tuned
  to make a clean boundary.**
- Freeze procedure (do this before the full analysis):
  1. `python experiments/exp_meltingpot_cleanup.py --pilot` → read suggested threshold.
  2. Write `{"threshold": <value>}` to
     `docs/expMeltingpot_cleanup_figures/meltingpot_collapse_threshold.json`.
  3. `git commit` that JSON with a timestamped message, BEFORE running
     `analyze_meltingpot_cleanup.py`.
  4. Record the hash here and in the driver header.

  - **Pilot collective-return scale:** low-w `<FILL>` vs high-w `<FILL>`
  - **FROZEN THRESHOLD:** `<FILL>`
  - **FROZEN-LABEL COMMIT HASH:** `<FILL>`

The label module fails loud (raises) if no threshold is frozen, so a sweep can never
silently score collapse against an un-frozen label.

## Results (fill after the run)

Primary claim under test: **direction** — does `w_crit` rise with `λ`? (Magnitude and
functional form are expected to be substrate-specific; that is already the paper's stance.)

| λ | w_crit | CI_lo | CI_hi | Firth w_crit | Δw | slope | p | n |
|---|---|---|---|---|---|---|---|---|
| 1 | `<FILL>` | | | | | | | |
| … | | | | | | | | |
| 7 | `<FILL>` | | | | | | | |

- **Max \|MLE − Firth\| w_crit disagreement:** `<FILL>` (the paper reports this).
- **Transition width Δw vs the supply chain:** `<FILL>` (expected WIDER — partial
  observability + pixels give a looser transition; honest comparison per the brief).
- **Direction verdict:** `<FILL — PASS / PASS-direction / UNRESOLVED>` (printed by the
  analyzer; PASS = monotone or CI-separated endpoints upward).

## Honest-reporting protocol (do not deviate)

- If CIs are too wide to resolve the direction at 20 seeds (they span the grid), report
  the numbers **as-is** and propose more seeds, or fall back to `commons_harvest__open`
  (note the substitution) — do **not** pick thresholds to manufacture a boundary
  (brief §5/§6).
- If the result is **null or reversed**, that is a real finding: write it up honestly
  and stop; the human decides framing. Do not bury or retune.

## Run procedure (cluster)

1. `python experiments/exp_meltingpot_cleanup.py --write-runmap` (done; committed CSV).
2. `qsub run_meltingpot_cleanup_dryrun.pbs` — confirms GPU visibility (`nvidia-smi` +
   `torch.cuda.is_available()`), meltingpot import, wrapper/CPT/CSV, and measures a
   single-element wall-time. **Reset the full job's walltime from this before step 4.**
   - NOTE: `pbsnodes` does not advertise an `ngpus` resource on this cluster, so the
     dry-run's CUDA check is the ground truth for whether `-l ngpus=1` yields a GPU. If
     torch reports no CUDA, drop `:ngpus=1` and rely on `-q gpu` alone.
3. `--pilot` → freeze + commit the collapse threshold (record the hash above).
4. `qsub run_meltingpot_cleanup.pbs` — full 1820-element array (`%4` GPU concurrency cap;
   adjust to the granted allocation). Idempotent + resumable: completed elements skip,
   preempted ones resume from checkpoint.
5. `python experiments/exp_meltingpot_cleanup.py --merge` — concatenates per-element
   results and prints any MISSING indices for a targeted requeue.
6. `python experiments/analyze_meltingpot_cleanup.py` — logistic + Firth + bootstrap,
   figures, and this results table.

## Compute risk (flagged to human)

Pixels + CNN + N=7 + full `w×λ×seeds` is heavy and **GPU-bound**. The cluster's `gpu`
queue (compute3/compute4) is small — the array caps concurrency at `%4`; expect this to
dominate the compute. Walltime (10h/element) is a **padded guess**; the dry-run measures
the real single-run time — reset before the full launch. Checkpointing is on
(`checkpoints/expMeltingpot/run_${idx}.zip`) so preempted elements resume.
