# Experiment Status — AAAI revision (as of 2026-06-16)

HPC: `ssh <user>@10.1.4.95`, project dir `~/prospect-theory`, scheduler **PBS/Torque**
(`qsub`, `-J` arrays, `-q cpu`, `module load compiler/anaconda3`, `conda activate marl`).

## Status matrix

| Experiment | Driver | Runs | Data? | Scheduler | Status |
|---|---|---|---|---|---|
| A @150k / @300k (structural boundary) | exp1 / phase | 360 | ✅ | — | **done** |
| C (basin control) | — | 144 | ✅ | — | **done** |
| N (null baseline) | exp_null_model | 180 | ✅ | — | **done** |
| B-Reduced (λ∈{1,3,7}) | exp_b_reduced | 300 | ✅ | — | **done** |
| λ-sweep fill-in (λ∈{2,4,5,6}) | exp_ppo_lambda_sweep | 400 | ✅ | — | **done** |
| T (600k metastability) | exp_T_horizon | 140 | ✅ | — | **done** |
| MPE/MAPPO {null,curv,λ3} | exp_mappo_mpe | 330 | ✅ | — | **done** (curvature replicates; λ-direction unresolved) |
| SNR validation | exp_snr_validation | ~180 | ⚠️ partial (10 rows) | run_snr.pbs | **check/rerun** |
| **Shapley credit (Pt 2)** | exp_shapley | 80 | ❌ | run_shapley.pbs | **queued (running)** |
| **Heterogeneous (Pt 3)** | exp_heterogeneous | 400 | ❌ | run_heterogeneous.pbs | **queued** |
| **Entropy ablation (Pt 4)** | exp_entropy_ablation | 400 | ❌ | run_entropy.pbs | **queued** |
| **MAPPO on or-gym (2nd algo)** | exp_mappo_orgym | 700 | ❌ | run_mappo_orgym.pbs | **FIXED + PREPARED — wiring-check then submit** |
| MPE full λ sweep (2nd env) | (your queued job) | — | ❌ | — | **queued by you** |
| **QMIX · supply (value-based, Tier 1)** | exp_qmix_orgym --algo qmix | 800 | ❌ | run_qmix_orgym(.pbs/_null.pbs) | **BUILT — smoke then submit** |
| **IQL · supply (Tier 2)** | exp_qmix_orgym --algo iql | 800 | ❌ | run_iql_orgym(.pbs/_null.pbs) | **BUILT** |
| **QMIX · MPE (Tier 2)** | exp_qmix_mpe --algo qmix | 880 | ❌ | run_qmix_mpe(.pbs/_null.pbs) | **BUILT** |
| **IQL · MPE (Tier 3)** | exp_qmix_mpe --algo iql | 880 | ❌ | run_iql_mpe(.pbs/_null.pbs) | **BUILT** |

> **Value-based 2×2 (VALUE_BASED_EXPERIMENT_PLAN.md):** self-contained QMIX
> (monotonic mixer) + IQL on the existing SB3/SuperSuit stack — NOT EPyMARL (not
> installed locally or on HPC). Supply gets a `Discrete(K=11)` base-stock shell;
> MPE is natively `Discrete(5)`. CSV schema mirrors exp_mappo_orgym so
> analyze_firth_lambda_sweep ingests it. Three local PASS gates before any qsub:
> `--wiring_check` (mixer monotone + reads joint state; IQL has none),
> `run_qmix_toy_sanity.py` (known-good 2-agent coordination matrix),
> `run_qmix_smoke_local.py` (20k-step real run). Supply budget 300k env steps,
> MPE 4M. PBS lives in `docs/slurm/` (8 files: sweep+null × 4 cells). Each cell =
> 700 sweep (λ∈1..7, curvature = the λ=1 point) + 100/110 null (α=β=1, λ=1).

> **⚠️ FIX (this session): `exp_mappo_orgym.py` was NOT real MAPPO.** The previous
> version fed the critic only local observations (a renamed wider MLP, shared
> actor/critic) — i.e. it was IPPO/PPO mislabeled as a centralized critic, and
> would have reproduced the PPO result while claiming MAPPO. Rewritten to use the
> validated `GlobalStateWrapper` + `CentralizedCriticPolicy` (the same machinery
> as the MPE path): actor reads the local slice, critic reads the joint state.
> A built-in `--wiring_check` asserts the critic is real before you submit.

## What I prepared this session (ready to run)

- `experiments/exp_mappo_mpe_highlambda.py` — high-λ MPE driver (syntax-checked, imports verified).
- `run_mappo_mpe_highlambda.pbs` — PBS array 0–219, 4M steps.
- `run_mappo_orgym.pbs` — refreshed to current PBS convention, array 0–699.
- `experiments/analyze_mpe_decomposition.py` — 5-condition MPE decomposition (validated on existing data: curvature −0.067, λ3 −0.022, both CIs currently span 0).
- `experiments/analyze_reviewer_response.py` — Shapley / heterogeneous / entropy analyzers (degrade gracefully until data arrives).
- `experiments/analyze_firth_lambda_sweep.py` — Firth penalized fits for the supply-chain λ law (**already run**; reproduces Table 2 slopes/p).
- `submit_pending.sh`, `sync_results.sh` — submission + rsync helpers.

## Run order (copy/paste on the login node)

```bash
# 0) sync code up (interactive; captcha + password). No trailing slash on the
#    source dirs so their names are preserved under ~/prospect-theory/.
rsync -avz experiments env agents utils *.pbs <you>@10.1.4.95:~/prospect-theory/

# 1) VALIDATE the centralized critic is real (no training, ~10s)
python3 experiments/exp_mappo_orgym.py --wiring_check     # must print WIRING CHECK PASS

# 2) smoke-test one run
python3 experiments/exp_mappo_orgym.py --run_idx 0 --timesteps 20000 --out /tmp/smoke_orgym.csv

# 3) submit the array
qsub run_mappo_orgym.pbs

# 4) watch
qstat -tu $USER

# 5) when finished, pull + analyze
rsync -avz <you>@10.1.4.95:~/prospect-theory/docs/ ./docs/
python3 experiments/analyze_mappo_orgym.py    # w_crit(lambda) MAPPO vs PPO + figures
```

## Why these two are the priority (from the review)

- **MAPPO on or-gym** answers "single-algorithm": does the loss-aversion rightward
  shift in `w_crit` reproduce under a centralised-critic learner on the same env?
- **MPE high-λ** answers "single-environment for the headline effect": MPE so far
  only replicates *curvature*; λ was never swept past 3, and the null boundary sits
  at w≈1 (no headroom). High λ + 4M-step budget is the test of whether loss aversion
  impedes coordination in a *second* environment. The currently-queued `mappo_mpe`
  job does NOT cover this (same three conditions).

## Open (needs your judgment, not compute)
Novelty-framing tightening · collapse the 3-baseline narrative to one · final body
trim under 7 pp · swap to official AAAI-2027 `.sty` when released · verify the SNR
sweep (only ~10 rows present vs ~180 expected).
