#!/bin/bash
set -euo pipefail

# ============================================================
#  MODIFIED FILES
# ============================================================

git add docs/slurm/run_mappo_mpe.pbs
git commit -m "Update MAPPO-MPE PBS script for v2 production settings

Revises resource requests, module loads, and Python paths based on the
calibrated MPE configuration from the v2 calibration runs."

git add env/mpe_coop_nav.py
git commit -m "Harden MPE Cooperative Navigation wrapper for production

Fixes edge cases in observation/action space handling discovered during
the full 840-run production sweep. Ensures robust agent lifecycle
management when agents terminate mid-episode."

git add experiments/analyze_mpe.py
git commit -m "Update MPE combined analysis with MAPPO decomposition support

Extends the analysis to handle the MAPPO-MPE raw data, producing the
cross-algorithm comparison (PPO vs MAPPO) on Cooperative Navigation."

git add experiments/exp_mappo_mpe.py
git commit -m "Fix MAPPO-MPE experiment with corrected CTDE pipeline

Aligns the centralised-critic MAPPO with the production MPE wrapper,
fixes vectorisation chain, and ensures proper global-state observation
augmentation for the value function."

git add experiments/exp_mpe_calibrate.py
git commit -m "Update MPE calibration with v2 threshold refinement

Incorporates learnings from the initial calibration pilot to refine
the collapse threshold search for MPE Cooperative Navigation."

git add experiments/exp_ppo_mpe.py
git commit -m "Update PPO-MPE experiment for production robustness

Fixes remaining vectorisation issues from the initial 260-run buggy
sweep. Ensures correct action-space padding and true_reward extraction
from the CPTRewardWrapper info dict."

git add latex/main.tex
git commit -m "Update paper with reviewer response experiments and SNR validation

Adds the mean-field SNR validation section, Shapley credit assignment
results, entropy ablation, and heterogeneous cognitive profiles. Expands
the theoretical grounding with the signal-to-noise phase boundary
derivation. Updates all figure references and numerical claims."

git add latex/references.bib
git commit -m "Update bibliography with reviewer response citations

Adds BibTeX entries for Shapley value credit assignment, mean-field
game theory, entropy regularisation, and heterogeneous agent
references cited in the expanded reviewer response sections."

git add latex/references.tex
git commit -m "Update BibTeX source with new reviewer response references

Adds entries for Level-Based Foraging (Christianos et al.), mean-field
SNR theory, and other citations needed for the expanded experiments."

git add latex/main.aux latex/main.bbl latex/main.blg latex/main.log latex/main.out latex/main.pdf
git commit -m "Update LaTeX build artifacts from latest compilation

Refreshed aux, bbl, blg, log, out, and PDF reflecting the expanded paper
with reviewer response experiments, SNR validation, and new figures."

git add auto_commit.sh
git commit -m "Update auto_commit.sh with latest batch commit script

Refreshes the utility commit script to match the current file inventory."

# Deleted/stale log handling (none to delete this round)

# ============================================================
#  UNTRACKED FILES — new files
# ============================================================

# --- Core infrastructure ---
git add agents/mappo_policy.py
git commit -m "Add centralised-critic MAPPO policy for SB3 (CTDE)

Implements Multi-Agent PPO with a centralised value function that receives
global state (concatenated observations of all agents) while the actor
retains local observations only. Built as a custom SB3 feature extractor
with separate actor/critic network architectures."

git add env/global_state_wrapper.py
git commit -m "Add global-state observation augmentation wrapper for MAPPO

PettingZoo wrapper that concatenates all agents' observations into a
global state vector and appends it to each agent's local observation.
Enables the centralised critic in MAPPO to condition on the full
system state while maintaining decentralised execution."

git add env/shapley_reward_wrapper.py
git commit -m "Add Shapley value structural credit assignment wrapper (Reviewer Point 2)

Implements marginal-contribution reward decomposition for the multi-echelon
supply chain. Each agent's reward is its Shapley value — the average
marginal contribution across all agent orderings — replacing the simple
w-blended reward to test whether structural credit assignment eliminates
the lazy-agent phase transition."

git add env/lbf_wrapper.py
git commit -m "Add Level-Based Foraging (LBF) environment wrapper

PettingZoo-compatible wrapper for the LBF cooperative foraging environment.
Provides the same interface as the or-gym, MPE, and RWARE wrappers for
consistent integration with the CPT reward shaping stack."

# --- Reviewer response experiment scripts ---
git add experiments/REVIEWER_RESPONSE_EXPERIMENTS.md
git commit -m "Add reviewer response experiments documentation

Documents the three reviewer-response experiment drivers (Points 2-4):
Shapley credit assignment, heterogeneous cognitive profiles, and
exploration noise ablation. Describes the experimental design, hypotheses,
and expected outcomes for each."

git add experiments/exp_shapley.py
git commit -m "Add Shapley credit assignment experiment (Reviewer Point 2)

Tests whether replacing the w-blended reward with Shapley value
decomposition eliminates the lazy-agent phase transition. If the
transition persists under Shapley rewards, it confirms that the
collapse is structural rather than an artifact of reward blending."

git add experiments/exp_heterogeneous.py
git commit -m "Add heterogeneous cognitive profiles experiment (Reviewer Point 3)

Tests mixed-population scenarios where agents within the same team have
different CPT parameters (varying λ, α, β). Investigates whether
heterogeneous loss aversion creates asymmetric collapse patterns and
whether a single loss-averse agent can destabilise an otherwise
cooperative team."

git add experiments/exp_entropy_ablation.py
git commit -m "Add exploration noise / entropy ablation experiment (Reviewer Point 4)

Ablation study that sweeps the PPO entropy coefficient to disentangle
whether the phase transition is driven by CPT reward distortion or by
representational dynamics (entropy collapse). Tests the hypothesis that
loss aversion accelerates entropy collapse, creating the appearance of
a reward-driven phase transition."

git add experiments/exp_snr_validation.py
git commit -m "Add mean-field SNR validation experiment

Empirical validation of the theoretical signal-to-noise ratio prediction
for the phase boundary location. Runs 180 jobs (9 w-values × 20 seeds)
at 150k steps, recording per-step reward trajectories to compute the
empirical SNR crossing point and compare it against the predicted
w_crit = 0.666 from the mean-field theory."

git add experiments/analyze_snr.py
git commit -m "Add SNR validation analysis script

Processes the trajectory data from exp_snr_validation to compute empirical
signal-to-noise ratios, identify the SNR=1 crossing point, and compare
against the theoretical prediction. Produces the mean-field validation
figure for the paper."

# --- Calibration & pilot scripts ---
git add experiments/exp_mappo_mpe_calibrate.py
git commit -m "Add MAPPO-MPE calibration experiment

Pilot calibration for the MAPPO × MPE cell, determining the appropriate
collapse threshold for MAPPO on Cooperative Navigation before the full
production sweep."

git add experiments/analyze_mappo_mpe_calibrate.py
git commit -m "Add MAPPO-MPE calibration analysis script

Processes MAPPO-MPE pilot data to compute the calibrated collapse
threshold and writes it to the configuration for the production runs."

git add experiments/exp_lbf_calibrate.py
git commit -m "Add LBF calibration experiment

Pilot calibration for the Level-Based Foraging environment, sweeping
w at the extremes (0.0 and 1.0) to determine the collapse/cooperation
reward threshold before the full sweep."

git add experiments/analyze_lbf_calibrate.py
git commit -m "Add LBF calibration analysis script

Processes LBF pilot data to determine the empirical collapse threshold
for the Level-Based Foraging environment."

git add experiments/exp_rware_calibrate.py
git commit -m "Add RWARE calibration experiment

Pilot calibration for the Robot Warehouse environment, determining the
collapse threshold before the full w × λ sweep."

git add experiments/analyze_rware_calibrate.py
git commit -m "Add RWARE calibration analysis script

Processes RWARE pilot data to compute the calibrated collapse threshold
for the Robot Warehouse environment."

git add experiments/exp_ppo_lbf.py
git commit -m "Add PPO on Level-Based Foraging experiment script

Cross-environment replication of the collapse surface on LBF. Sweeps
w × λ with CPT reward shaping to test whether the w_crit(λ) relationship
generalises to cooperative foraging tasks."

# --- Smoke test / local runners ---
git add experiments/run_lbf_smoke_local.py
git commit -m "Add LBF local smoke test runner

Quick local validation script for the LBF environment wrapper and CPT
stack before submitting HPC jobs."

git add experiments/run_mappo_smoke_local.py
git commit -m "Add MAPPO local smoke test runner

Quick local validation for the MAPPO centralised-critic pipeline with
global state augmentation."

git add experiments/run_mpe_smoke_local.py
git commit -m "Add MPE local smoke test runner

Quick local validation for the MPE Cooperative Navigation wrapper
and vectorisation chain."

git add experiments/run_rware_smoke_local.py
git commit -m "Add RWARE local smoke test runner

Quick local validation for the Robot Warehouse wrapper and CPT stack
before submitting HPC jobs."

# --- PBS scripts ---
git add docs/slurm/analyze_lbf_calibrate_v2.pbs
git commit -m "Add PBS script for LBF calibration analysis (v2)

Batch script to run the LBF calibration analysis on the cluster after
the pilot sweep completes."

git add docs/slurm/analyze_mappo_mpe_calibrate.pbs
git commit -m "Add PBS script for MAPPO-MPE calibration analysis

Batch script to process MAPPO-MPE pilot data and determine the
calibrated collapse threshold."

git add docs/slurm/analyze_mpe_calibrate_v2.pbs
git commit -m "Add PBS script for MPE calibration analysis (v2)

Revised calibration analysis with updated threshold search parameters."

git add docs/slurm/analyze_rware_calibrate_v2.pbs
git commit -m "Add PBS script for RWARE calibration analysis (v2)

Batch script to process RWARE pilot data and determine the calibrated
collapse threshold."

git add docs/slurm/check_lbf_determinism.pbs
git commit -m "Add PBS script for LBF determinism check

Diagnostic job that verifies the LBF environment produces deterministic
results with fixed seeds, ensuring reproducibility of calibration."

git add docs/slurm/run_lbf_calibrate_v2.pbs
git commit -m "Add PBS script for LBF calibration pilot (v2)

Array job for the Level-Based Foraging calibration sweep at w=0.0
and w=1.0 with multiple seeds."

git add docs/slurm/run_mappo_mpe_calibrate.pbs
git commit -m "Add PBS script for MAPPO-MPE calibration pilot

Array job for the MAPPO × MPE calibration sweep to determine the
collapse threshold before the full production run."

git add docs/slurm/run_mappo_mpe_v2.pbs
git commit -m "Add PBS script for MAPPO-MPE production sweep (v2)

Revised array job for the full MAPPO × MPE sweep with calibrated
settings from the v2 pilot."

git add docs/slurm/run_mpe_calibrate_v2.pbs
git commit -m "Add PBS script for MPE calibration pilot (v2)

Revised calibration pilot with updated resource requests."

git add docs/slurm/run_ppo_lbf_v2.pbs
git commit -m "Add PBS script for PPO on LBF (v2)

Array job for the PPO × LBF cross-environment replication sweep."

git add docs/slurm/run_ppo_mpe_v2.pbs
git commit -m "Add PBS script for PPO-MPE production sweep (v2)

Revised array job with corrected vectorisation and calibrated settings."

git add docs/slurm/run_ppo_rware_v2.pbs
git commit -m "Add PBS script for PPO on RWARE (v2)

Array job for the PPO × RWARE cross-environment replication sweep."

git add docs/slurm/run_rware_calibrate_v2.pbs
git commit -m "Add PBS script for RWARE calibration pilot (v2)

Array job for the Robot Warehouse calibration sweep."

git add docs/slurm/run_snr_validation.pbs
git commit -m "Add PBS script for mean-field SNR validation

Array job (0-179) for the signal-to-noise ratio experiment that
validates the theoretical phase boundary prediction."

# --- Experiment data & figures ---
git add docs/expMAPPO_mpe_figures/mappo_mpe_raw_data.csv
git commit -m "Add MAPPO-MPE production raw data

Contains the tabular results from the MAPPO × MPE Cooperative Navigation
sweep for the cross-algorithm comparison."

git add docs/expMAPPO_mpe_figures/mappo_mpe_raw_data.csv.corrupt_backup
git commit -m "Archive corrupt MAPPO-MPE data backup for provenance

Preserves the corrupt initial run data for reproducibility audit trail.
The production data in mappo_mpe_raw_data.csv supersedes this."

git add docs/expMPE_combined_figures/mappo_mpe_decomposition.md
git commit -m "Add MAPPO-MPE decomposition results summary

Documents the MAPPO vs PPO comparison on MPE, including per-λ w_crit
estimates and the algorithmic sensitivity analysis."

git add docs/expPPO_lambda_sweep_figures/affine_law_7pt.png
git commit -m "Add affine scaling law figure for the 7-point lambda sweep

Plots the fitted affine relationship w_crit = a + b·λ across the full
7-point sweep, demonstrating the linear scaling law for loss aversion's
effect on the collapse boundary."

git add docs/expSNR_figures/expSNR_raw_data.csv
git commit -m "Add SNR validation raw data

Contains the per-run summary statistics from the mean-field SNR
validation experiment (180 runs: 9 w × 20 seeds)."

git add docs/expSNR_figures/trajectories/
git commit -m "Add SNR validation trajectory data (9 representative runs)

Per-step reward trajectories stored as compressed .npz files, used to
compute the empirical signal-to-noise ratio crossing point for
comparison against the theoretical prediction."

git add docs/meanfield_figures/fig_meanfield_validation.png
git commit -m "Add mean-field validation figure

Visualises the empirical SNR crossing against the theoretical prediction
w_crit = 0.666, providing the key evidence for the mean-field theory
section of the paper."

# --- Root-level PBS scripts ---
git add run_entropy.pbs
git commit -m "Add PBS script for entropy ablation experiment (Reviewer Point 4)

Array job for the entropy coefficient sweep that disentangles CPT reward
distortion from representational dynamics (entropy collapse)."

git add run_heterogeneous.pbs
git commit -m "Add PBS script for heterogeneous profiles experiment (Reviewer Point 3)

Array job for the mixed-population experiment testing asymmetric collapse
patterns with heterogeneous CPT parameters."

git add run_shapley.pbs
git commit -m "Add PBS script for Shapley credit assignment experiment (Reviewer Point 2)

Array job for the structural credit assignment experiment testing whether
Shapley value rewards eliminate the phase transition."

git add run_shapley_smoke.pbs
git commit -m "Add PBS script for Shapley experiment smoke test

Quick validation job to verify the Shapley wrapper and experiment
pipeline before launching the full sweep."

git add run_snr.pbs
git commit -m "Add PBS script for mean-field SNR validation (180 runs)

Array job (0-179) for the signal-to-noise ratio experiment. Uses 150k
training steps with a 60k-step observation window to compute empirical
SNR at each w-value."

git add run_snr_analyze.pbs
git commit -m "Add PBS script for SNR validation analysis

Batch script to run analyze_snr.py after the SNR sweep completes,
producing the mean-field validation figure."

git add run_snr_smoke.pbs
git commit -m "Add PBS script for SNR experiment smoke test

Quick validation job to verify the SNR trajectory recording pipeline
before launching the full 180-run sweep."

# --- LaTeX variant/draft files ---
git add latex/main_nomt.tex latex/main_nomt.aux latex/main_nomt.log latex/main_nomt.out latex/main_nomt.pdf
git commit -m "Add paper variant: main_nomt (no margin TODO notes)

Clean version of the paper with margin TODO annotations removed for
reviewer submission. Includes compiled PDF and build artifacts."

git add latex/k.tex latex/k.aux latex/k.log latex/k.out
git commit -m "Add paper draft variant k with experimental edits

Working draft exploring alternative section organisation for the
results presentation. Build artifacts included."

git add latex/mm.tex latex/mm.aux latex/mm.log latex/mm.out latex/mm_refs.bib
git commit -m "Add paper draft variant mm with expanded methods section

Working draft with expanded mathematical methods and mean-field theory
derivation. Includes variant-specific bibliography."

git add latex/m_chk.tex latex/m_chk.aux latex/m_chk.log latex/m_chk.out
git commit -m "Add paper cross-check variant m_chk

Variant used for systematic numerical cross-checking of all claims
against raw experimental data."

git add latex/v.tex latex/v.aux latex/v.log latex/v.out
git commit -m "Add paper draft variant v with visualisation experiments

Working draft exploring alternative figure presentations and layout
for the expanded experimental results."

git add latex/references_bak.tex
git commit -m "Add references.tex backup before reviewer response edits

Snapshot of the bibliography before adding reviewer-response citations,
preserved for diff auditing."

# --- 3D plot script (already committed in analyze but exists separately) ---
# experiments/plot_3d_w_lambda_T.py was already committed in previous round

echo "=== All commits done. Pushing... ==="
git push
