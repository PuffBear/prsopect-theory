#!/bin/bash
set -euo pipefail

# ============================================================
#  MODIFIED FILES — existing files with updates
# ============================================================

# --- Slurm/PBS scripts (modified) ---
git add docs/slurm/run_expB_reduced.sh
git commit -m "Update Exp B-Reduced Slurm script with revised PBS directives
Adjusts job configuration for the reduced behavioral surface experiment (5 w × 3 λ × 20 seeds = 300 runs)."

git add docs/slurm/run_expN.sh
git commit -m "Update Exp N Slurm script with revised PBS directives
Adjusts job configuration for the null model baseline experiment (9 w × 20 seeds = 180 runs)."

git add docs/slurm/run_expT.sh
git commit -m "Update Exp T Slurm script with revised PBS directives
Adjusts job configuration for the time-horizon metastability experiment."

# --- Analysis scripts (modified) ---
git add experiments/analyze_behavioral_excess.py
git commit -m "Update behavioral excess decomposition analysis
Refines the three-way null-model decomposition that isolates curvature vs loss aversion effects on w_crit. Produces fig2_behavioral_decomposition.png and the excess bar chart."

git add experiments/analyze_expA.py
git commit -m "Update Exp A analysis with refined logistic fitting
Improves the structural phase boundary analysis for the w-sweep at 150k and 300k steps."

git add experiments/analyze_expB.py
git commit -m "Update Exp B analysis script
Refines analysis of the full behavioral surface experiment across w × λ conditions."

git add experiments/analyze_expB_reduced.py
git commit -m "Update Exp B-Reduced analysis with improved bootstrap CI estimation
Refines the per-λ logistic fits, w_crit bootstrap CIs, and the behavioral claim CI-overlap test for the reduced w × λ grid."

git add experiments/analyze_expN.py
git commit -m "Update Exp N null model analysis with reusable statistical functions
Provides the canonical implementations of wilson_ci, fit_logistic, bootstrap_wcrit, _coerce, per_w_table, and curve that are imported by all downstream analysis scripts."

git add experiments/analyze_expT.py
git commit -m "Update Exp T time-horizon analysis
Refines the collapse heatmap generation and metastability analysis across the (w, T) parameter space."

# --- Experiment scripts (modified) ---
git add experiments/exp0_convergence.py
git commit -m "Update Exp 0 convergence baseline script
Refines the initial convergence validation experiment that establishes training stability on the or-gym substrate."

git add experiments/exp0b_horizon_meta.py
git commit -m "Update Exp 0b horizon metastability script
Refines the matched-condition (α=β=1) 600k experiment used to bound the pure horizon effect on the collapse boundary."

git add experiments/expA_interior_w.py
git commit -m "Update Exp A interior-w sweep with canonical env builder and eval
Provides the reference _build_env, evaluate_and_extract_metrics, is_collapsed, and MetricsCallback imported by all downstream experiment scripts."

git add experiments/expB_run.py
git commit -m "Update Exp B runner script
Refines the full behavioral surface experiment runner."

git add experiments/expC_basin.py
git commit -m "Update Exp C basin control script
Refines the initialization sensitivity experiment that rules out starting conditions as a driver of collapse outcomes."

git add experiments/exp_T_horizon.py
git commit -m "Update Exp T horizon execution script
Refines the 600k-step time-horizon experiment to investigate metastability of the lazy-agent attractor."

git add experiments/exp_b_reduced.py
git commit -m "Update Exp B-Reduced execution script
Refines the reduced w × λ collapse surface experiment (5 w × 3 λ × 20 seeds) with CPT curvature α=β=0.88."

git add experiments/exp_null_model.py
git commit -m "Update Exp N null model execution script
Refines the v(x)=x baseline (α=β=1, λ=1) that provides the structural null for behavioral decomposition."

git add experiments/phase3_6_replication.py
git commit -m "Update phase 3.6 replication script
Refines the mid-phase replication validation for consistency checking."

git add experiments/phase3_7_verification.py
git commit -m "Update phase 3.7 verification script
Refines the verification experiment with expanded diagnostics."

git add experiments/phase3_8_incentives.py
git commit -m "Update phase 3.8 incentive structure experiment
Refines the reward incentive investigation for the supply chain substrate."

git add experiments/phase3_9_leadtime.py
git commit -m "Update phase 3.9 lead time experiment
Refines the lead time sensitivity analysis on the multi-echelon network."

git add experiments/phase3_validation.py
git commit -m "Update phase 3 validation script
Refines the phase 3 validation runner with improved diagnostics."

git add experiments/phase4_pt_replication.py
git commit -m "Update phase 4 prospect theory replication script
Refines the PT replication experiment and the mean_S evaluation methodology that calibrates the collapse label."

git add experiments/phase6_behavioral_claim.py
git commit -m "Update phase 6 behavioral claim experiment
Refines the formal behavioral claim test with pre-registered collapse definitions."

# --- Utility scripts (modified) ---
git add utils/plot_phase1.py
git commit -m "Update phase 1 plotting utility
Refines the visualization for phase 1 experimental results."

git add utils/plot_phase2.py
git commit -m "Update phase 2 plotting utility
Refines the visualization for phase 2 divergence analysis."

git add utils/plot_phase2a.py
git commit -m "Update phase 2a plotting utility
Refines the visualization for phase 2a replication analysis."

git add utils/plot_phase3_5_success.py
git commit -m "Update phase 3.5 success plotting utility
Refines the visualization for phase 3.5 challenge results."

# ============================================================
#  UNTRACKED FILES — new files
# ============================================================

# --- Exp 0b extended data ---
git add docs/exp0b_extended/exp0b_600k_raw_data.csv
git commit -m "Add Exp 0b extended 600k raw data (α=β=1 matched condition)
Contains the matched-condition 600k training data used to bound the pure horizon effect on w_crit, separating it from the curvature confound in Exp T."

git add docs/exp0b_extended/collapse_heatmap_matched.png
git commit -m "Add matched-condition collapse heatmap figure
Visualises P(collapse) in (w, T) space under α=β=1 at 600k, confirming metastability holds without curvature inflation."

git add docs/exp0b_extended/metastability_lineplot.png
git commit -m "Add metastability line plot figure
Shows the monotone decrease of collapse probability with training horizon under matched α=β=1 conditions."

# --- Exp T figures ---
git add docs/expT_figures/expT_raw_data.csv
git commit -m "Add Exp T raw data
Contains the tabular results from the time-horizon experiment (α=β=0.88, 600k steps) used for the collapse heatmap."

git add docs/expT_figures/collapse_heatmap.png
git commit -m "Add Exp T collapse heatmap figure
Visualises the P(collapse) surface in (w, T) space — the paper's Figure 1 showing the metastability result."

git add docs/expT_figures/expT_results.md
git commit -m "Add Exp T results summary
Documents the key findings: the collapse boundary slides left with training horizon, confirming the lazy-agent attractor is metastable."

git add docs/expT_figures/wcrit_vs_horizon.png
git commit -m "Add w_crit vs horizon figure
Plots the estimated critical centralization threshold as a function of training steps."

# --- Cross-check doc ---
git add docs/main_tex_crosscheck_v1.md
git commit -m "Add main.tex cross-check audit document v1
Documents the systematic verification of all numerical claims in the paper against raw experimental data, flagging inconsistencies and corrections applied."

# --- New PBS/Slurm scripts ---
git add docs/slurm/run_exp1.pbs
git commit -m "Add PBS script for Exp 1 (null 600k matched condition)
Batch script to run the matched α=β=1 600k experiment on the cluster for bounding the pure horizon effect."

git add docs/slurm/run_mappo_mpe.pbs
git commit -m "Add PBS script for MAPPO on MPE Cooperative Navigation
Array job (0-839) for the cross-algorithm × cross-environment replication: MAPPO on simple_spread_v3."

git add docs/slurm/run_mappo_orgym.pbs
git commit -m "Add PBS script for MAPPO on or-gym
Array job (0-699) for MAPPO on NetworkManagement-v1 with full λ ∈ {1,...,7} sweep."

git add docs/slurm/run_mpe_calibrate.pbs
git commit -m "Add PBS script for MPE collapse threshold calibration
Pilot job to calibrate the MPE collapse reward threshold before running full sweeps."

git add docs/slurm/run_ppo_lambda_sweep.pbs
git commit -m "Add PBS script for PPO extended lambda sweep
Array job (0-399) to fill in λ ∈ {2,4,5,6} on or-gym for the complete 7-point w_crit(λ) curve."

git add docs/slurm/run_ppo_mpe.pbs
git commit -m "Add PBS script for PPO on MPE Cooperative Navigation
Array job (0-839) for cross-environment replication of the collapse surface on simple_spread_v3."

# --- New environment wrapper ---
git add env/mpe_coop_nav.py
git commit -m "Add MPE Cooperative Navigation environment wrapper
Factory function and collapse definition for simple_spread_v3 with try/except import chain handling the PettingZoo MPE package split across versions."

# --- env_manifest ---
git add env_manifest.txt
git commit -m "Add environment manifest file
Documents the Python package versions and environment configuration used across experiments."

# --- New experiment scripts ---
git add experiments/exp1_null_600k.py
git commit -m "Add Exp 1 null 600k execution script
Runs the matched-condition (α=β=1, λ=1) 600k experiment to isolate the pure horizon effect on w_crit from the curvature confound."

git add experiments/exp_mappo_orgym.py
git commit -m "Add MAPPO on or-gym experiment script
MAPPO (PPO with centralised critic) on NetworkManagement-v1 with full λ ∈ {1,...,7} × w sweep. 700 runs as PBS array job."

git add experiments/exp_ppo_lambda_sweep.py
git commit -m "Add PPO extended lambda sweep experiment script
Fills in λ ∈ {2,4,5,6} on or-gym to complement the existing {1,3,7} data for a complete 7-point w_crit(λ) monotonicity curve."

git add experiments/exp_ppo_mpe.py
git commit -m "Add PPO on MPE Cooperative Navigation experiment script
Cross-environment replication of the collapse surface on simple_spread_v3 with full λ × w sweep. 840 runs."

git add experiments/exp_mappo_mpe.py
git commit -m "Add MAPPO on MPE Cooperative Navigation experiment script
Completes the 2×2 factorial {PPO, MAPPO} × {or-gym, MPE}. 840 runs as PBS array job."

git add experiments/exp_mpe_calibrate.py
git commit -m "Add MPE collapse threshold calibration script
Pilot experiment to determine the appropriate collapse reward threshold for MPE before running full sweeps."

# --- New analysis scripts ---
git add experiments/analyze_exp1_heatmap.py
git commit -m "Add Exp 1 heatmap analysis script
Generates the matched-condition collapse heatmap from Exp 0b/1 data for the metastability figure."

git add experiments/analyze_exp1_lineplot.py
git commit -m "Add Exp 1 line plot analysis script
Produces the metastability line plot showing collapse probability vs training horizon under matched conditions."

git add experiments/analyze_grand_comparison.py
git commit -m "Add grand cross-environment cross-algorithm comparison analysis
Produces the master 2×2 factorial figure {PPO, MAPPO} × {or-gym, MPE} and summary table of w_crit across all conditions."

git add experiments/analyze_mappo_orgym.py
git commit -m "Add MAPPO on or-gym analysis script
Produces MAPPO collapse curves by λ and PPO vs MAPPO w_crit comparison on the or-gym substrate."

git add experiments/analyze_mpe.py
git commit -m "Add MPE combined analysis script
Produces per-algorithm collapse curves and PPO vs MAPPO w_crit comparison on MPE Cooperative Navigation."

git add experiments/analyze_mpe_calibrate.py
git commit -m "Add MPE calibration analysis script
Processes pilot data to determine and validate the MPE collapse threshold."

git add experiments/analyze_ppo_lambda_sweep.py
git commit -m "Add PPO full lambda sweep analysis script
Merges existing {1,3,7} data with new {2,4,5,6} data to produce the complete 7-point w_crit(λ) curve with monotonicity test."

# --- Smoke test ---
git add fix_and_smoke_test.py
git commit -m "Add fix and smoke test script
Quick validation script to verify environment setup and wrapper chain before launching cluster jobs."

# --- LaTeX files ---
git add latex/main.tex
git commit -m "Add paper main.tex with inline figures and updated structure
Complete LaTeX source for the paper. Figures moved inline next to their referencing sections, acknowledgements commented out, clearpage before references. All numerical claims cross-checked against experimental data."

git add latex/references.tex
git commit -m "Update references.tex with verified citations
All BibTeX entries cross-checked against Google Scholar, arXiv, CrossRef, and JMLR. Corrected author lists, DOIs, arXiv IDs, and entry types. Flagged zhang2026lcerd as unverifiable."

git add latex/references.bib
git commit -m "Add references.bib compiled bibliography
Compiled BibTeX database for the paper bibliography."

git add latex/ppt.tex
git commit -m "Add presentation slides (ppt.tex)
Beamer slides explaining the paper's findings in accessible language: the phase transition, metastability, and opposite effects of probability weighting vs loss aversion on coordination."

git add latex/references_table.csv
git commit -m "Add references cross-check table
CSV table used during the systematic verification of all bibliography entries against primary sources."

git add latex/references_table1.csv
git commit -m "Add supplementary references table
Additional cross-reference data for bibliography verification."

git add latex/main.pdf
git commit -m "Add compiled paper PDF
Latest compiled version of the paper with all figures, references, and cross-checked numerical claims."

git add latex/main.aux latex/main.bbl latex/main.blg latex/main.log latex/main.out
git commit -m "Add LaTeX auxiliary build files
Compilation artifacts (aux, bbl, blg, log, out) from the latest pdflatex + bibtex build."

git add latex/bib.log latex/p1.log latex/p2.log latex/p3.log
git commit -m "Add LaTeX build pass logs
Individual compilation pass logs for debugging bibliography and cross-reference resolution."

# Check if latex/docs exists and has content
if [ -d "latex/docs" ] && [ "$(ls -A latex/docs 2>/dev/null)" ]; then
    git add latex/docs/
    git commit -m "Add latex/docs directory with supporting materials
Contains supplementary documentation and figure source files referenced during paper compilation."
fi

# ============================================================
#  PUSH
# ============================================================
git push
